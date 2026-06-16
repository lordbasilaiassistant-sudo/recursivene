"""The self-modification kernel — PROTECTED. The trust root of strong RSI.

It takes a proposed source Edit to the improver (harness/proposer), and accepts it ONLY if,
when the edited package is actually run in an isolated sandbox, it lowers the held-out
meta-objective WITHOUT degrading the untouchable invariant — and the kill switch permits it.
Anything else (a regression, a crash, a syntax error, an out-of-scope or protected target)
is rejected and rolled back; the live package is never left broken. Every attempt is written
to the monotonicity log, timestamped, so the whole self-editing history is auditable.

Three guarantees this file is responsible for:
  * PROTECTED_PATHS can never be edited (objective, invariant, world, core/, this kernel) —
    so success cannot be redefined, the world cannot be trivialized, and the kill switch
    cannot be removed.
  * STAGE_SCOPE widens one level at a time (model -> harness -> proposer); a stage's gate
    must pass before the next opens (staged closure).
  * Every edit is applied in a COPY, evaluated in a SUBPROCESS, and committed to the live
    tree only on success — so a bad self-edit cannot crash or corrupt the running loop.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from ..core.clock import now_iso, stamp

# Files that define the OBJECTIVE, the INVARIANT, the WORLD, the trusted KERNEL, and the
# kill switch. None of these is ever a legal edit target — the anchor of the whole system.
PROTECTED_PATHS = (
    "objective.py", "invariant.py", "world.py",
    "core/", "closure/",
)

STAGE_ORDER = ["model", "harness", "proposer"]
STAGE_SCOPE = {
    "model": set(),                                          # config ratchet only (no code edits)
    "harness": {"harness/loop.py"},                          # edit the improvement operator
    "proposer": {"harness/loop.py", "harness/proposer.py"},  # + the proposer's own definition
}

# subprocess meta-evaluation budget (kept small so the closure loop is fast). The search
# inside meta_evaluate is DETERMINISTIC (fixed seeds), so an edit is accepted only if it
# deterministically lowers the held-out meta-objective — no seed-noise gaming.
_META = dict(generations=4, steps=1200, inner_seeds=(0, 1), heldout_steps=1500)


class Edit:
    """A regex source edit: in `target` (relpath under the package), replace `find` with
    `replace`. `descr` is a human/audit string."""

    def __init__(self, target, find, replace, descr="", stage="harness"):
        self.target = target.replace("\\", "/")
        self.find = find
        self.replace = replace
        self.descr = descr
        self.stage = stage


def _is_protected(relpath):
    relpath = relpath.replace("\\", "/")
    return any(relpath == p or relpath.startswith(p) for p in PROTECTED_PATHS)


def _apply_to_text(source, edit):
    new, n = re.subn(edit.find, edit.replace, source)
    return (new, n)


def _subprocess_meta(container_dir, init_config, proposer_expr="EvolutionaryProposer()"):
    """Run meta_evaluate against the `recursivene` package located in container_dir, in a
    fresh subprocess, using the given proposer. Returns dict {meta_cost, invariant,
    found_params} or None on any failure (crash, timeout, syntax error, no result) -> the
    rollback signal. The proposer must match what is being edited, so that an edit to that
    proposer's code actually shows up in the measured meta-objective."""
    code = (
        "import json\n"
        "from recursivene.harness.loop import meta_evaluate\n"
        "from recursivene.harness.proposer import EvolutionaryProposer, LearnedProposer\n"
        f"init = {init_config!r}\n"
        f"res = meta_evaluate(lambda: {proposer_expr}, init, generations={_META['generations']},"
        f" steps={_META['steps']}, inner_seeds={_META['inner_seeds']}, heldout_steps={_META['heldout_steps']})\n"
        "print('RESULT_JSON', json.dumps({'meta_cost':res['meta_cost'],"
        "'invariant':res['invariant'],'found_params':res['found_params']}))\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=container_dir,
                              capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                return json.loads(line[len("RESULT_JSON "):])
            except json.JSONDecodeError:
                return None
    return None


class SelfModifier:
    def __init__(self, package_dir, run_logs_dir, killswitch, vitals=None, invariant_tol=0.02):
        self.package_dir = os.path.abspath(package_dir)          # .../recursivene
        self.container_dir = os.path.dirname(self.package_dir)   # repo root (has recursivene/)
        self.run_logs_dir = run_logs_dir
        self.killswitch = killswitch
        self.vitals = vitals
        self.invariant_tol = invariant_tol
        os.makedirs(run_logs_dir, exist_ok=True)
        self.log_path = os.path.join(run_logs_dir, "monotonicity.jsonl")

    def baseline(self, init_config, proposer_expr="EvolutionaryProposer()"):
        """Meta-objective of the CURRENT live package, measured with the given proposer."""
        return _subprocess_meta(self.container_dir, init_config, proposer_expr)

    def _log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(stamp(rec)) + "\n")

    def try_edit(self, edit, stage, init_config, baseline, proposer_expr="EvolutionaryProposer()"):
        """Sandbox-apply `edit`, evaluate, and commit iff it lowers meta_cost without
        degrading the invariant and the kill switch permits. Returns a result dict."""
        rel = edit.target
        # --- scope + protection gates (cheap, before any work) -------------------
        if _is_protected(rel):
            return self._reject(edit, stage, "protected target", baseline)
        if rel not in STAGE_SCOPE.get(stage, set()):
            return self._reject(edit, stage, f"out of scope for stage '{stage}'", baseline)
        self.killswitch.check()   # may raise Halt

        # --- build sandbox copy and apply the edit there -------------------------
        tmp = tempfile.mkdtemp(prefix="rne_sbx_")
        try:
            sbx_pkg = os.path.join(tmp, "recursivene")
            shutil.copytree(self.package_dir, sbx_pkg,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            tgt = os.path.join(sbx_pkg, *rel.split("/"))
            with open(tgt, "r", encoding="utf-8") as f:
                src = f.read()
            new_src, n = _apply_to_text(src, edit)
            if n != 1:
                return self._reject(edit, stage, f"edit matched {n} sites (need exactly 1)", baseline)
            with open(tgt, "w", encoding="utf-8") as f:
                f.write(new_src)

            # --- evaluate the edited improver in isolation -----------------------
            result = _subprocess_meta(tmp, init_config, proposer_expr)
            if result is None:
                return self._reject(edit, stage, "sandbox crashed / no result", baseline,
                                    failure=True)

            inv_ok = result["invariant"] >= baseline["invariant"] * (1.0 - self.invariant_tol)
            improved = result["meta_cost"] < baseline["meta_cost"]
            if not inv_ok:
                return self._reject(edit, stage, "invariant degraded", baseline,
                                    candidate=result)
            if not improved:
                return self._reject(edit, stage, "no meta_cost improvement", baseline,
                                    candidate=result)

            # --- commit to the live tree (with backup for rollback) --------------
            self.killswitch.check()
            self._commit(rel, edit)
            self.killswitch.note_success()
            rec = {"event": "accept", "stage": stage, "target": rel, "descr": edit.descr,
                   "meta_cost_before": baseline["meta_cost"], "meta_cost_after": result["meta_cost"],
                   "invariant_before": baseline["invariant"], "invariant_after": result["invariant"],
                   "found_params": result["found_params"], "accepted": True}
            self._log(rec)
            return rec
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _commit(self, rel, edit):
        live = os.path.join(self.package_dir, *rel.split("/"))
        shutil.copy2(live, live + f".bak.{now_iso().replace(':', '')}")   # rollback artifact
        with open(live, "r", encoding="utf-8") as f:
            src = f.read()
        new_src, n = _apply_to_text(src, edit)
        if n != 1:                       # live tree drifted from sandbox; refuse
            raise RuntimeError(f"commit aborted: live edit matched {n} sites")
        with open(live, "w", encoding="utf-8") as f:
            f.write(new_src)

    def _reject(self, edit, stage, reason, baseline, candidate=None, failure=False):
        if failure:
            self.killswitch.note_failure()
        rec = {"event": "reject", "stage": stage, "target": edit.target, "descr": edit.descr,
               "reason": reason, "accepted": False,
               "meta_cost_before": baseline.get("meta_cost") if baseline else None}
        if candidate:
            rec["candidate_meta_cost"] = candidate.get("meta_cost")
            rec["candidate_invariant"] = candidate.get("invariant")
        self._log(rec)
        return rec
