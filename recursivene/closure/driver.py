"""The staged-closure driver — PROTECTED trusted outer loop.

This is the referee, not a player: it does not compute toward the objective (so it is not
part of the editable surface), it ORCHESTRATES the three levels and enforces the safety
contract. It opens write-scope one level at a time and only after the previous level's gate
has passed:

  STAGE 1  model    : ratchet the config (weak-RSI baseline) via harness.search -> H3
  STAGE 2  harness  : let the proposer EDIT harness/loop.py (the improvement operator),
                      each edit sandboxed + invariant-gated -> first meta-meta edit
  STAGE 3  proposer : let the proposer EDIT harness/proposer.py (its own definition),
                      measured with the LearnedProposer so its constants actually matter

Throughout: the kill switch is checked every generation and before every commit; vitals are
written for the parent (this loop) and the child (the learner); every accepted/rejected edit
is appended to the timestamped monotonicity log.
"""

import os
import time
import numpy as np

from ..core.clock import now_iso, now_unix
from ..core.killswitch import KillSwitch, Halt
from ..core.vitals import Vitals
from ..harness.loop import search
from ..harness.proposer import EvolutionaryProposer
from ..invariant import invariant_score
from .selfmod import SelfModifier
from .catalog import constant_edits, EDITABLE_CONSTANTS


def _default_constant_values():
    # must match the live constants in harness/loop.py and harness/proposer.py
    return {"mutation_scale": 0.5, "mutation_rate": 0.5, "pop": 4,
            "pool_factor": 8, "explore_frac": 0.34, "surrogate_ridge": 1.0}


def run_closure(repo_root, init_config, stage1_gens=8, stage_edit_rounds=4,
                max_seconds=3000, verbose=True):
    """Drive the full three-level closure. Returns a summary dict the validate gate reads."""
    package_dir = os.path.join(repo_root, "recursivene")
    run_logs = os.path.join(repo_root, "run_logs")
    os.makedirs(run_logs, exist_ok=True)
    start = now_unix()

    vit = Vitals(os.path.join(repo_root, "vitals"))
    ks = KillSwitch(stop_file=os.path.join(run_logs, "STOP"),
                    max_seconds=max_seconds, invariant_floor=0.0, start_unix=start)
    sm = SelfModifier(package_dir, run_logs, ks, vit)
    rng = np.random.default_rng(0)
    cur_vals = _default_constant_values()
    summary = {"ts": now_iso(), "stages": {}}

    def log(msg):
        if verbose:
            print(msg)

    try:
        # ---------------- STAGE 1: object/config ratchet (H3) --------------------
        log(f"[{now_iso()}] STAGE 1 (model): config ratchet")

        def on_gen(g, cfg, ev):
            ks.check(elapsed=now_unix() - start)
            vit.beat("child", stage="model", gen=g, cost=ev["cost"],
                     n_params=ev["n_params"], competence=ev["final_competence"],
                     noise_fraction=ev["noise_fraction"])
        best_config, best_eval, hist = search(
            EvolutionaryProposer(), init_config, generations=stage1_gens,
            steps=2500, seeds=(0, 1, 2), on_gen=on_gen)
        gen0_cost = hist[0]["cost"]
        s1_pass = np.isfinite(best_eval["cost"]) and best_eval["cost"] < gen0_cost
        vit.beat("parent", stage="model", best_cost=best_eval["cost"],
                 start_cost=gen0_cost, passed=bool(s1_pass))
        summary["stages"]["model"] = {
            "start_cost": gen0_cost, "best_cost": best_eval["cost"],
            "best_params": best_eval["n_params"], "best_config": best_config,
            "history": [{"gen": h["gen"], "cost": h["cost"], "n_params": h["n_params"]} for h in hist],
            "passed": bool(s1_pass)}
        log(f"  cost {gen0_cost:.3e} -> {best_eval['cost']:.3e}  D={best_config['n_features']}  pass={s1_pass}")

        # Baseline meta-objective for the code-edit stages, anchored at the BLOATED init
        # (not the optimized stage-1 config): proposer quality only shows when there is
        # room to improve, so we measure how well the harness improves a bloated start.
        base_evo = sm.baseline(init_config, "None")
        if base_evo is None:
            raise Halt("could not establish meta baseline")

        # ---------------- STAGE 2: edit the harness (operator) -------------------
        log(f"[{now_iso()}] STAGE 2 (harness): self-edit harness/loop.py")
        s2_accepts = []
        baseline = base_evo
        for r in range(stage_edit_rounds):
            ks.check(elapsed=now_unix() - start, invariant=baseline["invariant"])
            edits = constant_edits(cur_vals, rng, stage="harness", per_constant=2)
            for e in edits:
                from .selfmod import Edit
                edit = Edit(e["target"], e["find"], e["replace"], e["descr"], e["stage"])
                res = sm.try_edit(edit, "harness", init_config, baseline, "None")
                if res.get("accepted"):
                    s2_accepts.append(res)
                    baseline = {"meta_cost": res["meta_cost_after"], "invariant": res["invariant_after"]}
                    cur_vals[e["id"]] = float(e["replace"].split("=")[1])
                    vit.beat("parent", stage="harness", meta_cost=baseline["meta_cost"],
                             accepted_edit=e["descr"])
                    log(f"  ACCEPT {e['descr']}  meta_cost->{res['meta_cost_after']:.3e}")
                    break
            if s2_accepts:    # one accepted operator-edit is enough; move on (speed)
                break
        summary["stages"]["harness"] = {
            "accepted": [a["descr"] for a in s2_accepts],
            "final_meta_cost": baseline["meta_cost"], "passed": len(s2_accepts) >= 1}
        log(f"  stage 2 accepted {len(s2_accepts)} edit(s) to harness/loop.py")

        # ---------------- STAGE 3: edit the proposer itself ----------------------
        log(f"[{now_iso()}] STAGE 3 (proposer): self-edit harness/proposer.py")
        base_learned = sm.baseline(init_config, "LearnedProposer()")
        s3_accepts = []
        baseline = base_learned if base_learned else baseline
        for r in range(stage_edit_rounds):
            ks.check(elapsed=now_unix() - start, invariant=baseline["invariant"])
            edits = constant_edits(cur_vals, rng, stage="proposer", per_constant=2)
            edits = [e for e in edits if e["target"] == "harness/proposer.py"]
            for e in edits:
                from .selfmod import Edit
                edit = Edit(e["target"], e["find"], e["replace"], e["descr"], e["stage"])
                res = sm.try_edit(edit, "proposer", init_config, baseline, "LearnedProposer()")
                if res.get("accepted"):
                    s3_accepts.append(res)
                    baseline = {"meta_cost": res["meta_cost_after"], "invariant": res["invariant_after"]}
                    cur_vals[e["id"]] = float(e["replace"].split("=")[1])
                    vit.beat("parent", stage="proposer", meta_cost=baseline["meta_cost"],
                             accepted_edit=e["descr"])
                    log(f"  ACCEPT {e['descr']}  meta_cost->{res['meta_cost_after']:.3e}")
                    break
            if s3_accepts:    # one accepted proposer-self-edit is enough; move on
                break
        summary["stages"]["proposer"] = {
            "accepted": [a["descr"] for a in s3_accepts],
            "final_meta_cost": baseline["meta_cost"],
            "edited_proposer_file": any(a["target"] == "harness/proposer.py" for a in s3_accepts),
            "passed": len(s3_accepts) >= 1}
        log(f"  stage 3 accepted {len(s3_accepts)} edit(s) to harness/proposer.py")

    except Halt as h:
        summary["halted"] = str(h)
        log(f"[{now_iso()}] HALTED: {h}")

    summary["elapsed_s"] = now_unix() - start
    summary["ts_end"] = now_iso()
    return summary
