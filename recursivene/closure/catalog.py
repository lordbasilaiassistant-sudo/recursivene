"""A catalog of SAFE, parameterized source edits the automated proposer can generate
without producing broken code. Each edit rewrites one module-level constant of the
improvement operator (in harness/loop.py) or of the proposer itself (harness/proposer.py).

Why a catalog: a fully free-form code rewriter is what a model in the seam can do (and the
SeamProposer path allows it, under the same gate). But the AUTOMATED, no-human-in-the-loop
loop needs to mint edits mechanically, and bounded constant-rewrites are guaranteed to stay
syntactically valid — so the loop can run unattended and any genuinely bad edit is caught by
evaluation/rollback rather than by crashing the interpreter on a syntax error.
"""

import numpy as np

# id -> (target_relpath, constant_name, (lo, hi), kind, stage)
EDITABLE_CONSTANTS = {
    "mutation_scale": ("harness/loop.py", "MUTATION_SCALE", (0.1, 1.2), "float", "harness"),
    "mutation_rate":  ("harness/loop.py", "MUTATION_RATE", (0.2, 0.9), "float", "harness"),
    "pop":            ("harness/loop.py", "POP", (3, 12), "int", "harness"),
    "explore_frac":   ("harness/proposer.py", "EXPLORE_FRAC", (0.1, 0.6), "float", "proposer"),
    "surrogate_ridge": ("harness/proposer.py", "SURROGATE_RIDGE", (0.1, 10.0), "float", "proposer"),
    "pool_factor":    ("harness/proposer.py", "POOL_FACTOR", (4, 24), "int", "proposer"),
}


def _fmt(value, kind):
    return str(int(round(value))) if kind == "int" else f"{value:.4f}"


# deterministic multiplier spread: brackets the current value both ways so that if the
# current setting is suboptimal, at least one candidate improves the meta-objective.
_MULTIPLIERS = (0.5, 0.7, 1.4, 2.0)


def value_edit(cid, value, current_values):
    """Build ONE edit that sets editable constant `cid` to a specific (continuous) `value`, reusing the
    same indentation-preserving regex as constant_edits. This is what the LearnedSelfEditProposer emits:
    arbitrary values predicted to lower the meta-objective, not just the fixed bracket multipliers."""
    target, name, (lo, hi), kind, stage = EDITABLE_CONSTANTS[cid]
    val = float(np.clip(value, lo, hi))
    cur = current_values.get(cid, 0.5 * (lo + hi))
    new = _fmt(val, kind)
    return {
        "id": cid, "target": target, "constant": name,
        "find": rf"(?m)^([ \t]*){name} = .*$",
        "replace": rf"\g<1>{name} = {new}",
        "descr": f"{name}: {_fmt(cur, kind)} -> {new} (learned)",
        "stage": stage,
    }


def constant_edits(current_values, rng, stage, per_constant=None):
    """Yield candidate Edits for every editable constant whose stage is <= the current
    closure stage. `current_values` maps id -> current numeric value. Deterministic spread
    of values per constant. Returns dicts {id, target, constant, find, replace, descr, stage}."""
    from .selfmod import STAGE_ORDER
    allowed = set(STAGE_ORDER[: STAGE_ORDER.index(stage) + 1])
    edits = []
    for cid, (target, name, (lo, hi), kind, est) in EDITABLE_CONSTANTS.items():
        if est not in allowed:
            continue
        cur = current_values.get(cid, 0.5 * (lo + hi))
        seen = set()
        for m in _MULTIPLIERS:
            val = float(np.clip(cur * m, lo, hi))
            new = _fmt(val, kind)
            if new in seen or new == _fmt(cur, kind):
                continue
            seen.add(new)
            edits.append({
                "id": cid, "target": target, "constant": name,
                # preserve leading indentation so class attributes (indented) match too,
                # not just module-level constants (column 0)
                "find": rf"(?m)^([ \t]*){name} = .*$",
                "replace": rf"\g<1>{name} = {new}",
                "descr": f"{name}: {_fmt(cur, kind)} -> {new}",
                "stage": est,
            })
    return edits
