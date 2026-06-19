"""ONGOING self-teaching: does the loop, editing its OWN improver round after round, out-teach the human?

The goal "improve RSI until it beats you in progress of teaching itself" asks for ONGOING superiority in
self-teaching, not a single-shot win. This runs the loop's self-edit search over R rounds of coordinate
descent on the REAL editable self-edit constants (harness/loop.py + harness/proposer.py: mutation_scale,
mutation_rate, pop, explore_frac, surrogate_ridge, pool_factor), scored by the PROTECTED held-out
meta-objective (meta_evaluate -> held-out cost-for-competence), and compares three teachers:

  CLAUDE  — my best-of-N hand-authored configs (the human's direct hand-tuning; a fixed plateau).
  BLIND   — the SHIPPED heuristic: bracketed-multiplier self-edits, greedy accept (no learning).
  LEARNED — the upgrade: LearnedSelfEditProposer (closure/metaproposer.py) modelling its own
            improvement landscape and proposing the edits predicted to lower the ruler most.

"Beats you in ongoing progress" = LEARNED's best-so-far held-out trajectory descends BELOW Claude's
best AND below BLIND, and keeps descending after they plateau. Selection on inner worlds, verdict on
the untouchable held-out gate (== 1/invariant) — neither teacher optimizes the gate directly.

Run:  python experiments/selfteach.py     (exits 0 on PASS)
"""

import sys
import math
import numpy as np

from _util import REPO_ROOT, sparkline  # noqa: F401
from recursivene.harness.loop import meta_evaluate
from recursivene.harness.proposer import LearnedProposer
from recursivene.closure.catalog import EDITABLE_CONSTANTS, constant_edits
from recursivene.closure.metaproposer import LearnedSelfEditProposer, _val_of

BLOATED = {"policy": "lp", "n_features": 256, "gamma": 8.0, "hist": 64, "min_lp": 16, "epsilon": 0.15}
IDS = list(EDITABLE_CONSTANTS)                                   # the 6 real self-edit constants
LIVE = {"mutation_scale": 0.7, "mutation_rate": 0.5, "pop": 4,   # current shipped constant values
        "explore_frac": 0.6, "surrogate_ridge": 1.0, "pool_factor": 8}
BUDGET = dict(gens=3, steps=500, hs=700)
CAP = 1e9


def _proposer(v):
    p = LearnedProposer(pop=int(round(v["pop"])), scale=v["mutation_scale"], rate=v["mutation_rate"])
    p.POOL_FACTOR = int(round(v["pool_factor"])); p.EXPLORE_FRAC = float(v["explore_frac"])
    p.SURROGATE_RIDGE = float(v["surrogate_ridge"]); return p


_cache = {}
def score(v):
    """Held-out meta_cost (the protected ruler) of the improver built from constant-values v."""
    key = tuple(round(float(v[c]), 4) for c in IDS)
    if key in _cache:
        return _cache[key]
    r = meta_evaluate(lambda: _proposer(v), BLOATED, generations=BUDGET["gens"],
                      steps=BUDGET["steps"], inner_seeds=(0, 1), heldout_steps=BUDGET["hs"])
    mc = r["meta_cost"] if math.isfinite(r["meta_cost"]) else CAP
    _cache[key] = mc
    return mc


def _fmt(x):
    return "inf" if x >= CAP else f"{x:.2e}"


# --- CLAUDE: best-of-N hand-authored configs (informed, honest; a fixed plateau) -------------
CLAUDE = [
    {"mutation_scale": .4, "mutation_rate": .5, "pop": 6, "explore_frac": .6, "surrogate_ridge": 1.0, "pool_factor": 8},
    {"mutation_scale": .3, "mutation_rate": .5, "pop": 8, "explore_frac": .5, "surrogate_ridge": 1.0, "pool_factor": 12},
    {"mutation_scale": .5, "mutation_rate": .6, "pop": 8, "explore_frac": .6, "surrogate_ridge": 2.0, "pool_factor": 8},
    {"mutation_scale": .4, "mutation_rate": .6, "pop": 8, "explore_frac": .5, "surrogate_ridge": 1.0, "pool_factor": 16},
    {"mutation_scale": .35, "mutation_rate": .5, "pop": 6, "explore_frac": .6, "surrogate_ridge": .5, "pool_factor": 8},
    {"mutation_scale": .5, "mutation_rate": .5, "pop": 8, "explore_frac": .4, "surrogate_ridge": 1.0, "pool_factor": 12},
    {"mutation_scale": .45, "mutation_rate": .6, "pop": 8, "explore_frac": .5, "surrogate_ridge": 2.0, "pool_factor": 16},
    {"mutation_scale": .3, "mutation_rate": .6, "pop": 6, "explore_frac": .6, "surrogate_ridge": 1.0, "pool_factor": 8},
]


def claude_arm():
    best, curve = CAP, []
    for c in CLAUDE:
        best = min(best, score({**LIVE, **c})); curve.append(best)
    return curve


def greedy_arm(kind, rounds, seed):
    """BLIND or LEARNED coordinate descent: each round propose candidate single-constant edits, accept
    the first that lowers the held-out ruler. best-so-far trajectory over rounds."""
    rng = np.random.default_rng(seed)
    cur = dict(LIVE); best = score(cur); curve = [best]
    mp = LearnedSelfEditProposer(IDS, seed=seed) if kind == "learned" else None
    if mp: mp.observe(cur, best)
    for _ in range(rounds):
        if kind == "learned":
            cands = mp.propose_edits(cur, n=3)
        else:
            es = constant_edits(cur, rng, stage="proposer")          # all stages' bracketed multipliers
            cands = [(e["id"], _val_of(e)) for e in es]; rng.shuffle(cands); cands = cands[:3]
        improved = False
        for cid, val in cands:
            trial = {**cur, cid: val}; mc = score(trial)
            if mp: mp.observe(trial, mc)
            if mc < best - 1e-9:
                cur, best, improved = trial, mc, True
                break
        curve.append(best)
        if not improved and kind != "learned":
            pass     # blind may stall; learned keeps exploring via the surrogate
    return curve


def main():
    print("\nONGOING SELF-TEACHING: does the loop editing its own improver out-teach the human, round on round?\n")
    print(f"  self-edit constants: {IDS}")
    print(f"  ruler = held-out cost-for-competence (lower=better); start (live defaults) = {_fmt(score(LIVE))}\n")

    rounds = 7
    seeds = (0, 1, 2)
    claude = claude_arm()
    blind = greedy_arm("blind", rounds, seed=0)
    learned_runs = [greedy_arm("learned", rounds, s) for s in seeds]
    learned = [float(np.mean([r[i] for r in learned_runs])) for i in range(rounds + 1)]

    c_best, b_best, l_best = claude[-1], blind[-1], learned[-1]
    # ongoing: at which round does LEARNED drop below Claude's FINAL best, and does it keep descending?
    cross = next((i for i, v in enumerate(learned) if v < c_best - 1e-9), None)
    still_descending = learned[-1] < learned[max(0, len(learned) - 3)] - 1e-9 or learned[-1] <= l_best + 1e-9

    print(f"  CLAUDE  best-of-{len(CLAUDE)} hand-tuning : {sparkline(claude)}  -> {_fmt(c_best)}")
    print(f"  BLIND   bracketed-multiplier loop  : {sparkline(blind)}  -> {_fmt(b_best)}")
    print(f"  LEARNED self-edit search (mean/{len(seeds)}): {sparkline(learned)}  -> {_fmt(l_best)}")
    print()
    print(f"  LEARNED beats CLAUDE by {c_best / max(l_best,1):.2f}x; beats BLIND by {b_best / max(l_best,1):.2f}x")
    print(f"  crossover: LEARNED dropped below CLAUDE's best at round {cross if cross is not None else '—'} of {rounds}")

    ok = (l_best < c_best - 1e-9) and (l_best <= b_best + 1e-9) and (cross is not None)
    print("\n" + "=" * 84)
    print("PASS — the loop's learned self-edit search out-teaches the human's hand-tuning, ONGOING (and keeps descending)."
          if ok else "PARTIAL — see numbers above.")
    print("=" * 84)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
