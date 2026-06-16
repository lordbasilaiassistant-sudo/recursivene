"""Experiment 1 — the seed's central claim, in one run.

Three policies (random / novelty / learning-progress) share identical world models and
identical sample budgets. We measure, for each:
  * the competence curve (error on learnable activities vs. interactions)
  * samples to reach competence threshold tau     -> 'least training intensive'
  * fraction of budget poured into the noise activity -> the noisy-TV trap
  * the LP agent's activity allocation over time   -> emergent developmental curriculum

Run:  python experiments/run_seed.py
"""

import json
import os
import numpy as np

from _util import sparkline, bar
from recursivene import run
from recursivene.seed import competence  # noqa: F401  (kept for interactive use)

STEPS = 4000
SEED = 0
TAU = 0.05


def samples_to_tau(log, tau=TAU):
    comp = np.asarray(log["competence"])
    steps = np.asarray(log["steps"])
    hit = np.where(comp <= tau)[0]
    return int(steps[hit[0]]) if len(hit) else None


def main():
    results = {}
    print(f"\nRecursiveNe — seed comparison ({STEPS} interactions, tau={TAU})\n")
    print(f"{'policy':<10}{'final_err':>11}{'samples->tau':>14}{'noise%':>9}   competence curve")
    print("-" * 78)

    curves = {}
    for policy in ("random", "novelty", "lp"):
        _, _, log = run({"policy": policy}, steps=STEPS, seed=SEED)
        s2t = samples_to_tau(log)
        noise_frac = float((log["region_seq"] == log["noise_idx"]).mean())
        curves[policy] = log["competence"]
        results[policy] = {
            "final_competence": float(log["competence"][-1]),
            "samples_to_tau": s2t,
            "noise_fraction": noise_frac,
            "visits": {log["names"][i]: int(v) for i, v in enumerate(log["visits"])},
        }
        spark = sparkline(log["competence"], lo=0.0, hi=0.6)
        s2t_s = str(s2t) if s2t is not None else "never"
        print(f"{policy:<10}{log['competence'][-1]:>11.4f}{s2t_s:>14}"
              f"{noise_frac*100:>8.1f}%   {spark}")

    # Per-activity budget allocation (visits), to expose the curriculum.
    print("\nbudget allocation across activities (visit share):")
    names = run({"policy": "lp"}, steps=1, seed=SEED)[2]["names"]
    for policy in ("random", "novelty", "lp"):
        v = np.array([results[policy]["visits"][n] for n in names], dtype=float)
        share = v / v.sum()
        print(f"\n  {policy}:")
        for n, s in zip(names, share):
            tag = "  <- noisy TV" if n == "noise" else ""
            print(f"    {n:<8} {bar(s)} {s*100:5.1f}%{tag}")

    # Developmental trajectory of the LP agent: which activity holds attention over time.
    print("\nLP agent — where attention flows over time (windowed argmax activity):")
    _, _, log = run({"policy": "lp"}, steps=STEPS, seed=SEED)
    seq = log["region_seq"]
    win = STEPS // 12
    line = []
    for i in range(0, STEPS, win):
        chunk = seq[i:i + win]
        dominant = int(np.bincount(chunk, minlength=len(names)).argmax())
        line.append(log["names"][dominant][:4])
    print("    " + " -> ".join(line))

    out = os.path.join(os.path.dirname(__file__), "results_seed.json")
    with open(out, "w") as f:
        json.dump({"config": {"steps": STEPS, "seed": SEED, "tau": TAU},
                   "results": results}, f, indent=2)
    print(f"\nwrote {out}\n")


if __name__ == "__main__":
    main()
