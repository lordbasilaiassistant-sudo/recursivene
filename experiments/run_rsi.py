"""Experiment 2 — the RSI ratchet ('race to 0').

Start the learner deliberately bloated (large D, large LP window) and let the
EvolutionaryProposer drive cost-to-competence down across generations WITHOUT a human
in the inner loop. Cost = samples_to_competence * sqrt(params): a FLOP proxy. The whole
point is to watch it fall while competence is held — capability getting cheaper.

Run:  python experiments/run_rsi.py
"""

import json
import os
import numpy as np

from _util import sparkline
from recursivene.harness import rsi_loop, EvolutionaryProposer

GENERATIONS = 8
BLOATED_START = {"n_features": 256, "hist": 56, "epsilon": 0.2, "policy": "lp"}


def main():
    print("\nRecursiveNe — RSI ratchet: driving cost-to-competence down\n")
    print("  (start deliberately bloated; evolutionary proposer, no human in inner loop)\n")
    best_cfg, best_eval, history = rsi_loop(
        generations=GENERATIONS,
        proposer=EvolutionaryProposer(pop=6, rate=0.5),
        init_config=BLOATED_START,
        steps=2500,
        seeds=(0, 1, 2),
        tau=0.05,
        verbose=True,
    )

    costs = [h["cost"] for h in history]
    finite = [c for c in costs if np.isfinite(c)]
    print("\n  cost trajectory:", sparkline(costs, lo=0.0, hi=max(finite) if finite else 1.0))

    gen0, genN = history[0], history[-1]
    drop = (gen0["cost"] - genN["cost"]) / gen0["cost"] * 100 if np.isfinite(gen0["cost"]) else float("nan")
    print(f"\n  gen 0 : cost={gen0['cost']:.1f}  params={gen0['n_params']}  samples->tau={gen0['samples_to_tau']}")
    print(f"  gen {len(history)-1} : cost={genN['cost']:.1f}  params={genN['n_params']}  samples->tau={genN['samples_to_tau']}")
    print(f"  -> cost reduced {drop:.1f}%  while final competence held at {genN['final_competence']:.4f}")
    print(f"\n  best config found: {json.dumps(best_cfg)}")

    out = os.path.join(os.path.dirname(__file__), "results_rsi.json")
    with open(out, "w") as f:
        json.dump({"best_config": best_cfg, "best_eval": best_eval, "history": history},
                  f, indent=2)
    print(f"\nwrote {out}\n")


if __name__ == "__main__":
    main()
