"""Tend the seed. Grow it across many seasons and report whether the loop compounds.

Run:  python experiments/run_garden.py
"""

import numpy as np

from _util import REPO_ROOT, sparkline
from recursivene.garden import grow_garden
from recursivene.core.clock import now_iso


def main():
    print(f"\n[{now_iso()}] RecursiveNe — the gardener tends the seed\n")
    rings = grow_garden(seasons=22, repo_root=REPO_ROOT, verbose=True)

    reached = [r for r in rings if r["event"] != "stalled"]
    if not reached:
        print("\nNo growth — the seed could not reach beyond its starting repertoire.")
        return
    hardest = [r["hardest_known"] for r in reached]
    rep = [r["repertoire"] for r in reached]
    ctk = [r["cost_to_know_samples"] for r in reached if r["cost_to_know_samples"] not in (None, np.inf)]
    grows = sum(r["grows"] for r in reached)

    print("\n" + "=" * 64)
    print("GROWTH RINGS — what the season-by-season record now makes KNOWN:")
    print(f"  started knowing up to w=12.0 (repertoire 6)")
    print(f"  ended knowing up to w={hardest[-1]} (repertoire {rep[-1]})")
    print(f"  the seed grew its own capacity {grows} time(s) to reach what it couldn't")
    print(f"  complexity climb : {sparkline(hardest)}  ({hardest[0]} -> {hardest[-1]})")
    print(f"  repertoire climb : {sparkline(rep)}  ({rep[0]} -> {rep[-1]})")
    if ctk:
        print(f"  cost-to-know/rung: {sparkline(ctk)}  ({ctk[0]} -> {ctk[-1]} samples)")
    print("=" * 64)
    print("\nCompounded." if hardest[-1] > 12.0 and rep[-1] > 6 else "\nDid not compound.")


if __name__ == "__main__":
    main()
