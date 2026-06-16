"""Self-test for the benchmark suite. Runs in <90s and asserts the load-bearing claims.

    python recursivene/bench/test_benchmark.py        # from repo root

Every assertion is a falsifiable property of a benchmark, checked against its REAL output:
  - race_to_zero slope is NEGATIVE (cost decreasing across accepted generations);
  - plateau-break: a frozen operator plateaus, the full operator descends BELOW it;
  - open-ended metrics are well-formed and the frontier is non-trivial;
  - saturation contrast: the fixed test saturates while the frontier keeps climbing.
If any fails, the benchmark's central claim is false and we want to know.
"""

import sys
import time

import numpy as np

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recursivene.bench.benchmark import (
    race_to_zero_curve, plateau_break_demo, open_ended_report, saturation_contrast,
)


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(f"{name} FAILED: {detail}")


def test_race_to_zero():
    """Cost-for-competence is DECREASING: log-linear slope < 0 (the race to 0)."""
    r = race_to_zero_curve()                       # uses run_logs or generates a real trajectory
    costs = [c for _, c in r["points"]]
    _check("race_to_zero has >=2 points", r["n_points"] >= 2, f"n={r['n_points']} src={r['source']}")
    _check("race_to_zero slope < 0 (cost decreasing)", r["slope"] < 0,
           f"slope={r['slope']:+.4f}/gen src={r['source']}")
    _check("race_to_zero cost non-increasing (running-min ratchet)",
           all(costs[i + 1] <= costs[i] + 1e-6 for i in range(len(costs) - 1)),
           f"first={costs[0]:.3e} last={costs[-1]:.3e}")
    _check("race_to_zero frac_per_gen < 1 (cheaper each gen)", r["frac_per_gen"] < 1.0,
           f"frac/gen={r['frac_per_gen']:.3f}")


def test_plateau_break():
    """A FROZEN operator plateaus; the FULL operator descends BELOW that plateau."""
    p = plateau_break_demo(generations=4, steps=900, seeds=(0, 1), nf0=80)
    _check("both arms start competent (finite shared gen-0)",
           np.isfinite(p["gen0_cost"]) and p["weak_reached"] >= 0.999 and p["strong_reached"] >= 0.999,
           f"gen0={p['gen0_cost']:.3e} weak_reached={p['weak_reached']} strong_reached={p['strong_reached']}")
    _check("weak arm PLATEAUS (frozen operator stays flat)", p["plateaued"],
           f"weak_costs={[f'{c:.3e}' for c in p['weak_costs']]}")
    _check("strong arm BREAKS the plateau (post < plateau)", p["broke"],
           f"plateau={p['plateau']:.3e} post={p['post_plateau']:.3e}")
    _check("plateau break is a real speedup (>1.5x)", p["speedup"] > 1.5,
           f"speedup={p['speedup']:.2f}x  ({p['weak_params']} -> {p['strong_params']} params)")


def test_open_ended():
    """Frontier metrics are well-formed and the repertoire is non-empty but not trivially full."""
    o = open_ended_report(steps=1500, seed=0)
    _check("repertoire within bounds", 0 <= o["repertoire_size"] <= o["n_learnable"],
           f"{o['repertoire_size']}/{o['n_learnable']}")
    _check("at least one activity mastered", o["repertoire_size"] >= 1,
           f"mastered={o['repertoire_size']}")
    _check("frontier extends above the canonical inner ladder (room to grow)",
           o["hardest_possible_w"] > 12.0,
           f"possible_w={o['hardest_possible_w']}")
    _check("hardest_solved_w <= hardest_possible_w", o["hardest_solved_w"] <= o["hardest_possible_w"],
           f"solved={o['hardest_solved_w']} possible={o['hardest_possible_w']}")
    _check("stepping_stone in [0,1]", 0.0 <= o["stepping_stone"] <= 1.0,
           f"stepping_stone={o['stepping_stone']:.2f}")


def test_saturation_contrast():
    """A FIXED test saturates while the FRONTIER metric keeps climbing afterward."""
    s = saturation_contrast(steps_grid=(300, 600, 1000, 1600, 2400), seed=0)
    front = s["frontier_series"]
    _check("fixed test SATURATED (hit ceiling and flatlined)", s["fixed_saturated"],
           f"fixed={s['fixed_series']}")
    _check("frontier kept CLIMBING after fixed saturated", s["frontier_rose_post_sat"],
           f"frontier={front} sat_idx={s['saturation_index']}")
    _check("frontier is monotone non-decreasing in budget",
           all(front[i + 1] >= front[i] for i in range(len(front) - 1)), f"frontier={front}")
    _check("CONTRAST holds (fixed flat AND frontier rose)", s["contrast"],
           f"contrast={s['contrast']}")


def main():
    t0 = time.time()
    print("=" * 64)
    print("  benchmark self-test")
    print("=" * 64)
    tests = [
        ("race_to_zero", test_race_to_zero),
        ("plateau_break", test_plateau_break),
        ("open_ended", test_open_ended),
        ("saturation_contrast", test_saturation_contrast),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        ts = time.time()
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")
        print(f"  ({time.time() - ts:.1f}s)")
    dt = time.time() - t0
    print("\n" + "=" * 64)
    verdict = "ALL PASS" if failed == 0 else f"{failed} FAILED"
    print(f"  {verdict}  in {dt:.1f}s")
    print("=" * 64)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
