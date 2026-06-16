"""Run all four benchmarks, print ASCII curves / sparklines, write run_logs/bench_report.json.

    python recursivene/bench/run_benchmark.py            # from repo root
    python -m recursivene.bench.run_benchmark            # equivalently

Everything printed is REAL output of the functions in benchmark.py — no fabricated numbers.
"""

import json
import os
import sys

import numpy as np

# Windows consoles default to cp1252, which cannot encode the sparkline block glyphs.
# Force UTF-8 on stdout/stderr so the ASCII/Unicode curves render everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

if __package__ in (None, ""):
    # allow `python recursivene/bench/run_benchmark.py` (no package context)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from recursivene.bench import benchmark as B
    from recursivene.bench.benchmark import (
        race_to_zero_curve, plateau_break_demo, open_ended_report, saturation_contrast,
        REPO_ROOT, RUN_LOGS,
    )
    from recursivene.core.clock import now_iso
else:
    from . import benchmark as B
    from .benchmark import (
        race_to_zero_curve, plateau_break_demo, open_ended_report, saturation_contrast,
        REPO_ROOT, RUN_LOGS,
    )
    from ..core.clock import now_iso

_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values):
    """Unicode sparkline over a numeric series (non-finite -> '?')."""
    v = np.asarray([x if np.isfinite(x) else np.nan for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return "?" * len(values)
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    out = []
    for x in v:
        if not np.isfinite(x):
            out.append("?")
            continue
        idx = 0 if span == 0 else int(round((x - lo) / span * (len(_BLOCKS) - 1)))
        out.append(_BLOCKS[idx])
    return "".join(out)


def ascii_curve(series, height=8, width=None, label="cost"):
    """Vertical ASCII line chart of a series (log-scaled if it spans >10x)."""
    s = np.asarray([x for x in series if np.isfinite(x)], dtype=float)
    if s.size == 0:
        return "  (no finite points)"
    plot = np.log10(s + 1.0) if (s.max() / max(s.min(), 1e-9) > 10) else s.copy()
    lo, hi = float(plot.min()), float(plot.max())
    span = hi - lo or 1.0
    width = width or len(plot)
    rows = []
    for h in range(height, -1, -1):
        thresh = lo + span * h / height
        line = []
        for val in plot:
            line.append("█" if val >= thresh - span / (2 * height) else " ")
        rows.append("  |" + "".join(f"{c} " for c in line))
    rows.append("  +" + "--" * len(plot))
    rows.append("   " + "".join(f"{i} " for i in range(len(plot))) + f"  ({label}, gen index)")
    return "\n".join(rows)


def _fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf"
    return f"{x:.3e}" if isinstance(x, float) and abs(x) >= 1e4 else f"{x}"


def main():
    print("=" * 72)
    print("  RecursiveNe -- non-saturating benchmark suite")
    print("=" * 72)
    report = {}

    # ---- 1. RACE TO ZERO ------------------------------------------------
    print("\n[1] RACE TO ZERO  (cost-for-competence vs generation; slope<0 = improving)")
    r2z = race_to_zero_curve()
    report["race_to_zero"] = r2z
    costs = [c for _, c in r2z["points"]]
    print(f"    source        : {r2z['source']}  ({r2z['n_points']} points)")
    print(f"    log-lin slope : {r2z['slope']:+.4f} per gen   (frac/gen={r2z['frac_per_gen']:.3f})")
    hl = r2z["halflife"]
    print(f"    cost halflife : {hl:.2f} generations" if hl else "    cost halflife : n/a (slope>=0)")
    if costs:
        ratio = (costs[0] / costs[-1]) if costs[-1] else float("inf")
        print(f"    cost {_fmt(costs[0])} -> {_fmt(costs[-1])}   ({ratio:.1f}x cheaper)")
        print("    spark         : " + sparkline(costs))
        print(ascii_curve(costs, label="cost"))

    # ---- 2. PLATEAU BREAK ----------------------------------------------
    print("\n[2] PLATEAU BREAK  (weak frozen operator vs strong operator, same start/budget)")
    pb = plateau_break_demo()
    report["plateau_break"] = pb
    print(f"    shared gen-0  : {_fmt(pb['gen0_cost'])}  (both arms competent at start)")
    print(f"    WEAK  (frozen): " + sparkline(pb["weak_costs"]) +
          f"   plateau={_fmt(pb['plateau'])}  ({pb['weak_params']} params)")
    print(f"    STRONG (full) : " + sparkline(pb["strong_costs"]) +
          f"   post   ={_fmt(pb['post_plateau'])}  ({pb['strong_params']} params)")
    print(f"    weak costs    : {[_fmt(c) for c in pb['weak_costs']]}")
    print(f"    strong costs  : {[_fmt(c) for c in pb['strong_costs']]}")
    verdict = "BROKE plateau" if pb["broke"] else "did NOT break"
    print(f"    => {verdict}: strong {pb['speedup']:.1f}x below weak plateau "
          f"(drop {pb['drop_frac']*100:.0f}%);  plateaued={pb['plateaued']}")

    # ---- 3. OPEN-ENDED REPORT ------------------------------------------
    print("\n[3] OPEN-ENDED REPORT  (repertoire / hardest-solved / stepping-stone)")
    oe = open_ended_report()
    report["open_ended"] = oe
    print(f"    repertoire    : {oe['repertoire_size']}/{oe['n_learnable']} activities mastered "
          f"(frontier_fraction={oe['frontier_fraction']:.2f})")
    print(f"    hardest solved: w={oe['hardest_solved_w']:.1f} of possible w={oe['hardest_possible_w']:.1f}")
    print(f"    stepping-stone: {oe['stepping_stone']:.2f}  (reached rung {oe['reached_rung']})")
    print(f"    err-vs-freq slope (transfer proxy): {oe['err_freq_slope']:+.3f}")
    print(f"    per-activity MSE: {oe['per_activity_mse']}")

    # ---- 4. SATURATION CONTRAST ----------------------------------------
    print("\n[4] SATURATION CONTRAST  (fixed test flatlines while frontier keeps climbing)")
    sc = saturation_contrast()
    report["saturation_contrast"] = sc
    print(f"    budgets       : {sc['budgets']}")
    print(f"    FIXED  (easy) : " + sparkline(sc["fixed_series"]) +
          f"   {sc['fixed_series']}  (ceiling {sc['fixed_ceiling']:.0f})")
    print(f"    FRONTIER count: " + sparkline(sc["frontier_series"]) +
          f"   {sc['frontier_series']}")
    print(f"    hardest-w     : " + sparkline(sc["hardest_series"]) +
          f"   {[round(h,1) for h in sc['hardest_series']]}")
    si = sc["saturation_index"]
    if si is not None:
        print(f"    fixed saturated at budget index {si} (steps={sc['budgets'][si]}); "
              f"frontier rose {sc['frontier_series'][si]}->{sc['frontier_series'][-1]} AFTERWARD")
    print(f"    => CONTRAST holds: {sc['contrast']}  "
          f"(fixed_saturated={sc['fixed_saturated']}, frontier_rose_post_sat={sc['frontier_rose_post_sat']})")

    # ---- write report --------------------------------------------------
    os.makedirs(RUN_LOGS, exist_ok=True)
    out_path = os.path.join(RUN_LOGS, "bench_report.json")
    report["ts"] = now_iso()
    report["summary"] = {
        "race_to_zero_slope": r2z["slope"],
        "race_to_zero_source": r2z["source"],
        "plateau_break_speedup": pb["speedup"],
        "plateau_broke": pb["broke"],
        "frontier_fraction": oe["frontier_fraction"],
        "hardest_solved_w": oe["hardest_solved_w"],
        "saturation_contrast": sc["contrast"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print("\n" + "=" * 72)
    print(f"  wrote {out_path}")
    print(f"  HEADLINE: race-to-0 slope={r2z['slope']:+.3f}/gen | "
          f"plateau-break {pb['speedup']:.1f}x | "
          f"frontier {oe['repertoire_size']}/{oe['n_learnable']} | "
          f"saturation-contrast={sc['contrast']}")
    print("=" * 72)
    return report


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
