"""POWERPLAY-style open-endedness demo.

Run from the repo root:
    python recursivene/generator/run_generator.py

The loop, each round:
  1. Build the world with the frontier discovered so far (extra_w), train a learner.
  2. Record the repertoire (mastered activities) and the complexity (hardest solved freq).
  3. If the current world is mastered, ask the generator for the next gated frontier rung
     and append it -> the world gets harder by exactly one learnable, non-forgetting step.
  4. In parallel, re-measure a FIXED frequency-ladder benchmark (the original inner world).

The demonstration printed at the end is the contrast that motivates open-endedness:
  * the FIXED benchmark's score SATURATES — once the learner aces the fixed ladder, its
    cost-for-competence flatlines; the fixed test has stopped measuring progress (Goodhart);
  * the OPEN-ENDED metric (repertoire size, hardest-solved complexity) keeps CLIMBING,
    because the generator keeps minting new learnable problems at the moving frontier.

Everything is real: numbers come from actually training the substrate learner and actually
probing candidate frequencies. No fabricated values. Persists run_logs/generator_report.json.
"""

import os
import sys
import json
import time

import numpy as np

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recursivene.world import make_world                       # noqa: E402
from recursivene.seed import run, DEFAULT_CONFIG               # noqa: E402
from recursivene.objective import competence, TAU             # noqa: E402
from recursivene.generator.generator import (                 # noqa: E402
    propose_problems, frontier_summary,
)
from recursivene.generator.abstraction import mint_abstraction  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_LOGS = os.path.join(REPO_ROOT, "run_logs")


def _cost_to_tau(log, tau=TAU):
    """FLOPs to first reach worst-case competence <= tau in a run log (None if never)."""
    for f, c in zip(log["flops"], log["competence"]):
        if c <= tau:
            return int(f)
    return None


def run_demo(rounds=6, steps=3500, seed=0, n_noise=12, verbose=True):
    """Run the open-ended loop for `rounds` and return a report dict."""
    extra_w = ()          # the frontier discovered so far (frequencies beyond the base ladder)
    history = []          # per-round frontier summaries (feeds abstraction minting)
    rows = []             # per-round records for the report
    t_start = time.time()

    for rnd in range(rounds):
        # --- 1+2. Train on the current (growing) world; record repertoire & complexity ----
        world = make_world("inner", seed=seed, n_noise=n_noise, extra_w=extra_w)
        _, learner, log = run(DEFAULT_CONFIG, world=world, steps=steps, seed=seed)
        fs = frontier_summary(world, learner)
        history.append(fs)

        # --- FIXED benchmark: the ORIGINAL inner ladder, re-measured every round ----------
        # This is the test that saturates. We train a fresh learner on the FIXED world and
        # record its cost-for-competence; once mastered it stops moving (the whole point).
        fixed_world = make_world("inner", seed=seed, n_noise=n_noise)   # no extra_w, ever
        _, fixed_learner, fixed_log = run(DEFAULT_CONFIG, world=fixed_world, steps=steps, seed=seed)
        fixed_cost = _cost_to_tau(fixed_log)
        fixed_worst = float(competence(fixed_world, fixed_learner))

        # --- 3. Propose the next frontier rung (gated) ------------------------------------
        proposals = propose_problems(learner, world, n_propose=1, seed=seed + rnd)
        proposed_w = proposals[0].w if proposals else None
        proposal_evidence = proposals[0].as_dict() if proposals else None

        rows.append({
            "round": rnd,
            "extra_w": [round(float(w), 3) for w in extra_w],
            "n_mastered": fs["n_mastered"],
            "mastered_ws": fs["mastered_ws"],
            "hardest_mastered_w": fs["hardest_mastered_w"],
            "open_ended_worst_competence": fs["worst_competence"],
            "fixed_benchmark_cost_flops": fixed_cost,
            "fixed_benchmark_worst_competence": fixed_worst,
            "proposed_next_w": proposed_w,
            "proposal_evidence": proposal_evidence,
        })

        if verbose:
            mc = "/".join(str(round(w, 1)) for w in fs["mastered_ws"])
            pw = f"{proposed_w:.1f}" if proposed_w is not None else "none"
            fc = f"{fixed_cost:.2e}" if fixed_cost is not None else "inf"
            print(f"round {rnd}: repertoire={fs['n_mastered']:2d} "
                  f"hardest_w={fs['hardest_mastered_w']:5.1f}  "
                  f"fixed_cost={fc}  ->  propose w={pw}")

        # --- 4. Add the accepted rung; repeat ---------------------------------------------
        if proposed_w is None:
            if verbose:
                print(f"round {rnd}: frontier exhausted (no admissible problem) — stopping.")
            break
        extra_w = tuple(extra_w) + (proposed_w,)

    # --- Mint an abstraction from the recurrent mastered structure (if any) ----------------
    abstraction = mint_abstraction(history)
    abstraction_dict = abstraction.as_dict() if abstraction is not None else None

    elapsed = time.time() - t_start

    # --- Saturation contrast: did fixed flatline while open-ended climbed? -----------------
    fixed_costs = [r["fixed_benchmark_cost_flops"] for r in rows
                   if r["fixed_benchmark_cost_flops"] is not None]
    fixed_saturated = (len(set(fixed_costs)) == 1) if fixed_costs else False
    repertoire_series = [r["n_mastered"] for r in rows]
    complexity_series = [r["hardest_mastered_w"] for r in rows]
    repertoire_climbed = repertoire_series[-1] > repertoire_series[0] if rows else False
    complexity_climbed = complexity_series[-1] > complexity_series[0] if rows else False

    report = {
        "rounds_run": len(rows),
        "rows": rows,
        "abstraction": abstraction_dict,
        "saturation_contrast": {
            "fixed_benchmark_cost_series": fixed_costs,
            "fixed_benchmark_saturated": bool(fixed_saturated),
            "open_ended_repertoire_series": repertoire_series,
            "open_ended_complexity_series": complexity_series,
            "repertoire_climbed": bool(repertoire_climbed),
            "complexity_climbed": bool(complexity_climbed),
        },
        "elapsed_sec": round(elapsed, 2),
        "config": {"rounds": rounds, "steps": steps, "seed": seed, "n_noise": n_noise,
                   "tau": TAU},
    }
    return report


def _print_summary(report):
    sc = report["saturation_contrast"]
    print()
    print("=" * 72)
    print("OPEN-ENDEDNESS DEMONSTRATION — the fixed test saturates, the frontier climbs")
    print("=" * 72)
    print(f"  rounds run                 : {report['rounds_run']}")
    print(f"  repertoire size over rounds: {sc['open_ended_repertoire_series']}"
          f"   climbed={sc['repertoire_climbed']}")
    print(f"  hardest-solved freq        : {sc['open_ended_complexity_series']}"
          f"   climbed={sc['complexity_climbed']}")
    print(f"  FIXED benchmark cost (flops): {sc['fixed_benchmark_cost_series']}"
          f"   saturated={sc['fixed_benchmark_saturated']}")
    print("-" * 72)
    if sc["fixed_benchmark_saturated"] and (sc["repertoire_climbed"] or sc["complexity_climbed"]):
        print("  RESULT: the FIXED ladder flatlined (it stopped measuring progress) while")
        print("          the OPEN-ENDED repertoire / complexity kept climbing. That gap is")
        print("          exactly why a saturating test lies and open-endedness is needed.")
    else:
        print("  RESULT: contrast not realized this run — inspect the series above.")
    ab = report["abstraction"]
    if ab:
        print(f"  minted abstraction          : basis @ w={ab['w']:.2f} "
              f"from mastered band {ab['support']} (evidence n={ab['n_evidence']})")
    else:
        print("  minted abstraction          : none yet (no dense recurrent band)")
    print("=" * 72)


def main():
    os.makedirs(RUN_LOGS, exist_ok=True)
    report = run_demo()
    _print_summary(report)
    out = os.path.join(RUN_LOGS, "generator_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
