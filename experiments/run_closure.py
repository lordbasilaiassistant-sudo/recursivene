"""Run the full three-level closure once and persist its artifacts.

This is the slow, autonomous run (no human in the inner loop): config ratchet, then the
proposer self-editing the harness, then self-editing the proposer itself — every edit
sandboxed and gated by the protected invariant + kill switch. Outputs:
  run_logs/closure_summary.json   — what each stage did, for the validate gate to read
  run_logs/monotonicity.jsonl     — every accepted/rejected edit, timestamped
  vitals/parent.jsonl, child.jsonl— heartbeats

Run:  python experiments/run_closure.py
"""

import json
import os

from _util import REPO_ROOT  # noqa: F401  (adds repo root to path)
from recursivene.closure.driver import run_closure

BLOATED = {"policy": "lp", "n_features": 256, "gamma": 8.0, "hist": 64,
           "min_lp": 16, "epsilon": 0.15}


def main():
    summary = run_closure(REPO_ROOT, BLOATED, stage1_gens=6, stage_edit_rounds=3,
                          max_seconds=2400, verbose=True)
    out = os.path.join(REPO_ROOT, "run_logs", "closure_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")
    for st in ("model", "harness", "proposer"):
        s = summary["stages"].get(st, {})
        print(f"  stage {st:9s} passed={s.get('passed')} accepted={s.get('accepted', s.get('best_cost'))}")


if __name__ == "__main__":
    main()
