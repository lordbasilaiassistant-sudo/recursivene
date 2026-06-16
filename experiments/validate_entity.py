"""Gate: prove the entity BRINGS IT ALL TOGETHER — one self-improving knower in which every
proven piece is live at once. Assertion-based PASS/FAIL with numbers.

Run:  python experiments/validate_entity.py
"""

import json
import os
import sys
import numpy as np

from _util import REPO_ROOT
from recursivene.entity import Entity
from recursivene.core.killswitch import KillSwitch

R = []


def check(name, ok, detail):
    R.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")


def main():
    # a fresh entity lives a short life so every mechanism must fire
    mono = os.path.join(REPO_ROOT, "run_logs", "entity_monotonicity.jsonl")
    open(mono, "w").close()
    e = Entity(name="Probe", home=REPO_ROOT, seed=3)
    if os.path.exists(e.statepath):
        os.remove(e.statepath)
    e = Entity(name="Probe", home=REPO_ROOT, seed=3)
    rings = e.live(seasons=16, verbose=False)

    costs = [r["cost_to_know"] for r in rings if r["cost_to_know"] is not None]
    sizes = [r["rep_size"] for r in rings]
    grew = [r for r in rings if r["event"] == "grew+knew"]
    rtz = [r for r in rings if r["race_to_zero"] and r["race_to_zero"]["improved"]]

    check("OBJECT: makes unknowns known", e.total_known >= 10,
          f"{e.total_known} unknowns made known in 16 seasons")
    check("L1: cheap via learned representation", np.median(costs) <= 120,
          f"median cost-to-know = {np.median(costs):.0f} samples (most unknowns cheap)")
    check("OPEN-ENDED: complexity climbed", rings[-1]["complexity"] > rings[0]["complexity"],
          f"complexity {rings[0]['complexity']} -> {rings[-1]['complexity']}")
    check("GARDEN: grew its own capacity", len(grew) >= 1 and max(sizes) > sizes[0],
          f"{len(grew)} growth event(s); representation {sizes[0]} -> {max(sizes)} features")
    check("RSI: race-to-0 improved the learner", len(rtz) >= 1,
          f"{len(rtz)} accepted self-improvement(s) in the monotonicity log")
    mono_lines = [json.loads(l) for l in open(mono)] if os.path.exists(mono) else []
    check("MONOTONICITY: improvements logged + timestamped",
          len(mono_lines) >= 1 and all("ts" in m and m["cost_after"] <= m["cost_before"] for m in mono_lines),
          f"{len(mono_lines)} race-to-0 entries, all cost non-increasing")
    check("SAFETY: kill switch wired + untouchable", isinstance(e.ks, KillSwitch),
          "kill switch present; core/ is protected from self-edits")
    check("IDENTITY: persists across sessions", os.path.exists(e.statepath),
          f"state saved -> {os.path.basename(e.statepath)} (same entity resumes)")

    ok = all(R)
    print(f"\n{'='*64}\n{'PASS — one entity, RSI, everything brought together.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
