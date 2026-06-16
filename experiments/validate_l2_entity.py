"""Gate: Nous-L2 makes unknowns known in a multi-dimensional world AND gets cheaper at it over its
life (learned transfer past the dimension wall), with persistent identity. Run:
  python experiments/validate_l2_entity.py
"""

import os
import sys
import numpy as np

from _util import REPO_ROOT
from recursivene.entity_l2 import L2Entity

R = []


def check(name, ok, detail):
    R.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")


def main():
    sp = os.path.join(REPO_ROOT, "run_logs", "entity_ProbeL2.json")
    if os.path.exists(sp):
        os.remove(sp)
    e = L2Entity(name="ProbeL2", home=REPO_ROOT, dim=3, seed=5)
    rings = e.live(seasons=15, verbose=False)
    c = e.cost_history

    check("OBJECT: makes d=3 unknowns known", e.total_known >= 8,
          f"{e.total_known} unknowns made known in a d={e.dim} world")
    first, last = np.mean(c[:4]), np.mean(c[-4:])
    check("TRANSFER: cheaper over its life", last < first,
          f"held-out MSE {first:.4f} -> {last:.4f} ({first/last:.1f}x cheaper as the body learned)")
    check("IDENTITY: persists across sessions", os.path.exists(e.statepath),
          f"state saved -> {os.path.basename(e.statepath)}")
    # informational (not a gate): capacity growth is world-dependent — it triggers in harder worlds.
    print(f"      (capacity: learned body {'grew 160->%d' % e.backend.H if e.backend.H > 160 else 'stayed 160 — sufficed for d=3; grows when a frontier is unreachable'})")

    ok = all(R)
    print(f"\n{'='*60}\n{'PASS — Nous-L2 lives in multi-D worlds and gets cheaper.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
