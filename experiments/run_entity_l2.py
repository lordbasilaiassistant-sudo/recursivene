"""L2 entity: Nous's world-model backend made LEARNED and multi-dimensional. A stream of d=3
compositional unknowns that share hidden structure; the entity makes each known and — because its
learned body PERSISTS and accumulates that structure — gets CHEAPER at it over time (learned transfer
in a multi-D world, the L1 flattening effect now in a representation that crosses the dimension wall).

Persistent body vs a fresh-body baseline (no memory): persistent should TREND DOWN, fresh stays flat.

Run:  python experiments/run_entity_l2.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.deep_encoder import SharedDeepBackend

D = 3
SHARED_W = None


def make_stream(n, seed=0):
    """Targets share a hidden pool of d-dim frequency vectors (structure to transfer)."""
    rng = np.random.default_rng(seed)
    pool = [rng.uniform(1.5, 3.0, D) for _ in range(6)]   # the shared hidden structure
    out = []
    for _ in range(n):
        idx = rng.choice(6, size=3, replace=False)
        coeffs = rng.uniform(-1, 1, 3)
        ws = [pool[i] for i in idx]
        out.append(lambda x, ws=ws, co=coeffs: float(sum(c * np.sin(w @ x) for c, w in zip(co, ws))))
    return out


def main():
    print("=" * 78)
    print(f"L2 ENTITY — making unknowns known in a d={D} world, ever more cheaply (learned transfer)")
    print("=" * 78)
    stream = make_stream(12, seed=1)

    persistent = SharedDeepBackend(D, hidden=160, seed=0)
    fresh_mses, pers_mses = [], []
    for t, target in enumerate(stream):
        # persistent body carries over across unknowns (accumulates the world's structure)
        pm = persistent.fit_target(target, N=600, iters=1800, seed=300 + t)
        # baseline: a brand-new body every time (no memory) — the no-transfer control
        fb = SharedDeepBackend(D, hidden=160, seed=t)
        fm = fb.fit_target(target, N=600, iters=1800, seed=300 + t)
        pers_mses.append(pm); fresh_mses.append(fm)
        print(f"  unknown {t:>2}: persistent body held-out MSE {pm:.4f}   |   fresh body {fm:.4f}")

    p0, p1 = np.mean(pers_mses[:3]), np.mean(pers_mses[-3:])
    f0, f1 = np.mean(fresh_mses[:3]), np.mean(fresh_mses[-3:])
    print("\n  persistent (transfer): " + sparkline(pers_mses) + f"   {p0:.4f} -> {p1:.4f}")
    print("  fresh (no memory)    : " + sparkline(fresh_mses) + f"   {f0:.4f} -> {f1:.4f}")
    print("=" * 78)
    drop = p0 / p1 if p1 > 0 else float("inf")
    if p1 < p0 * 0.8 and p1 < f1:
        print(f"KNOWN: the entity got {drop:.1f}x CHEAPER at making d={D} unknowns known as its learned")
        print("       body accumulated the world's shared structure; a fresh body (no memory) stayed flat.")
        print("       L2 transfer in a multi-D world — the same arrow as L1, now past the dimension wall.")
    else:
        print(f"PARTIAL: persistent {p0:.4f}->{p1:.4f}, fresh {f0:.4f}->{f1:.4f} — see numbers.")
    print("       Honest scope: toy d=3 world; mild continual-learning forgetting; illustrates the mechanism.")
    print("=" * 78)


if __name__ == "__main__":
    main()
