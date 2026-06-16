"""Answer the L1 question with code+output: does a LEARNED representation flatten the
cost-to-complexity curve that capstone.py showed rising?

Same open-ended stream as capstone (compositional targets, max frequency climbing 14->29), but
the learner uses the shared SpectralEncoder (L1), which discovers the data's frequencies from its
own buffer and represents them at fixed cost. Compared head-to-head with the fixed-RFF learner.
If the L1 learner's cost-to-know stays BOUNDED as complexity climbs while fixed RFF rises, then
L1 is the missing piece — the open-ended race-to-0 becomes reachable.

Run:  python experiments/l1_test.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.objective import TAU
from recursivene.encoder import SpectralEncoder

POOL = [8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0]
GRID = np.linspace(-1, 1, 201)


class RLS:
    def __init__(self, d, ridge=1.0):
        self.w = np.zeros(d); self.P = np.eye(d) / ridge

    def predict(self, f): return float(f @ self.w)

    def update(self, f, y):
        Pp = self.P @ f; k = Pp / (1.0 + f @ Pp)
        self.w = self.w + k * (y - f @ self.w); self.P = self.P - np.outer(k, Pp)


def fixed_rff(seed, D=200, gamma=8.0):
    r = np.random.default_rng(seed)
    W = r.normal(0, gamma, D); b = r.uniform(0, 2 * np.pi, D); s = np.sqrt(2.0 / D)
    return (lambda x: s * np.cos(W * x + b)), D


def cost_to_know(target_fn, feat, dim, seed, max_n=5000, encoder=None):
    rng = np.random.default_rng(seed)
    model = RLS(dim)
    xe = GRID[::5]; truth = np.array([target_fn(x) for x in xe])
    cost = np.inf
    for n in range(1, max_n + 1):
        x = rng.uniform(-1, 1); y = target_fn(x) + 0.02 * rng.standard_normal()
        model.update(feat(x), y)
        if encoder is not None:
            encoder.observe(x, y)
        if n % 10 == 0 and float(np.mean((truth - np.array([model.predict(feat(xx)) for xx in xe])) ** 2)) <= TAU:
            cost = n; break
    return cost


def stream(mode):
    rng = np.random.default_rng(5)
    costs, comp = [], []
    enc = SpectralEncoder(seed=0) if mode == "L1" else None
    rff_feat, rff_dim = fixed_rff(0)
    for t in range(24):
        era = min(len(POOL), 3 + t // 3)
        avail = POOL[:era]
        k = min(4, len(avail))
        fr = rng.choice(avail, size=k, replace=False)
        co = rng.uniform(0.4, 1.0, k) * rng.choice([-1, 1], k)
        target = lambda x, fr=fr, co=co: float(sum(c * np.sin(w * x) for c, w in zip(co, fr)))
        if mode == "L1":
            enc.discover()                                   # adapt representation to what it has seen
            cost = cost_to_know(target, enc.phi, enc.dim(), seed=400 + t, encoder=enc)
        else:
            cost = cost_to_know(target, rff_feat, rff_dim, seed=400 + t)
        costs.append(cost if np.isfinite(cost) else 5000)
        comp.append(max(avail))
    return costs, comp, (enc.freqs if enc else None)


def main():
    print("\nL1 question: does a learned representation flatten the cost-to-complexity curve?\n")
    fixed, comp, _ = stream("fixed")
    l1, _, freqs = stream("L1")
    print(f"  complexity (hardest freq present): {comp[0]:.0f} -> {comp[-1]:.0f}\n")
    print(f"  fixed RFF  cost-to-know: {sparkline(fixed)}   first~{np.mean(fixed[:4]):.0f} -> last~{np.mean(fixed[-4:]):.0f}")
    print(f"  L1 encoder cost-to-know: {sparkline(l1)}   first~{np.mean(l1[:4]):.0f} -> last~{np.mean(l1[-4:]):.0f}")
    if freqs is not None:
        disc = sorted(round(f, 1) for f in freqs if f >= 7)
        print(f"\n  L1 discovered high frequencies: {disc[:10]}")
        print(f"  (true pool: {POOL})")

    fixed_trend = np.mean(fixed[-4:]) / np.mean(fixed[:4])
    l1_trend = np.mean(l1[-4:]) / np.mean(l1[:4])
    print("\n" + "=" * 64)
    print(f"KNOWN: as complexity grew {comp[0]:.0f}->{comp[-1]:.0f}, fixed RFF cost-to-know went {fixed_trend:.2f}x,")
    print(f"       L1 encoder went {l1_trend:.2f}x and ended ~{np.mean(fixed[-4:]) / max(np.mean(l1[-4:]), 1):.1f}x cheaper.")
    if l1_trend < fixed_trend * 0.85 and np.mean(l1[-4:]) < np.mean(fixed[-4:]):
        print("=> L1 FLATTENS the cost-to-complexity curve. The blocker capstone.py found is removed:")
        print("   a learned representation keeps new unknowns affordable as complexity climbs. Answered.")
    else:
        print("=> L1 did not flatten it as hoped — numbers above. Honest result; next diagnosis below.")
    print("=" * 64)


if __name__ == "__main__":
    main()
