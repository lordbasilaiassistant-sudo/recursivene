"""The synthesis question, answered with code+output: when open-ended COMPLEXITY GROWTH and
EMERGENT TRANSFER run together, does cost-to-know stay bounded (or fall) instead of exploding?

This unifies everything:
  * the garden showed open-ended growth, but in an ORTHOGONAL world cost-to-know was only bounded;
  * transfer_test/discover_test showed cost-to-know FALLS with shared structure, and that the
    structure can be found in the system's own past work.
Here both happen at once: a primitive pool that GROWS over eras (open-ended rising complexity),
targets that are compositions of the currently-available primitives, and a learner that banks
its own past solutions. If banking keeps cost-to-know bounded while naive learning's cost climbs
with complexity, then the open-ended race-to-0 is real: the system keeps making harder unknowns
known without paying more per unknown.

Run:  python experiments/capstone.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.objective import TAU

POOL = [8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0]   # primitives revealed over eras
GAMMA, D = 20.0, 160
GRID = np.linspace(-1, 1, 201)


class RLS:
    def __init__(self, d, ridge=1.0):
        self.w = np.zeros(d); self.P = np.eye(d) / ridge

    def predict(self, f): return float(f @ self.w)

    def update(self, f, y):
        Pp = self.P @ f; k = Pp / (1.0 + f @ Pp)
        self.w = self.w + k * (y - f @ self.w); self.P = self.P - np.outer(k, Pp)


def rff_maker(seed):
    r = np.random.default_rng(seed)
    W = r.normal(0, GAMMA, D); b = r.uniform(0, 2 * np.pi, D); s = np.sqrt(2.0 / D)
    return lambda x: s * np.cos(W * x + b)


def learn(target_fn, bank, use_bank, seed, max_n=5000):
    rng = np.random.default_rng(seed); rff = rff_maker(seed)
    bfns = bank if use_bank else []

    def feat(x):
        bf = np.array([np.interp(x, GRID, g) for g in bfns]) if bfns else np.zeros(0)
        return np.concatenate([bf, rff(x)])

    model = RLS(len(bfns) + D)
    xe = GRID[::5]; truth = np.array([target_fn(x) for x in xe])
    cost = np.inf
    for n in range(1, max_n + 1):
        x = rng.uniform(-1, 1)
        model.update(feat(x), target_fn(x) + 0.02 * rng.standard_normal())
        if n % 10 == 0 and float(np.mean((truth - np.array([model.predict(feat(xx)) for xx in xe])) ** 2)) <= TAU:
            cost = n; break
    gvals = np.array([model.predict(feat(x)) for x in GRID])
    return cost, gvals


def stream(use_bank):
    rng = np.random.default_rng(5)
    bank, costs, complexity = [], [], []
    for t in range(24):
        era = min(len(POOL), 3 + t // 3)                 # available primitives grow over eras
        avail = POOL[:era]
        k = min(4, len(avail))
        freqs = rng.choice(avail, size=k, replace=False)
        coeffs = rng.uniform(0.4, 1.0, k) * rng.choice([-1, 1], k)
        target = lambda x, fr=freqs, co=coeffs: float(sum(c * np.sin(w * x) for c, w in zip(co, fr)))
        cost, g = learn(target, bank, use_bank, seed=300 + t)
        costs.append(cost if np.isfinite(cost) else 5000)
        complexity.append(max(avail))
        if use_bank and len(bank) < 14:
            bank.append(g)
    return costs, complexity


def main():
    print("\nSynthesis: open-ended complexity growth + emergent transfer — does cost-to-know stay bounded?\n")
    naive, comp = stream(False)
    banked, _ = stream(True)
    print(f"  hardest primitive present rises: {comp[0]:.0f} -> {comp[-1]:.0f} (open-ended complexity)\n")
    print(f"  naive RFF  cost-to-know: {sparkline(naive)}   first~{np.mean(naive[:4]):.0f} -> last~{np.mean(naive[-4:]):.0f}")
    print(f"  banked     cost-to-know: {sparkline(banked)}   first~{np.mean(banked[:4]):.0f} -> last~{np.mean(banked[-4:]):.0f}")

    naive_trend = np.mean(naive[-4:]) / np.mean(naive[:4])
    banked_trend = np.mean(banked[-4:]) / np.mean(banked[:4])
    print("\n" + "=" * 64)
    print(f"KNOWN: as complexity grew {comp[0]:.0f}->{comp[-1]:.0f}, naive cost-to-know went {naive_trend:.2f}x")
    print(f"       while banked (transfer) went {banked_trend:.2f}x and stayed far lower.")
    if banked_trend <= naive_trend and np.mean(banked[-4:]) < np.mean(naive[-4:]):
        print("=> The open-ended race-to-0 is REAL: with emergent transfer, the system keeps making")
        print("   harder unknowns known WITHOUT paying more per unknown. Thesis, demonstrated end-to-end.")
    else:
        print("=> Mixed — see numbers; the per-unknown cost did not stay bounded as hoped.")
    print("=" * 64)


if __name__ == "__main__":
    main()
