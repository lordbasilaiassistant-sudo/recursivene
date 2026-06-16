"""Answer A with code+output: does decreasing cost-to-know EMERGE without handing the system
its primitives?

First design (kept in git history of this file's intent) failed honestly: a dense correlated
sine bank smears weight so pruning can't concentrate, and smooth low-freq targets are already
cheap over the full bank — no expensive baseline to improve on. Fix, both flaws:
  * make the naive case genuinely expensive: HIGH-frequency primitives an RFF basis fits only
    slowly;
  * use a cleaner emergence mechanism that needs no frequency detector — BANK PAST SOLUTIONS as
    features. Every target is a sum of the same hidden primitives, so once enough past solutions
    span that K-dim subspace, ANY new target is an exact linear combination of them -> learnable
    in ~K samples. The library (the span of past solutions) IS the discovered abstraction.

If per-target cost-to-know falls from RFF-scratch toward ~K samples as the bank fills, transfer
is EMERGENT (the system found the structure in its own past work), answering A.

Run:  python experiments/discover_test.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.objective import TAU

PRIMS = [8.0, 12.0, 16.0, 20.0]      # hidden shared primitives (high freq -> RFF is slow)
K = len(PRIMS)
GAMMA, D = 16.0, 128
GRID = np.linspace(-1, 1, 201)       # for banking a learned function as interpolable values


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


def make_target(rng):
    c = rng.uniform(0.4, 1.0, K) * rng.choice([-1, 1], K)
    return lambda x: float(sum(ci * np.sin(w * x) for ci, w in zip(c, PRIMS)))


def learn(target_fn, bank, use_bank=True, seed=0, max_n=4000, obs_noise=0.02):
    """Fit target over [banked past-solution features] + RFF. Returns (cost_to_know, grid_values)."""
    rng = np.random.default_rng(seed)
    rff = rff_maker(seed)
    bfns = bank if use_bank else []

    def feat(x):
        bf = np.array([np.interp(x, GRID, g) for g in bfns]) if bfns else np.zeros(0)
        return np.concatenate([bf, rff(x)])

    model = RLS(len(bfns) + D)
    truth = np.array([target_fn(x) for x in GRID[::5]])
    xe = GRID[::5]
    cost = np.inf
    for n in range(1, max_n + 1):
        x = rng.uniform(-1, 1)
        model.update(feat(x), target_fn(x) + obs_noise * rng.standard_normal())
        if n % 10 == 0:
            pred = np.array([model.predict(feat(xx)) for xx in xe])
            if float(np.mean((truth - pred) ** 2)) <= TAU:
                cost = n; break
    gvals = np.array([model.predict(feat(x)) for x in GRID])    # bank this solution
    return cost, gvals


def run_stream(use_bank, label):
    rng = np.random.default_rng(11)
    bank, costs = [], []
    for t in range(16):
        cost, gvals = learn(make_target(rng), bank, use_bank=use_bank, seed=200 + t)
        costs.append(cost)
        if use_bank and len(bank) < 2 * K:        # keep a compact spanning set
            bank.append(gvals)
    fin = [c if np.isfinite(c) else max_finite(costs) for c in costs]
    print(f"  {label:16s}: {sparkline(fin)}   first~{np.mean(fin[:3]):.0f} -> last~{np.mean(fin[-3:]):.0f} samples")
    return fin


def max_finite(costs):
    f = [c for c in costs if np.isfinite(c)]
    return max(f) * 1.5 if f else 4000


def main():
    print("\nAnswer A: does decreasing cost-to-know EMERGE from the system's own past work?\n")
    print(f"  hidden primitives: {PRIMS} (K={K}); RFF gamma={GAMMA}, D={D}\n")
    no = run_stream(False, "RFF-only (naive)")
    yes = run_stream(True, "bank past solns")
    drop = np.mean(yes[:3]) / np.mean(yes[-3:]) if np.mean(yes[-3:]) else float("inf")
    flat = np.mean(no[:3]) / np.mean(no[-3:]) if np.mean(no[-3:]) else float("inf")
    print("\n" + "=" * 64)
    if drop > 1.5 and np.mean(yes[-3:]) < np.mean(no[-3:]):
        print(f"KNOWN: banking past solutions made new unknowns {drop:.1f}x cheaper over the stream")
        print(f"       (naive RFF stays flat at {flat:.1f}x). The K-dim shared subspace was found in")
        print(f"       the system's OWN past work — emergent abstraction. Answer A: yes, with structure.")
    else:
        print(f"PARTIAL: bank drop {drop:.1f}x, naive {flat:.1f}x — see numbers above.")
    print("=" * 64)


if __name__ == "__main__":
    main()
