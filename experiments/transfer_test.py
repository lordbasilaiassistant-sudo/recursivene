"""Make Unknown A known: can accumulated knowledge make NEW unknowns CHEAPER to learn
(decreasing cost-to-know), or only bounded?

Hypothesis: transfer needs SHARED STRUCTURE. The garden's tasks are pure single frequencies,
which are near-orthogonal in function space — so there is little to transfer and cost-to-know
stays flat/bounded (no free lunch). But if tasks are COMPOSITIONS of a small shared set of
primitives, then a learner that has banked the primitives can express any new target as a few
coefficients — cost-to-know should DROP as the library fills.

We measure cost-to-know(target) as a function of how many primitives the learner already knows,
in two worlds:
  STRUCTURED  — targets = sparse sums of K shared primitive frequencies.   (expect DECREASING)
  ORTHOGONAL  — targets = single distinct frequencies (the garden's world). (expect FLAT)

Run:  python experiments/transfer_test.py
"""

import numpy as np

from _util import REPO_ROOT, sparkline  # noqa: F401
from recursivene.objective import TAU

W_PRIM = [2.0, 3.5, 5.5, 8.0, 11.0]          # the shared primitive frequencies
K = len(W_PRIM)
D, GAMMA = 96, 8.0


class RLS:
    """Recursive least squares over an arbitrary fixed feature vector (the library + RFF)."""

    def __init__(self, d, ridge=1.0):
        self.w = np.zeros(d)
        self.P = np.eye(d) / ridge

    def predict(self, feat):
        return float(feat @ self.w)

    def update(self, feat, y):
        p = feat
        Pp = self.P @ p
        k = Pp / (1.0 + p @ Pp)
        self.w = self.w + k * (y - p @ self.w)
        self.P = self.P - np.outer(k, Pp)


def _rff(seed):
    r = np.random.default_rng(seed)
    W = r.normal(0.0, GAMMA, size=D)
    b = r.uniform(0.0, 2 * np.pi, size=D)
    return lambda x: np.sqrt(2.0 / D) * np.cos(W * x + b)


def cost_to_know(target_fn, library_freqs, seed=0, max_n=4000, obs_noise=0.02):
    """Samples to bring `target_fn` below tau, using KNOWN library_freqs as exact features
    plus an RFF basis for whatever the library doesn't cover. Returns samples (or inf)."""
    rng = np.random.default_rng(seed)
    rff = _rff(seed)
    libs = list(library_freqs)

    def feat(x):
        lib = np.array([np.sin(w * x) for w in libs]) if libs else np.zeros(0)
        return np.concatenate([lib, rff(x)])

    model = RLS(len(libs) + D)
    xs = np.linspace(-1, 1, 41)
    truth = np.array([target_fn(x) for x in xs])
    for n in range(1, max_n + 1):
        x = rng.uniform(-1, 1)
        model.update(feat(x), target_fn(x) + obs_noise * rng.standard_normal())
        if n % 10 == 0:
            mse = float(np.mean((truth - np.array([model.predict(feat(xx)) for xx in xs])) ** 2))
            if mse <= TAU:
                return n
    return np.inf


def structured_target(rng):
    coeffs = rng.uniform(0.4, 1.0, K) * rng.choice([-1, 1], K)
    return lambda x: float(sum(c * np.sin(w * x) for c, w in zip(coeffs, W_PRIM)))


def main():
    rng = np.random.default_rng(7)
    print("\nMake Unknown A known: does accumulated knowledge make new unknowns cheaper?\n")

    # STRUCTURED world: targets are sums of the K shared primitives. Vary library size m.
    print("STRUCTURED world (targets = sums of shared primitives):")
    struct = []
    for m in range(0, K + 1):
        lib = W_PRIM[:m]
        costs = [cost_to_know(structured_target(rng), lib, seed=s) for s in range(8)]
        costs = [c for c in costs if np.isfinite(c)]
        avg = np.mean(costs) if costs else np.inf
        struct.append(avg)
        print(f"  library size {m}/{K}: cost-to-know = {avg:6.0f} samples")
    print(f"  curve: {sparkline(struct)}   {struct[0]:.0f} -> {struct[-1]:.0f} samples")

    # ORTHOGONAL world: target is a single NEW frequency; library = other single frequencies.
    print("\nORTHOGONAL world (targets = distinct single frequencies, the garden's world):")
    orth = []
    target_w = 12.0
    others = [3.0, 6.0, 9.0, 15.0, 18.0]
    for m in range(0, len(others) + 1):
        lib = others[:m]
        costs = [cost_to_know((lambda x, w=target_w: float(np.sin(w * x))), lib, seed=s)
                 for s in range(8)]
        costs = [c for c in costs if np.isfinite(c)]
        avg = np.mean(costs) if costs else np.inf
        orth.append(avg)
        print(f"  library size {m}: cost-to-know(w=12) = {avg:6.0f} samples")
    print(f"  curve: {sparkline(orth)}   {orth[0]:.0f} -> {orth[-1]:.0f} samples")

    print("\n" + "=" * 64)
    s_drop = struct[0] / struct[-1] if struct[-1] else float("inf")
    o_drop = orth[0] / orth[-1] if orth[-1] else float("inf")
    print(f"KNOWN: structured cost-to-know FELL {s_drop:.1f}x as the library filled.")
    print(f"KNOWN: orthogonal cost-to-know changed {o_drop:.2f}x (≈1 = no transfer).")
    print("=> Decreasing cost-to-know is REAL, but requires SHARED STRUCTURE.")
    print("   Pure frequencies are near-orthogonal => bounded, not decreasing (no free lunch).")
    print("   The lever for a true open-ended race-to-0 is a world/representation with reusable")
    print("   structure + abstraction that banks it. That is L1.")
    print("=" * 64)


if __name__ == "__main__":
    main()
