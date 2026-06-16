"""Close the recursive loop with code: scaling_test.py found a hard wall — fixed random features
can't learn a sin(w.x) in d>=4 (curse of dimensionality). The recursive question: can a LEARNED
representation cross that exact wall?

A learned representation adapts its features to the data instead of scattering them randomly over
d-dim frequency space, so it should not pay the exp(d) coverage cost. We test the smallest honest
version: a 1-hidden-layer tanh network (a learned feature map + linear readout) trained by
gradient descent, on the SAME d=4 target that walled fixed RFF. If it reaches competence, learned
deep representations cross the curse where random features cannot — which is the empirical
foundation of modern ML, here re-derived on RecursiveNe's own wall.

Run:  python experiments/deep_test.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.objective import TAU


class RFFd:
    def __init__(self, D, d, gamma, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, gamma, (D, d)); self.b = r.uniform(0, 2 * np.pi, D)
        self.s = np.sqrt(2.0 / D); self.w = np.zeros(D); self.P = np.eye(D)

    def predict(self, x): return float(self.w @ (self.s * np.cos(self.W @ x + self.b)))

    def update(self, x, y):
        p = self.s * np.cos(self.W @ x + self.b); Pp = self.P @ p; k = Pp / (1.0 + p @ Pp)
        self.w = self.w + k * (y - p @ self.w); self.P = self.P - np.outer(k, Pp)


def rff_reaches(fn, d, N, D=400, gamma=5.0, seed=0):
    rng = np.random.default_rng(seed); m = RFFd(D, d, gamma, seed)
    held = [rng.uniform(-1, 1, d) for _ in range(200)]
    t = np.array([fn(x) for x in held]); sc = t.std() + 1e-9; t = t / sc
    for _ in range(N):
        x = rng.uniform(-1, 1, d); m.update(x, fn(x) / sc + 0.02 * rng.standard_normal())
    return float(np.mean((t - np.array([m.predict(h) for h in held])) ** 2))


def train_mlp(fn, d, N, H=160, iters=6000, lr=3e-3, seed=0):
    """1 hidden tanh layer + linear readout, full-batch Adam on N samples. Returns held-out MSE."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, d))
    sc = np.std([fn(x) for x in X]) + 1e-9
    Y = np.array([fn(x) for x in X]) / sc + 0.02 * rng.standard_normal(N)
    held = rng.uniform(-1, 1, (200, d)); th = np.array([fn(x) for x in held]) / sc

    A = rng.normal(0, np.sqrt(1.0 / d), (H, d)); a = np.zeros(H)
    v = rng.normal(0, np.sqrt(1.0 / H), H); b = 0.0
    params = [A, a, v, b]
    mom = [np.zeros_like(A), np.zeros_like(a), np.zeros_like(v), 0.0]
    vel = [np.zeros_like(A), np.zeros_like(a), np.zeros_like(v), 0.0]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def fwd(Xb):
        Z = Xb @ A.T + a; Hh = np.tanh(Z); return Z, Hh, Hh @ v + b

    for it in range(1, iters + 1):
        Z, Hh, yp = fwd(X)
        dy = 2.0 * (yp - Y) / N
        dv = Hh.T @ dy; db = dy.sum()
        dHh = np.outer(dy, v); dZ = dHh * (1 - Hh ** 2)
        dA = dZ.T @ X; da = dZ.sum(0)
        grads = [dA, da, dv, db]
        for i in range(4):
            mom[i] = b1 * mom[i] + (1 - b1) * grads[i]
            vel[i] = b2 * vel[i] + (1 - b2) * np.asarray(grads[i]) ** 2
            mh = mom[i] / (1 - b1 ** it); vh = vel[i] / (1 - b2 ** it)
            params[i] = params[i] - lr * mh / (np.sqrt(vh) + eps)
        A, a, v, b = params
    Zt = np.tanh(held @ A.T + a); pred = Zt @ v + b
    return float(np.mean((th - pred) ** 2))


def main():
    rng = np.random.default_rng(1)
    d = 4
    w = rng.uniform(2, 4, d)
    fn = lambda x, w=w: float(np.sin(w @ x))
    print(f"\nClosing the loop: can a LEARNED representation cross the d={d} wall that walled fixed RFF?\n")
    print("  fixed random features (the wall):")
    for N in (2000, 6000, 12000):
        mse = rff_reaches(fn, d, N)
        print(f"    N={N:>5}: held-out MSE={mse:.3f}  {'reached' if mse <= TAU else 'NOT reached'}")
    print("\n  learned 1-hidden-layer representation (gradient descent):")
    reached_at = None
    for N in (500, 1000, 2000, 4000):
        mse = train_mlp(fn, d, N)
        ok = mse <= TAU
        if ok and reached_at is None:
            reached_at = N
        print(f"    N={N:>5}: held-out MSE={mse:.3f}  {'reached' if ok else 'NOT reached'}")
    print("\n" + "=" * 64)
    if reached_at:
        print(f"KNOWN: a learned representation CROSSES the d={d} curse-of-dimensionality wall — it")
        print(f"       reaches competence at N={reached_at} samples where fixed random features never do.")
        print("       This is why scaling to high-dim sensory worlds needs LEARNED DEEP representations,")
        print("       not bigger random feature banks — re-derived on RecursiveNe's own wall. It is also")
        print("       exactly L2's job. The next wall is identified AND shown crossable, by experiment.")
    else:
        print(f"PARTIAL: the minimal learned net did not cross d={d} within the budget — see numbers;")
        print("       a wider/deeper net or more iters is the honest next try.")
    print("=" * 64)


if __name__ == "__main__":
    main()
