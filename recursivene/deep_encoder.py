"""L2 — a LEARNED representation that adapts to the data, so the entity can make unknowns known
in worlds whose structure is NOT a 1-D frequency.

L1 (the SpectralEncoder) discovers FREQUENCIES — perfect for band-limited 1-D worlds, useless when
the structure is higher-dimensional or compositional. deep_test.py proved that a learned 1-hidden-
layer representation CROSSES the curse-of-dimensionality wall where fixed/spectral features cannot
(competence at N~500 on a d=4 target that fixed RFF never reaches). This module is that learner,
generalized into a clean L2 backend: a small tanh MLP trained by Adam, whose hidden layer LEARNS the
features the data actually needs instead of fixing them in advance.

It exposes the same contract the rest of the project speaks (fit / predict / cost_to_know), so it
drops in beside L0/L1 as the world-model encoder for richer worlds. No claim of scale — this extends
the entity's reach from 1-D frequencies to genuine multi-dimensional / sensory structure, with the
same held-out, cost-to-know honesty discipline as everything else.
"""

import numpy as np


class DeepEncoder:
    """1-hidden-layer tanh MLP + linear readout, full-batch Adam. The hidden layer IS the learned
    representation; the readout is the cheap per-task head. Works for any input dimension."""

    def __init__(self, in_dim, hidden=128, lr=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.H, self.lr = int(in_dim), int(hidden), float(lr)
        self.A = rng.normal(0, np.sqrt(1.0 / self.d), (self.H, self.d))
        self.a = np.zeros(self.H)
        self.v = rng.normal(0, np.sqrt(1.0 / self.H), self.H)
        self.b = 0.0

    def _forward(self, X):
        Z = X @ self.A.T + self.a
        Hh = np.tanh(Z)
        return Hh, Hh @ self.v + self.b

    def predict(self, X):
        return self._forward(np.atleast_2d(X))[1]

    def fit(self, X, Y, iters=6000):
        """Full-batch Adam on (X, Y). Returns final training MSE."""
        params = [self.A, self.a, self.v, self.b]
        mom = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]
        vel = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]
        b1, b2, eps, N = 0.9, 0.999, 1e-8, len(X)
        for it in range(1, iters + 1):
            Hh, yp = self._forward(X)
            dy = 2.0 * (yp - Y) / N
            dv = Hh.T @ dy
            db = dy.sum()
            dZ = np.outer(dy, self.v) * (1 - Hh ** 2)
            dA = dZ.T @ X
            da = dZ.sum(0)
            grads = [dA, da, dv, db]
            for i in range(4):
                mom[i] = b1 * mom[i] + (1 - b1) * grads[i]
                vel[i] = b2 * vel[i] + (1 - b2) * np.asarray(grads[i]) ** 2
                mh = mom[i] / (1 - b1 ** it)
                vh = vel[i] / (1 - b2 ** it)
                params[i] = params[i] - self.lr * mh / (np.sqrt(vh) + eps)
            self.A, self.a, self.v, self.b = params
        return float(np.mean((self._forward(X)[1] - Y) ** 2))


class SharedDeepBackend:
    """L2 entity backend: a PERSISTENT learned body (the hidden representation) shared across a stream
    of unknowns, with a fresh linear head per unknown. The body accumulates features useful across the
    world's structure, so later unknowns get cheaper to make known — learned transfer in multi-D, the
    same flattening effect L1 gave in 1-D, now in a representation that also crosses the dimension wall.
    (Honest caveat: continual fine-tuning drifts the body toward recent tasks — mild forgetting — which
    is fine for measuring per-new-unknown cost, the quantity that matters here.)"""

    def __init__(self, in_dim, hidden=128, lr=2e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.H, self.lr = int(in_dim), int(hidden), float(lr)
        self.A = rng.normal(0, np.sqrt(1.0 / self.d), (self.H, self.d))
        self.a = np.zeros(self.H)
        self._mA = np.zeros_like(self.A); self._vA = np.zeros_like(self.A)
        self._ma = np.zeros_like(self.a); self._va = np.zeros_like(self.a)
        self._t = 0

    def fit_target(self, target_fn, N=600, iters=2500, obs_noise=0.02, seed=0):
        """Fine-tune the PERSISTENT body + a fresh head on this unknown; return held-out MSE at a
        fixed budget. Lower over the stream = the accumulated representation made it cheaper to know."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, (N, self.d))
        sc = np.std([target_fn(x) for x in X]) + 1e-9
        Y = np.array([target_fn(x) for x in X]) / sc + obs_noise * rng.standard_normal(N)
        held = rng.uniform(-1, 1, (400, self.d)); th = np.array([target_fn(x) for x in held]) / sc
        v = rng.normal(0, np.sqrt(1.0 / self.H), self.H); b = 0.0
        mv = np.zeros_like(v); vv = np.zeros_like(v); mb = vb = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(iters):
            self._t += 1
            Z = X @ self.A.T + self.a; Hh = np.tanh(Z); yp = Hh @ v + b
            dy = 2.0 * (yp - Y) / N
            dv = Hh.T @ dy; db = dy.sum()
            dZ = np.outer(dy, v) * (1 - Hh ** 2)
            dA = dZ.T @ X; da = dZ.sum(0)
            for (p, g, m, vel) in (("A", dA, "_mA", "_vA"), ("a", da, "_ma", "_va")):
                M = getattr(self, m); V = getattr(self, vel)
                M[:] = b1 * M + (1 - b1) * g; V[:] = b2 * V + (1 - b2) * g ** 2
                setattr(self, p, getattr(self, p) - self.lr * (M / (1 - b1 ** self._t))
                        / (np.sqrt(V / (1 - b2 ** self._t)) + eps))
            mv = b1 * mv + (1 - b1) * dv; vv = b2 * vv + (1 - b2) * dv ** 2
            v = v - self.lr * (mv / (1 - b1 ** self._t)) / (np.sqrt(vv / (1 - b2 ** self._t)) + eps)
            mb = b1 * mb + (1 - b1) * db; vb = b2 * vb + (1 - b2) * db ** 2
            b = b - self.lr * (mb / (1 - b1 ** self._t)) / (np.sqrt(vb / (1 - b2 ** self._t)) + eps)
        pred = np.tanh(held @ self.A.T + self.a) @ v + b
        return float(np.mean((pred - th) ** 2))

    def grow(self, add=32, seed=0):
        """Widen the learned body (more hidden units) when the current representation can't reach a
        frontier — the garden's capacity-growth, at the representation level. Adam state grows with it;
        the per-target head is fresh each call so it auto-sizes to the new width."""
        rng = np.random.default_rng(seed)
        nA = rng.normal(0, np.sqrt(1.0 / self.d), (int(add), self.d))
        self.A = np.vstack([self.A, nA]); self.a = np.concatenate([self.a, np.zeros(int(add))])
        self._mA = np.vstack([self._mA, np.zeros((int(add), self.d))])
        self._vA = np.vstack([self._vA, np.zeros((int(add), self.d))])
        self._ma = np.concatenate([self._ma, np.zeros(int(add))])
        self._va = np.concatenate([self._va, np.zeros(int(add))])
        self.H += int(add)
        return self.H

    def state(self):
        return {"A": self.A.tolist(), "a": self.a.tolist(), "H": self.H, "d": self.d}

    def load_state(self, s):
        self.A = np.array(s["A"]); self.a = np.array(s["a"]); self.H = s["H"]
        self._mA = np.zeros_like(self.A); self._vA = np.zeros_like(self.A)
        self._ma = np.zeros_like(self.a); self._va = np.zeros_like(self.a)


def cost_to_know(target_fn, in_dim, tau, sizes=(250, 500, 1000, 2000, 4000),
                 hidden=128, obs_noise=0.02, seed=0, iters=5000):
    """Smallest number of observations a fresh DeepEncoder needs to bring `target_fn` (a function of
    an in_dim vector on [-1,1]^d) below held-out MSE `tau`. Returns (N or inf, final_heldout_mse).
    Held-out, not training — generalization is the signal (the lesson from exp_2d)."""
    rng = np.random.default_rng(seed)
    held = rng.uniform(-1, 1, (400, in_dim))
    th = np.array([target_fn(x) for x in held])
    sc = th.std() + 1e-9
    th = th / sc
    last = np.inf
    for N in sizes:
        enc = DeepEncoder(in_dim, hidden=hidden, seed=seed)
        X = rng.uniform(-1, 1, (N, in_dim))
        Y = np.array([target_fn(x) for x in X]) / sc + obs_noise * rng.standard_normal(N)
        enc.fit(X, Y, iters=iters)
        last = float(np.mean((enc.predict(held) - th) ** 2))
        if last <= tau:
            return N, last
    return np.inf, last
