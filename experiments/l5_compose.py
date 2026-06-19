"""L5 honesty probe — is zero-shot composition REAL, or just a linearity artifact?

KNOWN #22 showed grounded symbols compose zero-shot (name UNSEEN combinations) — but the grounding map
was LINEAR, so composition falls out of linear algebra for free. A skeptic is right to ask: is that
systematic generalization, or arithmetic? This pits two groundings on the SAME held-out-combo test:

  LINEAR     — symbol<->latent via ridge regression (composition automatic).
  NONLINEAR  — symbol<->latent via a small tanh MLP trained by SGD (composition NOT automatic; the
               net must LEARN compositional structure to generalize to unseen combos).

If the NONLINEAR grounding also generalizes to unseen combinations, composition is REAL systematic
generalization, not a linearity trick. If it collapses to chance, the earlier result was arithmetic and
we say so. Either way it's the honest answer. numpy only; laptop seconds.

Run:  python experiments/l5_compose.py
"""

import sys
import math
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.language import RFF, perceive

PRIMS = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
K = len(PRIMS); D = 128


def sf(S):
    return lambda x: float(sum(math.sin(PRIMS[k] * x) for k in S))


def mh(S):
    return np.array([1.0 if k in S else 0.0 for k in range(K)])


def dataset(rff):
    """(latent, multihot) pairs: singles (several perceptions each) + a SUBSET of 2-combos for train;
    the OTHER half of 2-combos are held out (never seen paired)."""
    X, Y = [], []
    for k in range(K):
        for r in range(6):
            X.append(perceive(sf([k]), rff, n=120, seed=1000 * k + r)); Y.append(mh([k]))
    allp = [(a, b) for a in range(K) for b in range(a + 1, K)]
    rng = np.random.default_rng(1); rng.shuffle(allp)
    tr, ho = allp[: len(allp) // 2], allp[len(allp) // 2:]
    for i, (a, b) in enumerate(tr):
        for r in range(3):
            X.append(perceive(sf([a, b]), rff, n=160, seed=50000 + 10 * i + r)); Y.append(mh([a, b]))
    return np.array(X), np.array(Y), ho


class MLP:
    """Tiny 1-hidden tanh MLP, SGD. Generic in->out regression (sigmoid out for multi-label naming)."""
    def __init__(self, din, dout, hidden=64, lr=0.05, seed=0, out="sigmoid"):
        r = np.random.default_rng(seed); self.out = out
        self.W1 = r.normal(0, 1 / math.sqrt(din), (din, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1 / math.sqrt(hidden), (hidden, dout)); self.b2 = np.zeros(dout); self.lr = lr

    def _fwd(self, X):
        h = np.tanh(X @ self.W1 + self.b1); o = h @ self.W2 + self.b2
        return (h, 1 / (1 + np.exp(-o))) if self.out == "sigmoid" else (h, o)

    def fit(self, X, Y, iters=4000, bs=32):
        n = len(X); rng = np.random.default_rng(0)
        for _ in range(iters):
            idx = rng.choice(n, min(bs, n), replace=False); xb, yb = X[idx], Y[idx]
            h, o = self._fwd(xb); g = (o - yb) / len(xb)
            gW2 = h.T @ g; gb2 = g.sum(0); gh = (g @ self.W2.T) * (1 - h ** 2)
            gW1 = xb.T @ gh; gb1 = gh.sum(0)
            self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2; self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1

    def __call__(self, x):
        return self._fwd(x[None])[1][0]


def linear_fit(X, Y, ridge=1.0):
    Wp = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y).T   # latent->symbol
    return Wp


def evaluate(name_fn, ho, rff):
    ok = 0
    for j, (a, b) in enumerate(ho):
        th = perceive(sf([a, b]), rff, n=160, seed=70000 + j)
        top2 = set(np.argsort(name_fn(th))[-2:].tolist())
        ok += (top2 == {a, b})
    return ok / len(ho)


def main():
    print("\nL5 HONESTY — is zero-shot composition REAL or a linearity artifact?\n")
    rff = RFF(D=D, gamma=12.0, seed=0)
    X, Y, ho = dataset(rff)
    chance = 1.0 / (K * (K - 1) / 2)
    print(f"  {K} primitives, {len(X)} train pairs (singles + half the 2-combos); {len(ho)} UNSEEN combos held out; chance={chance:.0%}\n")

    Wp = linear_fit(X, Y)
    lin = evaluate(lambda th: Wp @ th, ho, rff)

    mlp = MLP(D, K, hidden=64, lr=0.05, seed=0, out="sigmoid"); mlp.fit(X, Y, iters=6000)
    non = evaluate(lambda th: mlp(th), ho, rff)

    print(f"  chance             : {chance:.0%}")
    print(f"  LINEAR grounding   : zero-shot composition {lin:.0%}  ({lin/chance:.0f}x chance)")
    print(f"  NONLINEAR (MLP)    : zero-shot composition {non:.0%}  ({non/chance:.0f}x chance)")
    print()
    # REAL systematic generalization if the NONLINEAR map (composition NOT free) generalizes well above
    # chance and roughly matches linear — i.e. the net LEARNED compositional structure, not arithmetic.
    real = (non >= 5 * chance) and (non >= 0.6 * max(lin, 1e-9))
    verdict = ("REAL — the NONLINEAR grounding, which does NOT get composition for free, generalizes to "
               f"UNSEEN combinations at {non:.0%} ({non/chance:.0f}x chance), matching linear. So zero-shot "
               "composition is genuine SYSTEMATIC generalization, not a linearity artifact."
               if real else
               "LINEARITY-DEPENDENT — the nonlinear map fails to generalize to unseen combos; the earlier "
               "result leaned on linear structure. Stated honestly.")
    print("  VERDICT:", verdict)
    print("\n" + "=" * 84)
    print("PASS — composition is genuine systematic generalization (survives nonlinear grounding)."
          if real else "RESULT: composition is linearity-dependent (honest).")
    print("=" * 84)
    sys.exit(0)   # honest measurement either way, not a regression gate


if __name__ == "__main__":
    main()
