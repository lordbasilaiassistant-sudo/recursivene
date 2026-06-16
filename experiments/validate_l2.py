"""L2 gate: the learned representation crosses the dimension wall AND learns a sensory field.
Assertion-based PASS/FAIL with numbers. Run: python experiments/validate_l2.py
"""

import sys
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.deep_encoder import DeepEncoder, cost_to_know

TAU = 0.05
R = []


def check(name, ok, detail):
    R.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name:30s} {detail}")


def rff_reaches(target_fn, d, N, D=400, gamma=5.0, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, gamma, (D, d)); b = rng.uniform(0, 2 * np.pi, D); s = np.sqrt(2.0 / D)
    held = rng.uniform(-1, 1, (400, d)); th = np.array([target_fn(x) for x in held]); sc = th.std() + 1e-9
    X = rng.uniform(-1, 1, (N, d)); Phi = s * np.cos(X @ W.T + b)
    Y = np.array([target_fn(x) for x in X]) / sc + 0.02 * rng.standard_normal(N)
    w = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(D), Phi.T @ Y)
    return float(np.mean(((s * np.cos(held @ W.T + b)) @ w - th / sc) ** 2))


def main():
    rng = np.random.default_rng(7)
    # H_L2a: learned rep reaches competence on a d=4 target where fixed RFF (4000 samples) does not
    ws = [rng.uniform(1.5, 3.5, 4) for _ in range(3)]; cs = rng.uniform(-1, 1, 3)
    tgt = lambda x: float(sum(c * np.sin(w @ x) for c, w in zip(cs, ws)))
    Nd, md = cost_to_know(tgt, 4, TAU, sizes=(500, 1000, 2000), iters=4000, seed=1)
    rff_mse = rff_reaches(tgt, 4, 4000, seed=1)
    check("L2 crosses d=4 dimension wall", np.isfinite(Nd) and rff_mse > TAU,
          f"learned-rep reached at N={Nd} (MSE {md:.4f}); fixed RFF MSE {rff_mse:.3f} > tau (walled)")

    # H_L2b: learned rep recovers a 2-D sensory field
    def image(x):
        a, b = x[0], x[1]
        return float(np.exp(-(((a + 0.4) ** 2 + (b + 0.3) ** 2) / 0.10))
                     + 0.3 * a + 0.25 * np.sin(7.0 * np.sqrt(a * a + b * b)))
    N = 4000; X = rng.uniform(-1, 1, (N, 2)); sc = np.std([image(x) for x in X]) + 1e-9
    Y = np.array([image(x) for x in X]) / sc + 0.01 * rng.standard_normal(N)
    enc = DeepEncoder(2, hidden=160, seed=0); enc.fit(X, Y, iters=6000)
    ax = np.linspace(-1, 1, 36); grid = np.array([[image(np.array([a, b])) for a in ax] for b in ax])
    rec = np.array([[enc.predict(np.array([[a, b]]))[0] * sc for a in ax] for b in ax])
    rel = float(np.mean((grid - rec) ** 2) / grid.var())
    check("L2 learns a 2-D sensory field", rel < 0.05, f"reconstruction relative MSE {rel*100:.1f}% (< 5%)")

    ok = all(R)
    print(f"\n{'='*60}\n{'PASS — L2 verified: learned representation, richer worlds.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
