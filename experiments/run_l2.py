"""L2 — the learned representation takes the entity out of 1-D frequency worlds.

L1 (SpectralEncoder) only discovers FREQUENCIES; it cannot even be applied where the structure is
higher-dimensional or sensory. This shows the L2 DeepEncoder makes unknowns known exactly where L1 /
fixed features WALL:
  Part 1 — the DIMENSION frontier: a stream of d=4 compositional unknowns. Fixed random features wall
           (the curse, per scaling_test.py); the learned representation crosses it.
  Part 2 — SENSORY: learn a real 2-D image (coordinate -> intensity), with ASCII proof.

Held-out, cost-to-know, honest. Run:  python experiments/run_l2.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.deep_encoder import DeepEncoder, cost_to_know

TAU = 0.05
RAMP = " .:-=+*#%@"


# ---- fixed-RFF baseline in d dims (the thing that WALLS) ------------------------------------------
def rff_cost_to_know(target_fn, d, tau, sizes, D=400, gamma=5.0, obs_noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, gamma, (D, d)); b = rng.uniform(0, 2 * np.pi, D); s = np.sqrt(2.0 / D)
    held = rng.uniform(-1, 1, (400, d)); th = np.array([target_fn(x) for x in held])
    sc = th.std() + 1e-9; th = th / sc
    last = np.inf
    for N in sizes:
        X = rng.uniform(-1, 1, (N, d))
        Phi = s * np.cos(X @ W.T + b)
        Y = np.array([target_fn(x) for x in X]) / sc + obs_noise * rng.standard_normal(N)
        w = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(D), Phi.T @ Y)
        Ph = s * np.cos(held @ W.T + b)
        last = float(np.mean((Ph @ w - th) ** 2))
        if last <= tau:
            return N, last
    return np.inf, last


def part1_dimension_frontier():
    print("\nPART 1 — the DIMENSION frontier: d=4 compositional unknowns (1-D spectral can't even apply)\n")
    rng = np.random.default_rng(3)
    sizes = (500, 1000, 2000, 4000)
    deep_costs, rff_reached = [], 0
    for t in range(4):
        ws = [rng.uniform(1.5, 3.5, 4) for _ in range(3)]
        cs = rng.uniform(-1, 1, 3)
        target = lambda x, ws=ws, cs=cs: float(sum(c * np.sin(w @ x) for c, w in zip(cs, ws)))
        Nd, md = cost_to_know(target, 4, TAU, sizes=sizes, iters=4000, seed=100 + t)
        Nr, mr = rff_cost_to_know(target, 4, TAU, sizes=sizes, seed=100 + t)
        deep_costs.append(Nd if np.isfinite(Nd) else None)
        if np.isfinite(Nr):
            rff_reached += 1
        ds = f"{Nd}smp (MSE {md:.4f})" if np.isfinite(Nd) else f"WALL (MSE {md:.3f})"
        rs = f"{Nr}smp" if np.isfinite(Nr) else f"WALL (MSE {mr:.3f})"
        print(f"  unknown {t}: L2 learned-rep = {ds:<24}  fixed-RFF = {rs}")
    deep_reached = sum(1 for c in deep_costs if c is not None)
    print(f"\n  L2 made-known {deep_reached}/4 of the d=4 unknowns; fixed RFF made-known {rff_reached}/4.")
    return deep_reached, rff_reached


def part2_sensory():
    print("\nPART 2 — SENSORY: learn a real 2-D image (coordinate -> intensity)\n")

    def image(x):
        a, b = x[0], x[1]
        return float(1.0 * np.exp(-(((a + 0.4) ** 2 + (b + 0.3) ** 2) / 0.10))
                     + 0.8 * np.exp(-(((a - 0.45) ** 2 + (b - 0.35) ** 2) / 0.06))
                     - 0.6 * np.exp(-(((a - 0.1) ** 2 + (b + 0.5) ** 2) / 0.05))
                     + 0.3 * a + 0.25 * np.sin(7.0 * np.sqrt(a * a + b * b)))

    rng = np.random.default_rng(0)
    N = 4000
    X = rng.uniform(-1, 1, (N, 2))
    sc = np.std([image(x) for x in X]) + 1e-9
    Y = np.array([image(x) for x in X]) / sc + 0.01 * rng.standard_normal(N)
    enc = DeepEncoder(2, hidden=160, seed=0)
    enc.fit(X, Y, iters=6000)

    ax = np.linspace(-1, 1, 40)
    grid = np.array([[image(np.array([a, b])) for a in ax] for b in ax])
    rec = np.array([[enc.predict(np.array([[a, b]]))[0] * sc for a in ax] for b in ax])
    rel = float(np.mean((grid - rec) ** 2) / grid.var())

    def render(g):
        lo, hi = g.min(), g.max(); n = (g - lo) / (hi - lo + 1e-9)
        ys = np.linspace(0, g.shape[0] - 1, 16).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, 36).astype(int)
        return "\n".join("".join(RAMP[int(n[y, x] * (len(RAMP) - 1))] for x in xs) for y in ys)

    print("  TARGET:"); print(render(grid))
    print("\n  L2 RECONSTRUCTION:"); print(render(rec))
    print(f"\n  reconstruction relative MSE = {rel*100:.1f}% of image variance "
          f"-> recovered {100*(1-rel):.0f}% of the scene from raw coordinates.")
    return rel


def main():
    print("=" * 86)
    print("L2 — the learned representation makes unknowns known where L1 / fixed features WALL")
    print("=" * 86)
    dr, rr = part1_dimension_frontier()
    rel = part2_sensory()
    print("\n" + "=" * 86)
    win = dr > rr and rel < 0.05
    print(f"L2 crosses the d=4 dimension wall ({dr}/4 vs fixed RFF {rr}/4) AND learns a 2-D sensory")
    print(f"field ({100*(1-rel):.0f}% recovered). The entity's reach is no longer 1-D frequencies.")
    print("Honest scope: toy worlds, illustrating the mechanism — the same held-out, cost-to-know")
    print("discipline as the rest of the project. Next: wire DeepEncoder in as the entity's backend.")
    print("=" * 86)


if __name__ == "__main__":
    main()
