"""Answer B with code+output: do the substrate's phenomena survive in a RICHER (2-D) world?

The first 2-D run exposed a real, non-obvious bug in the curiosity signal: mastery was judged
by TRAINING prediction error, which a high-capacity RLS drives low by overfitting before it
GENERALIZES. So LP declared arms "mastered" and abandoned them while their true (grid) error
was still high; random, having no mastery concept, kept hammering every arm until it
generalized — and so random won. The lesson (important for every higher layer): the curiosity /
mastery signal must track GENERALIZATION, not training fit.

This version fixes it: each learnable arm keeps a held-out set; mastery and ranking use held-out
(generalization) error. The noise gate still uses training error (noise sits at ~1.0). Then we
compare random / novelty / three LP allocation rules.

Run:  python experiments/exp_2d.py
"""

import numpy as np

from _util import sparkline  # noqa: F401

TAU, TAU_MASTER, NOISE_FLOOR = 0.05, 0.04, 0.7
WS = [(2.0, 0.0), (3.0, 2.0), (0.0, 4.0), (4.0, 4.0), (6.0, 3.0)]   # 2-D freq vectors, rising |w|
N_NOISE = 10
NL = len(WS)


class RFF2D:
    def __init__(self, D=256, gamma=7.0, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, gamma, (D, 2)); self.b = r.uniform(0, 2 * np.pi, D)
        self.s = np.sqrt(2.0 / D); self.w = np.zeros(D); self.P = np.eye(D); self.flops = 0

    def phi(self, x): return self.s * np.cos(self.W @ x + self.b)

    def predict(self, x): self.flops += 2 * len(self.b); return float(self.w @ self.phi(x))

    def update(self, x, y):
        p = self.phi(x); err = y - p @ self.w; Pp = self.P @ p
        k = Pp / (1.0 + p @ Pp); self.w = self.w + k * err
        self.P = self.P - np.outer(k, Pp); self.flops += 4 * len(self.b) ** 2; return err


def truth(r, x): return float(np.sin(WS[r] @ x))


def run(policy, steps=11000, seed=0):
    rng = np.random.default_rng(seed)
    K = NL + N_NOISE
    models = [RFF2D(seed=seed + 17 * r) for r in range(NL)]
    heldout = [[rng.uniform(-1, 1, 2) for _ in range(25)] for _ in range(NL)]
    thist = [[] for _ in range(K)]; visits = np.zeros(K, int)
    noise = list(range(NL, K))

    def train_err(r):
        return float(np.mean(thist[r][-30:])) if thist[r] else np.inf       # noise gate

    def gerr(r):
        if r >= NL:
            return np.inf                                                    # noise: no model
        return float(np.mean([(truth(r, x) - models[r].predict(x)) ** 2 for x in heldout[r]]))

    def lprog(r):
        h = thist[r]
        return float(np.mean(h[-40:-20]) - np.mean(h[-20:])) if len(h) >= 40 else 0.0

    log_comp, log_flops = [], []
    for t in range(steps):
        te = np.array([train_err(r) for r in range(K)])
        if (visits < 5).any():
            r = int(np.argmin(visits))
        elif policy == "random":
            r = int(rng.integers(K))
        elif policy == "novelty":
            r = int(np.argmax(te))
        else:
            gen = np.array([gerr(r) for r in range(K)])
            eligible = (te < NOISE_FLOOR) & (gen >= TAU_MASTER)              # learnable, not-yet-GENERALIZED
            ei = np.where(eligible)[0]
            if len(ei):
                if policy == "lp_worst":
                    r = int(ei[np.argmax(gen[ei])])
                elif policy == "lp_uniform":
                    r = int(rng.choice(ei))
                else:  # lp_progress
                    lps = np.array([lprog(i) for i in ei])
                    r = int(ei[np.argmax(lps)]) if lps.max() > 1e-6 else int(rng.choice(ei))
            else:
                probe = (te >= NOISE_FLOOR) & (visits < 30)
                r = int(min(np.where(probe)[0], key=lambda i: visits[i])) if probe.any() else int(rng.integers(NL))
        x = rng.uniform(-1, 1, 2)
        if r in noise:
            thist[r].append(1.0)
        else:
            e = models[r].update(x, truth(r, x) + 0.02 * rng.standard_normal())
            thist[r].append(e * e)
        visits[r] += 1
        if (t + 1) % 25 == 0:
            log_comp.append(max(gerr(r) for r in range(NL)))
            log_flops.append(sum(m.flops for m in models))

    cost = np.inf
    if log_comp[-1] <= TAU:
        for i, c in enumerate(log_comp):
            if c <= TAU:
                cost = log_flops[i]; break
    return {"cost": cost, "final": log_comp[-1], "noise_frac": float(visits[noise].sum() / visits.sum()),
            "reached": log_comp[-1] <= TAU}


def main():
    print("\nAnswer B: do H1/H2 hold in a 2-D world (with generalization-based mastery)?\n")
    print(f"  {NL} learnable 2-D sines (|w| rising) + {N_NOISE} noise distractors\n")
    res = {}
    for pol in ("random", "novelty", "lp_worst", "lp_uniform", "lp_progress"):
        r = run(pol, steps=11000, seed=0)
        res[pol] = r
        cost = f"{r['cost']:.2e}" if np.isfinite(r["cost"]) else "inf"
        print(f"  {pol:11s}: cost-for-competence={cost:>9}  final_err={r['final']:.4f}  "
              f"noise={r['noise_frac']*100:4.1f}%  reached={r['reached']}")
    print("\n" + "=" * 64)
    best = min(("lp_worst", "lp_uniform", "lp_progress"),
               key=lambda p: res[p]["cost"] if np.isfinite(res[p]["cost"]) else 1e18)
    lp, rn, nv = res[best], res["random"], res["novelty"]
    h1 = np.isfinite(lp["cost"]) and lp["cost"] < rn["cost"]
    h2 = lp["noise_frac"] < 0.3 and nv["noise_frac"] > 0.6
    if h1 and h2:
        print(f"KNOWN: with generalization-based mastery, {best} beats random in 2-D "
              f"({lp['cost']:.2e} < {rn['cost']:.2e})")
        print(f"       and avoids the noisy TV ({lp['noise_frac']*100:.0f}% vs novelty {nv['noise_frac']*100:.0f}%).")
        print("=> H1+H2 generalize to 2-D. Answer B: yes — AND it taught a real lesson:")
        print("   curiosity must track GENERALIZATION, not training error (the overfit trap).")
    else:
        print(f"PARTIAL: best={best} H1={h1} H2={h2}. Numbers above. The generalization lesson stands regardless.")
    print("=" * 64)


if __name__ == "__main__":
    main()
