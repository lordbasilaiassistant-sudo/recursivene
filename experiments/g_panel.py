"""THE g-PANEL — a laptop scorecard of how smart the entity is, across orthogonal facets.

Pure numpy, CPU, runs in a couple of minutes. Each facet is a real measurement on HELD-OUT data with a
0-100 toy-scale rubric (NOT human-comparable g — explicitly a within-project tracker so you can watch it
climb as the substrate scales). Anti-woo: facets are held-out, and we print the honest caveat that the
absolute numbers are a relative tracker, not an IQ.

Facets:
  sample_efficiency        — samples to learn a held-out scene to tau (cheaper = smarter)
  transfer                 — does banking past solutions make new unknowns cheaper? (KNOWN #9)
  compositional_generaliz. — zero-shot naming of UNSEEN symbol combinations (L5)
  grounded_language        — samples-to-first-communication + production accuracy (L5)
  self_improvement         — learned self-edit search vs human hand-tuning (from the last race run)
  generalization_gap       — held-out vs train error (honesty axis: low gap = not overfitting)
  dimension_reach          — highest input dim it still learns (the curse-of-dim wall, KNOWN #14)

Run:  python experiments/g_panel.py
"""

import json
import math
import os
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.language import RFF, GRID, perceive, reconstruct, LanguageGround

TAU = 0.05


def _bar(score, w=24):
    n = int(round(score / 100 * w)); return "#" * n + "." * (w - n)


def _rff(D=128, gamma=12.0, seed=0):
    return RFF(D=D, gamma=gamma, seed=seed)


# ---------- facet probes (each returns dict {raw, score, detail}) -------------------------
def f_sample_efficiency():
    """Samples to fit held-out random band-limited scenes to TAU with a fresh RFF learner."""
    rff = _rff(seed=1); rng = np.random.default_rng(7)
    xe = GRID[::4]; costs = []
    for t in range(4):
        ws = rng.uniform(2, 12, 2); ph = rng.uniform(0, 6.28, 2)
        fn = lambda x, ws=ws, ph=ph: float(np.sum(np.sin(ws * x + ph)))
        truth = np.array([fn(x) for x in xe])
        P = np.eye(rff.D); w = np.zeros(rff.D); cost = 2000
        for n in range(1, 2001):
            x = rng.uniform(-1, 1); f = rff(x); y = fn(x) + 0.02 * rng.standard_normal()
            Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
            if n % 20 == 0:
                pred = np.array([w @ rff(z) for z in xe])
                if np.mean((truth - pred) ** 2) <= TAU: cost = n; break
        costs.append(cost)
    cost = float(np.mean(costs))
    score = float(np.clip(100 * (1 - math.log10(max(cost, 10)) / math.log10(2000)), 0, 100))
    return {"raw": cost, "score": score, "detail": f"~{cost:.0f} samples to tau (held-out scenes)"}


def f_transfer():
    """Banking past primitive solutions -> are new compositional unknowns cheaper? drop ratio."""
    rff = _rff(seed=2); prims = [4., 7., 10., 13.]; rng = np.random.default_rng(3)
    xe = GRID[::4]
    def learn(fn, bank):
        def feat(x):
            bf = np.array([np.interp(x, GRID, g) for g in bank]) if bank else np.zeros(0)
            return np.concatenate([bf, rff(x)])
        d = len(bank) + rff.D; P = np.eye(d); w = np.zeros(d)
        truth = np.array([fn(x) for x in xe]); cost = 2000
        for n in range(1, 2001):
            x = rng.uniform(-1, 1); f = feat(x); y = fn(x) + 0.02 * rng.standard_normal()
            Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
            if n % 20 == 0 and np.mean((truth - np.array([w @ feat(z) for z in xe])) ** 2) <= TAU:
                cost = n; break
        return cost, np.array([w @ feat(z) for z in GRID])
    bank = []
    for p in prims:                       # bank the primitives
        _, g = learn(lambda x, p=p: math.sin(p * x), []); bank.append(g)
    naive, banked = [], []
    for _ in range(4):
        c = rng.uniform(.5, 1, 4) * rng.choice([-1, 1], 4)
        fn = lambda x, c=c: float(np.sum(c * np.sin(np.array(prims) * x)))
        naive.append(learn(fn, [])[0]); banked.append(learn(fn, bank)[0])
    ratio = float(np.mean(naive) / max(np.mean(banked), 1))
    score = float(np.clip(100 * (math.log2(max(ratio, 1)) / math.log2(8)), 0, 100))
    return {"raw": ratio, "score": score, "detail": f"new unknowns {ratio:.1f}x cheaper with banked structure"}


def _l5(samples_cap=60):
    PR = [3., 5., 7., 9., 11., 13.]; K = len(PR); rff = _rff(seed=0); phi = rff.phi_grid()
    lg = LanguageGround(V=K, D=rff.D, ridge=1.0)
    sf = lambda S: (lambda x: float(sum(math.sin(PR[k] * x) for k in S)))
    mh = lambda S: np.array([1. if k in S else 0. for k in range(K)])
    truth = {k: np.array([math.sin(PR[k] * x) for x in GRID]) for k in range(K)}; var = np.mean([t.var() for t in truth.values()])
    def acc():
        return np.mean([lg.name(perceive(sf([k]), rff, n=120, seed=10000 + k), 1) == {k} for k in range(K)])
    rng = np.random.default_rng(0); s2c = None; pairs = 0
    for step in range(1, samples_cap + 1):
        k = int(rng.integers(K)); lg.observe(perceive(sf([k]), rff, n=120, seed=step), mh([k])); pairs += 1
        if pairs >= K:
            lg.fit()
            if s2c is None and acc() >= 0.9: s2c = pairs
        if s2c and pairs >= s2c + 4: break
    prod = float(acc())
    # composition
    allp = [(a, b) for a in range(K) for b in range(a + 1, K)]; r = np.random.default_rng(1); r.shuffle(allp)
    tr, ho = allp[:len(allp)//2], allp[len(allp)//2:]
    for i, (a, b) in enumerate(tr): lg.observe(perceive(sf([a, b]), rff, n=160, seed=20000+i), mh([a, b]))
    lg.fit()
    comp = np.mean([lg.name(perceive(sf([a, b]), rff, n=160, seed=30000+j), 2) == {a, b} for j, (a, b) in enumerate(ho)])
    return s2c, prod, float(comp), K


def f_grounded_language(cache):
    s2c, prod, comp, K = cache
    raw = s2c if s2c else 999
    score = float(np.clip(100 * (1 - (raw - K) / 50) * prod, 0, 100)) if s2c else 0.0
    return {"raw": raw, "score": score, "detail": f"first communication in {raw} pairs, production {prod:.0%}"}


def f_composition(cache):
    _, _, comp, K = cache
    chance = 1.0 / (K * (K - 1) / 2)
    score = float(np.clip(100 * (comp - chance) / (1 - chance), 0, 100))
    return {"raw": comp, "score": score, "detail": f"zero-shot name UNSEEN combos {comp:.0%} (chance {chance:.0%})"}


def f_self_improvement():
    p = os.path.join(REPO_ROOT, "run_logs", "race_6knob.json")
    if not os.path.exists(p):
        return {"raw": None, "score": 0.0, "detail": "no race result on disk"}
    d = json.load(open(p)); c = d["claude"]["best_val"]; s = d["rsi"]["best"]
    ratio = (c / s) if (c and s and math.isfinite(c) and math.isfinite(s)) else 1.0
    score = float(np.clip(100 * (math.log2(max(ratio, 1)) / math.log2(8)), 0, 100))
    return {"raw": ratio, "score": score, "detail": f"learned self-edit search {ratio:.1f}x better than my hand-tuning"}


def f_generalization_gap():
    rff = _rff(seed=5); rng = np.random.default_rng(9)
    fn = lambda x: math.sin(6 * x) + 0.5 * math.sin(11 * x)
    xs = [rng.uniform(-1, 1) for _ in range(120)]; P = np.eye(rff.D); w = np.zeros(rff.D)
    for x in xs:
        f = rff(x); y = fn(x) + 0.02 * rng.standard_normal(); Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
    tr = np.mean([(fn(x) - w @ rff(x)) ** 2 for x in xs])
    ho = np.mean([(fn(x) - w @ rff(x)) ** 2 for x in GRID])     # held-out grid
    gap = float(ho / max(tr, 1e-9))
    score = float(np.clip(100 * (1 - abs(math.log10(max(gap, 1e-3))) / 2), 0, 100))
    return {"raw": gap, "score": score, "detail": f"held-out/train error ratio {gap:.2f} (1.0 = no overfit)"}


def f_dimension_reach():
    """Highest input dim d where fixed RFF still fits a band-limited target to tau (the curse wall)."""
    rng = np.random.default_rng(11); reach = 0
    for d in (1, 2, 3, 4):
        r = np.random.default_rng(100 + d); W = r.normal(0, 8, (256, d)); b = r.uniform(0, 6.28, 256); s = math.sqrt(2 / 256)
        wt = r.normal(0, 1, d)
        fn = lambda x: math.sin(float(wt @ x))
        phi = lambda x: s * np.cos(W @ x + b)
        P = np.eye(256); w = np.zeros(256); xe = [rng.uniform(-1, 1, d) for _ in range(80)]
        truth = np.array([fn(x) for x in xe]); ok = False
        for n in range(1, 4001):
            x = rng.uniform(-1, 1, d); f = phi(x); y = fn(x) + 0.02 * rng.standard_normal()
            Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
            if n % 50 == 0 and np.mean((truth - np.array([w @ phi(z) for z in xe])) ** 2) <= TAU: ok = True; break
        if ok: reach = d
        else: break
    score = float(np.clip(100 * (reach / 4), 0, 100))
    return {"raw": reach, "score": score, "detail": f"fixed features reach d={reach} (walls beyond; learned reps cross it, KNOWN #15)"}


def main():
    print("\n" + "=" * 78)
    print("  RecursiveNe g-PANEL — how smart is the entity? (laptop, numpy; toy-scale tracker, not IQ)")
    print("=" * 78)
    l5cache = _l5()
    facets = [
        ("sample_efficiency", f_sample_efficiency()),
        ("transfer/abstraction", f_transfer()),
        ("grounded_language", f_grounded_language(l5cache)),
        ("compositional_generalization", f_composition(l5cache)),
        ("self_improvement", f_self_improvement()),
        ("generalization_gap", f_generalization_gap()),
        ("dimension_reach", f_dimension_reach()),
    ]
    print()
    for name, r in facets:
        print(f"  {name:30s} [{_bar(r['score'])}] {r['score']:5.1f}  {r['detail']}")
    g = float(np.mean([r["score"] for _, r in facets]))
    print("\n  " + "-" * 74)
    print(f"  g-score (mean of facets, toy-scale): {g:.1f} / 100")
    print("  honest: this is a WITHIN-PROJECT tracker to watch the entity climb as the substrate scales,")
    print("  NOT a human-comparable IQ. On breadth of real tasks the entity is still at the floor.")
    print("=" * 78)
    out = os.path.join(REPO_ROOT, "run_logs", "g_panel.json")
    json.dump({"facets": {n: r for n, r in facets}, "g_score": g}, open(out, "w"), indent=2)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
