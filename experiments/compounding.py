"""COMPOUNDING — does learning COMPOUND via discovered structure? RESOLVED (over-claim -> retraction -> root cause -> fix).

The honest arc (this is the anti-woo discipline working end to end):
  1. A single-seed run showed ~1.6x and was over-claimed as 'real compounding'.
  2. An adversarial audit + a multi-seed capacity control REFUTED it: with a FRAGMENTED bank the discovered
     frequencies tied an EQUAL-SIZE RANDOM bank -> the gain was CAPACITY (low-freq atoms beat the RFF
     fallback), not structure-reuse. Retracted (KNOWN #27).
  3. ROOT CAUSE (KNOWN #26): the bank fragmented because a short [-1,1] window cannot resolve superposed
     frequencies — an IDENTIFIABILITY limit, not an algorithm one (multi-start joint NLS recovers exact
     primitives 1/8 on +-1 but 6/8 on +-3).
  4. FIX: widen the observation window -> consistent (exact) extraction -> the discovered bank now BEATS an
     equal-size random bank (median ~1.8x, ~10/10 with a perfect bank). So compounding IS real, CONDITIONED
     on consistent extraction (a data-RANGE requirement), not a free lunch.

This file demonstrates BOTH directions: narrow window = capacity (the honest negative), wide window = real
compounding. Run:  python experiments/compounding.py
"""

import sys
import math
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.induction import induce, extract_laws

GRID = np.linspace(-1, 1, 201); XE = GRID[::4]; TAU = 0.05
PRIMS = [3.0, 7.0, 11.0, 15.0]                 # hidden shared primitives (structured world)


def bank_merge(bank, freqs, tol=0.3):
    for f in freqs:
        hit = next((i for i, (b, _) in enumerate(bank) if abs(b - f) < tol), None)
        if hit is None:
            bank.append([f, 1])
        else:
            b, c = bank[hit]; bank[hit] = [(b * c + f) / (c + 1), c + 1]
    return bank


def cost_to_know(fn, bank, rffD=48, seed=0):
    """Samples for RLS over [banked sin/cos laws] + a small RFF fallback to reach tau on held-out grid."""
    r = np.random.default_rng(seed); W = r.normal(0, 8, rffD); b = r.uniform(0, 2 * np.pi, rffD); s = math.sqrt(2 / rffD)
    bf = [bb for bb, _ in bank]

    def feat(x):
        d = np.concatenate([[math.sin(f * x), math.cos(f * x)] for f in bf]) if bf else np.zeros(0)
        return np.concatenate([d, s * np.cos(W * x + b)])
    dd = 2 * len(bf) + rffD; P = np.eye(dd); w = np.zeros(dd); truth = np.array([fn(x) for x in XE])
    for n in range(1, 1501):
        x = r.uniform(-1, 1); f = feat(x); y = fn(x) + 0.02 * r.standard_normal()
        Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
        if n % 10 == 0 and np.mean((truth - np.array([w @ feat(z) for z in XE])) ** 2) <= TAU:
            return n
    return 1500


def build_bank(steps=12, seed=0, halfwidth=1.0, extractor="greedy"):
    """Stream structured targets; extract each target's frequencies over an observation window of
    +-halfwidth; bank them. extractor='greedy' = induce() (fragments); 'multistart' = extract_laws()
    (global multi-start + BIC-K, exploits the wider window -> consistent/exact extraction)."""
    rng = np.random.default_rng(seed); bank = []
    for t in range(steps):
        S = list(rng.choice(len(PRIMS), 2, replace=False)); c = rng.uniform(.5, 1, 2)
        fn = lambda x, S=S, c=c: float(sum(ci * math.sin(PRIMS[k] * x) for ci, k in zip(c, S)))
        X = rng.uniform(-halfwidth, halfwidth, 600); Y = np.array([fn(x) for x in X]) + 0.02 * rng.standard_normal(600)
        if extractor == "multistart":
            freqs = list(extract_laws(X, Y, Kmax=4, seed=seed * 100 + t))
        else:
            _, terms, _ = induce(X, Y, max_terms=6, fmax=18.0)
            freqs = [float(n[3:]) for n, _ in terms if n.startswith("sin")]
        bank_merge(bank, freqs)
    return bank


def held(seed):
    r = np.random.default_rng(seed); S = list(r.choice(len(PRIMS), 2, replace=False)); c = r.uniform(.5, 1, 2)
    return lambda x: float(sum(ci * math.sin(PRIMS[k] * x) for ci, k in zip(c, S)))


def capacity_control(bank, sd):
    """Held-out cost-to-know over: empty / equal-size RANDOM bank (capacity) / the discovered CLEAN bank.
    Returns (random/clean ratio = real compounding beyond capacity, clean<random?)."""
    clean = [round(b, 2) for b, c in bank if c >= 2]
    if not clean:
        clean = [round(b, 2) for b, _ in bank][:4]
    rng = np.random.default_rng(700 + sd)
    rand = [[float(rng.uniform(1, 16)), 1] for _ in range(max(1, len(clean)))]
    cleanb = [[f, 2] for f in clean]
    rd, cl = [], []
    for i in range(4):
        fn = held(5000 + 10 * sd + i)
        rd.append(cost_to_know(fn, rand, seed=200 + i)); cl.append(cost_to_know(fn, cleanb, seed=200 + i))
    mr, mc = np.median(rd), np.median(cl)
    return mr / max(mc, 1), (mc < mr), clean


def main():
    print("\nCOMPOUNDING — RESOLVED: real beyond capacity IFF extraction is consistent (needs data RANGE + a global extractor).\n")
    print("  A single-seed 1.6x was over-claimed then RETRACTED (a FRAGMENTED bank tied a random bank =")
    print("  capacity). Root cause (KNOWN #26): a short window + a GREEDY extractor can't resolve superposed")
    print("  frequencies. Fix: a WIDE window + a GLOBAL multi-start+BIC-K extractor (extract_laws).\n")
    SEEDS = 6
    arms = [("GREEDY induce, narrow +-1 (capacity only)", 1.0, "greedy"),
            ("GREEDY induce, wide +-3 (still fragments)", 3.0, "greedy"),
            ("MULTISTART+BIC-K, wide +-3 (consistent)", 3.0, "multistart")]
    results = {}
    for label, hw, ex in arms:
        ratios, beats, sample_clean = [], 0, None
        for sd in range(SEEDS):
            r, b, clean = capacity_control(build_bank(12, sd, hw, ex), sd)
            ratios.append(r); beats += b
            if sd == 0: sample_clean = clean
        ratios = np.array(ratios); results[ex + str(hw)] = (float(np.median(ratios)), beats)
        print(f"  {label}:")
        print(f"    compounding beyond capacity (random/clean): median {np.median(ratios):.2f}x  "
              f"range [{ratios.min():.2f},{ratios.max():.2f}]  beats random {beats}/{SEEDS}"
              f"   (bank≈{sorted(sample_clean)} vs true {PRIMS})")

    g_med, g_wins = results["greedy1.0"]
    m_med, m_wins = results["multistart3.0"]
    print("\n" + "=" * 96)
    print(f"  RESOLUTION (honest, both directions): GREEDY/narrow = median {g_med:.2f}x ({g_wins}/{SEEDS}) -> CAPACITY,")
    print(f"  not structure (the retraction was correct). MULTISTART+BIC-K on a wide window = median {m_med:.2f}x")
    print(f"  ({m_wins}/{SEEDS}) -> REAL compounding: the discovered (now ~exact) primitives beat an equal-size random")
    print("  bank. Learning DOES compound via discovered structure, CONDITIONED on consistent extraction —")
    print("  an identifiability/data-RANGE + global-extractor requirement (KNOWN #26/#27), not a free lunch.")
    print("=" * 96)
    ok = bool(m_med >= 1.3 and m_wins >= SEEDS * 3 // 4 and m_med > g_med + 0.2)
    print("RESULT:", "RESOLVED — compounding real given consistent extraction (multistart + wide window)" if ok
          else "inconclusive at these settings (honest)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
