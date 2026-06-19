"""COMPOUNDING — does the entity get cheaper at learning as it accumulates discovered laws? (the real
open-ended-intelligence test, not a one-shot novelty.)

Novelty would be "it can do X once." Real intelligence COMPOUNDS: each law it discovers becomes a reusable
abstraction that makes the NEXT unknown cheaper, so cost-to-know DECREASES over a lifetime. This streams
compositional targets (sums of hidden shared primitives); after each, the entity induces the target's law
(recursivene/induction.py, now pruned to clean parsimonious atoms), merges the discovered frequencies into
a banked primitive library, and we measure the held-out cost-to-know of a FRESH target over the current
bank. If it falls toward the floor (just fit the coefficients), learning compounds.

CONTROL (anti-novelty / no-free-lunch): an ORTHOGONAL stream where every target is a distinct, never-reused
frequency. There is no shared structure to bank, so cost-to-know MUST stay flat. If the "compounding" shows
up there too, it is leakage, not real — and the result is void.

Run:  python experiments/compounding.py     (exits 0 on PASS)
"""

import sys
import math
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.induction import induce

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


def run(mode, steps=10, seed=0):
    rng = np.random.default_rng(seed); bank = []; traj = []
    for t in range(steps):
        if mode == "structured":
            S = list(rng.choice(len(PRIMS), 2, replace=False)); c = rng.uniform(.5, 1, 2)
            train = lambda x, S=S, c=c: float(sum(ci * math.sin(PRIMS[k] * x) for ci, k in zip(c, S)))
            heldfn = (lambda: (lambda S, c: (lambda x: float(sum(ci * math.sin(PRIMS[k] * x) for ci, k in zip(c, S)))))(
                list(rng.choice(len(PRIMS), 2, replace=False)), rng.uniform(.5, 1, 2)))()
        else:  # orthogonal: a fresh distinct frequency each time, never reused -> nothing to bank
            wt = 4.0 + 1.3 * t + rng.uniform(0, 0.5)
            train = lambda x, wt=wt: math.sin(wt * x)
            heldfn = (lambda wt: lambda x: math.sin(wt * x))(40.0 + 2.0 * t + rng.uniform(0, 0.5))
        X = rng.uniform(-1, 1, 500); Y = np.array([train(x) for x in X]) + 0.02 * rng.standard_normal(500)
        _, terms = induce(X, Y, max_terms=6)
        freqs = [float(n[3:]) for n, _ in terms if n.startswith("sin")]
        bank_merge(bank, freqs)
        traj.append(cost_to_know(heldfn, bank, seed=100 + t))
    return traj, bank


def main():
    print("\nCOMPOUNDING — does the entity get cheaper at learning as it banks discovered laws? (real, capacity-controlled)\n")
    _, bank = run("structured", steps=12, seed=0)
    # CLEAN the bank by RECURRENCE: a true shared primitive appears in MANY targets; a spurious atom
    # appears once. Keep only recurring frequencies (count>=2) — discovery evidence, not noise.
    clean = [round(b, 2) for b, c in bank if c >= 2]
    print(f"  raw bank ({len(bank)}): {[round(b,2) for b,_ in bank]}")
    print(f"  CLEAN bank (recur>=2) ({len(clean)}) vs true {PRIMS}: {sorted(clean)}")

    # held-out structured targets, scored over: empty / RANDOM-freq bank of same size (CAPACITY CONTROL) / clean discovered bank
    rng = np.random.default_rng(7)
    rand = [[float(rng.uniform(1, 20)), 1] for _ in range(max(1, len(clean)))]
    cleanb = [[f, 2] for f in clean]
    def held(seed):
        r = np.random.default_rng(seed); S = list(r.choice(len(PRIMS), 2, replace=False)); c = r.uniform(.5, 1, 2)
        return lambda x: float(sum(ci * math.sin(PRIMS[k] * x) for ci, k in zip(c, S)))
    e, rd, cl = [], [], []
    for i in range(6):
        fn = held(500 + i)
        e.append(cost_to_know(fn, [], seed=200 + i))
        rd.append(cost_to_know(fn, rand, seed=200 + i))
        cl.append(cost_to_know(fn, cleanb, seed=200 + i))
    me, mr, mc = np.median(e), np.median(rd), np.median(cl)
    print(f"  cost-to-know (median, held-out structured): empty {me:.0f}  |  RANDOM-bank(same size) {mr:.0f} (capacity)  |  CLEAN discovered bank {mc:.0f}")
    print(f"  -> capacity effect: {me/max(mr,1):.1f}x;  REAL compounding beyond capacity (clean vs random): {mr/max(mc,1):.1f}x")

    # orthogonal control: no shared structure -> banking can't help (cost stays flat)
    ot, _ = run("orthogonal", steps=10, seed=1)
    cdrop = np.mean(ot[:3]) / max(np.mean(ot[-3:]), 1)
    print(f"  ORTHOGONAL no-shared-structure control: {np.mean(ot[:3]):.0f} -> {np.mean(ot[-3:]):.0f} ({cdrop:.1f}x) [must stay flat]")

    real_beyond_capacity = mc < 0.75 * mr          # discovered primitives beat equal-size random capacity
    bank_clean = len(clean) <= 2 * len(PRIMS)      # parsimonious (not capacity-padded)
    control_ok = cdrop <= 1.5
    ok = real_beyond_capacity and control_ok and mc < me
    print("\n" + "=" * 92)
    print("PASS — REAL compounding: the discovered laws beat an equal-size RANDOM bank (so it is structure,"
          " not capacity), and the no-shared-structure control stays flat. Cleaned by recurrence." if ok else
          "PARTIAL — compounding present but not cleanly beyond capacity; see numbers (honest).")
    print(f"  (clean bank size {len(clean)} vs true {len(PRIMS)} primitives; parsimony {'ok' if bank_clean else 'still fragmented'}.)")
    print("=" * 92)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
