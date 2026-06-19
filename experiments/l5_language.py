"""L5 — can the entity learn to COMMUNICATE about its world, and how QUICK? (laptop, numpy, seconds.)

Grounded symbols (recursivene/language.py): a symbol's meaning is a world-state the substrate perceives
from raw samples. We train ONLY on (scene, symbol) pairs and measure two things that actually reveal
intelligence:

  1. SAMPLES-TO-FIRST-COMMUNICATION — how few paired examples until it round-trips HELD-OUT concepts:
     names a freshly-perceived scene correctly (production) AND reconstructs a scene from its symbol
     (comprehension). "Learns to communicate QUICK", made literal.
  2. ZERO-SHOT COMPOSITION — give it symbols for primitives, then test NOVEL combinations it never saw
     paired. Naming/understanding unseen combos = systematic generalization, the hallmark of real
     intelligence (rote lookup cannot do this; chance = tiny).

Honest framing: grounding here is LINEAR over a fixed RFF perceiver, so composition falling out is a
real-but-specific result ("linear grounding composes"), not a claim of open-ended language. It is the
smallest honest seed of grounded communication — words pointing into a world model — and it makes the
"can it talk / how fast" question a measured experiment instead of a hope.

Run:  python experiments/l5_language.py     (exits 0 on PASS)
"""

import sys
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.language import RFF, GRID, perceive, reconstruct, LanguageGround

PRIMS = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]      # K hidden primitives; one symbol each
K = len(PRIMS)
D, GAMMA = 128, 12.0


def scene_fn(active):
    a = np.zeros(K);
    for k in active: a[k] = 1.0
    return lambda x: float(sum(a[k] * np.sin(PRIMS[k] * x) for k in range(K)))


def multihot(active):
    v = np.zeros(K)
    for k in active: v[k] = 1.0
    return v


def main():
    print("\nL5 — GROUNDED LANGUAGE: does the entity learn to NAME and IMAGINE its world, and how quick?\n")
    print(f"  K={K} primitive concepts {PRIMS}; symbols grounded on RFF latents (D={D}); laptop/numpy\n")
    rff = RFF(D=D, gamma=GAMMA, seed=0)
    phi = rff.phi_grid()
    lg = LanguageGround(V=K, D=D, ridge=1.0)

    # truth scenes on the grid (for comprehension MSE), per single primitive
    truth = {k: np.array([np.sin(PRIMS[k] * x) for x in GRID]) for k in range(K)}
    var = np.mean([t.var() for t in truth.values()])

    def heldout_production_acc():
        ok = 0
        for k in range(K):                               # FRESH perception (disjoint seeds) = held-out
            th = perceive(scene_fn([k]), rff, n=120, seed=10000 + k)
            ok += (lg.name(th, k=1) == {k})
        return ok / K

    def heldout_comprehension_relmse():
        es = []
        for k in range(K):
            rec = reconstruct(lg.imagine(multihot([k])), phi)
            es.append(np.mean((rec - truth[k]) ** 2))
        return float(np.mean(es) / var)

    # ---- 1. samples-to-first-communication: stream single-primitive pairs ----
    rng = np.random.default_rng(0)
    s2c, pairs = None, 0
    acc_curve = []
    for step in range(1, 61):
        k = int(rng.integers(K))
        th = perceive(scene_fn([k]), rff, n=120, seed=step)   # training perception
        lg.observe(th, multihot([k])); pairs += 1
        if pairs >= K:                                        # need >=1 per class before naming works
            lg.fit()
            acc = heldout_production_acc(); rel = heldout_comprehension_relmse()
            acc_curve.append(acc)
            if s2c is None and acc >= 0.9 and rel <= 0.10:
                s2c = pairs
                print(f"  FIRST COMMUNICATION at {pairs} pairs: production acc={acc:.0%}, comprehension relMSE={rel:.3f}")
        if s2c is not None and pairs >= s2c + 6:
            break
    final_acc = heldout_production_acc(); final_rel = heldout_comprehension_relmse()
    print(f"  after {pairs} pairs: production acc={final_acc:.0%}, comprehension relMSE={final_rel:.3f}  (chance acc={1/K:.0%})")

    # ---- 2. zero-shot composition: train on SOME 2-prim combos, test on UNSEEN ones ----
    all_pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    rng2 = np.random.default_rng(1); rng2.shuffle(all_pairs)
    train_combos = all_pairs[: len(all_pairs) // 2]
    heldout_combos = all_pairs[len(all_pairs) // 2:]
    for i, (a, b) in enumerate(train_combos):
        th = perceive(scene_fn([a, b]), rff, n=160, seed=20000 + i)
        lg.observe(th, multihot([a, b]))
    lg.fit()
    prod_ok, comp_es = 0, []
    for j, (a, b) in enumerate(heldout_combos):                # NOVEL combos, never seen paired
        th = perceive(scene_fn([a, b]), rff, n=160, seed=30000 + j)
        prod_ok += (lg.name(th, k=2) == {a, b})
        rec = reconstruct(lg.imagine(multihot([a, b])), phi)
        tru = np.array([np.sin(PRIMS[a] * x) + np.sin(PRIMS[b] * x) for x in GRID])
        comp_es.append(np.mean((rec - tru) ** 2) / tru.var())
    comp_acc = prod_ok / len(heldout_combos); comp_rel = float(np.mean(comp_es))
    chance2 = 1.0 / (K * (K - 1) / 2)
    print(f"  ZERO-SHOT COMPOSITION on {len(heldout_combos)} UNSEEN combos: name-both acc={comp_acc:.0%} "
          f"(chance={chance2:.0%}), imagine relMSE={comp_rel:.3f}")

    ok = (s2c is not None) and (final_acc >= 0.9) and (comp_acc >= 0.8) and (comp_rel <= 0.15)
    print("\n" + "=" * 84)
    print(f"PASS — grounded communication: first contact in {s2c} pairs, and it generalizes to UNSEEN symbol combinations."
          if ok else "PARTIAL — see numbers above.")
    print("=" * 84)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
