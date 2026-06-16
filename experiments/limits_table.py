"""LIMITS TABLE — a runnable artifact for notes/05-limits-of-knowing.md.

Prints the taxonomy classification of every fundamental limit on computation / intelligence
as a compact table:  limit | category (#1/#2/#3/#4) | WALL or LOOPHOLE | the escape/grow-around move.

This is a CLASSIFICATION table, not a numerical experiment — but it is a runnable artifact so the
map in the notes is reproducible and grep-able, not just prose. The four categories (the meta-principle
for how an apparent law/limit breaks):

  #1 HIDDEN TERM      — bound is on a conserved resource; an unaccounted reservoir/charge pays it down.
                        Move: find the missing term.  (-> LOOPHOLE)
  #2 ASSUMED SYMMETRY — bound follows structurally from a symmetry/geometry reality genuinely has,
                        or from an unexplained boundary condition. NOT escapable by looking harder.
                        Move: grow AROUND it (more energy / stronger axioms / oracles / graded trust),
                        or EXPLAIN the boundary condition.  (-> REAL WALL)
  #3 WRONG DOMAIN     — true in its regime, but reality occupies a more permissive regime.
                        Move: identify the correct domain variable.  (-> LOOPHOLE / escapable)
  #4 PROXY CONFUSION  — exact for a derived/measured quantity that is NOT the protected invariant.
                        Move: re-measure the right invariant.  (-> LOOPHOLE)

Run:  python experiments/limits_table.py    (cwd = repo root)
"""

import numpy as np

from _util import bar  # noqa: F401  (path setup + repo import side effect; keeps the artifact uniform)

# (limit, category, wall_or_loophole, the_move)
# Each row mirrors notes/05-limits-of-knowing.md. Category strings carry the #1/#2/#3/#4 tag and a
# terse reason; wall_or_loophole is the one-word verdict; move is the escape / grow-around action.
ROWS = [
    ("Landauer bound (kT ln2 / erased bit)",
     "#4 then #1",
     "LOOPHOLE",
     "Re-measure: cost is on ERASED info, not op-count (reversible=0); erase into a non-thermal "
     "(spin/charge) bath -> energy->0 via that bath's chemical potential."),

    ("Margolus-Levitin / Bremermann speed (ops/s <= 2E/(pi hbar))",
     "#2 structural",
     "REAL WALL",
     "None. Fubini-Study geometry + unitarity fixes the orthogonalization rate; the measured rate "
     "IS the invariant. Only 'escape' is spending more E = paying the bound. Grow around: more energy."),

    ("Bekenstein / holographic density (S <= A/4)",
     "#2 structural",
     "REAL WALL",
     "None. DOF scale with AREA, not volume; gravity already in the ledger. The lacked structure is "
     "volume-extensivity of information. True UV info-capacity wall."),

    ("Carnot efficiency (eta <= 1 - Tc/Th)",
     "#4 proxy",
     "LOOPHOLE",
     "Re-measure: protected quantity is free energy/exergy, not two-temperature heat. Non-thermal "
     "resources (mu-gradients, coherence, squeezed/athermal baths) exceed the naive ceiling."),

    ("de Sitter horizon op-cap (S_dS = 3pi/(Lambda G); ~1e120 ops/patch)",
     "#2 boundary-cond",
     "REAL WALL*",
     "None within physics; GIVEN Lambda>0 the patch entropy is finite and caps ops/bits. ASTERISK: "
     "the VALUE of Lambda is itself an unexplained boundary condition (inherits the uncracked #2)."),

    ("No-Free-Lunch (uniform-prior average; all learners tie)",
     "#3 wrong-domain",
     "LOOPHOLE",
     "Identify domain: NFL assumes a UNIFORM prior over all targets, a symmetry reality lacks. Real "
     "targets are low-complexity/compressible -> we live OFF the NFL average; it never bites here."),

    ("Curse of dimensionality (need ~ exp(d_ambient))",
     "#3 wrong-domain",
     "LOOPHOLE",
     "Identify domain: real data lie on low-dim manifolds; right invariant is INTRINSIC dim, not "
     "ambient d. Learned deep representation crosses it (RecursiveNe deep_test.py)."),

    ("Sample-complexity / PAC (m >~ (VCdim+ln(1/delta))/eps)",
     "#2 then #4",
     "WALL+LOOP",
     "Real distribution-free wall ON CAPACITY (#2). But VC-dim is the wrong invariant for "
     "generalization (#4): use data-dependent complexity (margin, PAC-Bayes, true Rademacher); "
     "overparam nets violate only the naive proxy reading."),

    ("Godel incompleteness / halting undecidability",
     "#2 self-reference",
     "REAL WALL",
     "None. Structural to any consistent, expressive, r.e.-axiomatized system. Oracles only relocate "
     "it up the arithmetical hierarchy. Archetype: grow around (stronger axioms/oracles), never escape."),

    ("Loebian obstacle to self-trust (no safe []P -> P)",
     "#2 self-reference",
     "REAL WALL",
     "None fully. Reflection over weaker subsystems / logical induction / bounded tiling agents grant "
     "GRADED self-trust only. RecursiveNe certifies successors empirically (held-out), not by proof."),

    ("Cosmological constant problem (Lambda ~ 1e-120 of QFT estimate)",
     "#2 boundary-cond",
     "REAL WALL (open)",
     "UNCRACKED. Naive QFT assumes vacuum energy gravitates additively (structure reality lacks). "
     "Only anthropic/landscape moves EXPLAIN the boundary condition; no hidden term known."),

    ("Low-initial-entropy / Past Hypothesis",
     "#2 boundary-cond",
     "REAL WALL (open)",
     "UNCRACKED, deepest. Not dynamics, not domain-shiftable: a genuine initial condition to be "
     "EXPLAINED. Every act of computation/learning is a withdrawal from this one primordial account."),
]


def _truncate(s, w):
    return s if len(s) <= w else s[: w - 1] + "…"


def main():
    # column widths
    W_LIMIT, W_CAT, W_VERDICT = 56, 18, 16
    W_MOVE = 96

    n_wall = sum(1 for _, _, v, _ in ROWS if "WALL" in v and "+" not in v)
    n_loop = sum(1 for _, _, v, _ in ROWS if v == "LOOPHOLE")
    n_open = sum(1 for _, _, v, _ in ROWS if "open" in v)
    # category tallies via numpy (so the artifact actually uses numpy, not just imports it)
    cats = np.array([r[1].split()[0] for r in ROWS])
    uniq, counts = np.unique(cats, return_counts=True)

    print("=" * 188)
    print("THE LIMITS OF KNOWING — taxonomy of every fundamental limit on computation / intelligence")
    print("Categories: #1 hidden term | #2 assumed symmetry/structure (REAL WALL) | "
          "#3 wrong domain | #4 proxy confusion")
    print("=" * 188)
    header = (f"{'LIMIT':<{W_LIMIT}}  {'CATEGORY':<{W_CAT}}  "
              f"{'WALL/LOOPHOLE':<{W_VERDICT}}  {'THE MOVE (escape / grow-around)':<{W_MOVE}}")
    print(header)
    print("-" * len(header))
    for limit, cat, verdict, move in ROWS:
        print(f"{_truncate(limit, W_LIMIT):<{W_LIMIT}}  {cat:<{W_CAT}}  "
              f"{verdict:<{W_VERDICT}}  {_truncate(move, W_MOVE):<{W_MOVE}}")
    print("-" * len(header))

    print(f"\nTally: {len(ROWS)} limits | REAL WALLs (pure #2): {n_wall} | "
          f"clean LOOPHOLEs: {n_loop} | wall+loophole hybrids: "
          f"{sum(1 for _, _, v, _ in ROWS if '+' in v)} | UNCRACKED-open #2: {n_open}")
    print("Category distribution: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, counts)))

    print("\nMAPPING — RecursiveNe's OWN demonstrated walls:")
    print("  * cost-to-complexity blocker  -> #3 wrong-domain (fixed random features); grown around by "
          "linear capacity growth        [experiments/scaling_test.py]")
    print("  * curse of dimensionality     -> #3 wrong-domain; CROSSED by a learned representation     "
          "                               [experiments/deep_test.py]")
    print("  * Landauer energy floor       -> #1 hidden term = MEMORY (reversibility relocates heat->"
          "memory); + non-thermal bath    [experiments/landauer_test.py]")
    print("  * capacity exhaustion (garden)-> #2 structural; GROWN AROUND (seed grows its own capacity) "
          "                               [experiments/run_garden.py]")

    print("\nUNCRACKED #2 (boundary-condition subspecies; demand an EXPLANATION, not a hidden term):")
    print("  quantum gravity below the holographic bound | cosmological constant (1e-120) | "
          "low-initial-entropy / Past Hypothesis | the Loebian self-trust obstacle")
    print("=" * 188)


if __name__ == "__main__":
    main()
