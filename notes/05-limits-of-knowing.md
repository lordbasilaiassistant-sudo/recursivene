# 05 — The limits of knowing: a map of every wall, and which ones are real

Runnable companion: `experiments/limits_table.py` (prints the same classification as a table — run
`py experiments/limits_table.py` from the repo root). This note is the prose map; the script is the
grep-able artifact so the taxonomy is reproducible, not asserted.

Every claim is tagged **[established]** (textbook / published result), **[serious-speculative]**
(a defensible reading I'd defend but that isn't a settled theorem), or **[open]** (genuinely unsolved).

---

## 0. The taxonomy — the four ways an apparent law/limit breaks

A meta-principle that holds across physics and CS: when something looks like a hard limit, ask what
its *derivation secretly assumes*. There are exactly four failure modes. **[serious-speculative]** as
a unifying frame; each individual mechanism below is **[established]**.

| # | Failure mode | Diagnostic | The move | Verdict |
|---|---|---|---|---|
| **#1** | **HIDDEN TERM in the ledger** | the bound is on a *conserved resource* (free energy, information) and an unaccounted reservoir/charge pays it down | find the missing term | LOOPHOLE |
| **#2** | **ASSUMED SYMMETRY/STRUCTURE the system genuinely lacks** | the bound follows *structurally* from a symmetry/geometry reality actually has, **or** from an unexplained boundary condition | NOT escapable by looking harder — grow AROUND it (more energy / stronger axioms / oracles / graded trust) or EXPLAIN the boundary condition | **REAL WALL** |
| **#3** | **WRONG DOMAIN extrapolated** | true in its regime, but reality occupies a *more permissive* regime | identify the correct domain variable | LOOPHOLE |
| **#4** | **PROXY CONFUSION** | exact for a derived/measured quantity that is **not** the protected invariant | re-measure the right invariant | LOOPHOLE |

**Classification rule.** #1: a reservoir/charge closes the ledger → find it. #4: the limit is exact
for a measured quantity that isn't the protected invariant → remeasure (op-count vs erased-bits;
VC-capacity vs distribution-true generalization; Carnot's two-T heat vs exergy). #3: true in its
regime, reality is more permissive → name the right domain variable (ambient vs intrinsic dimension;
uniform-prior NFL vs structured targets). #2: structural from a symmetry the system has
(unitarity+Hilbert geometry; area-law/holography; self-reference) **or** from an unexplained boundary
condition (the value of Λ, the Past Hypothesis). #2 walls are only **grown around**, never escaped.

**Diagnostic for a *true open wall*:** a limit is a genuine open #2 iff (a) no reservoir closes the
ledger, (b) the measured quantity already *is* the invariant, and (c) reality does not sit in a more
permissive regime — leaving only a structural symmetry to grow around or a boundary condition to
explain. The hard cases (cosmological constant, low-initial-entropy, the Λ-inherited de Sitter cap)
are the **boundary-condition subspecies of #2**: they demand an *explanation* of the initial/parameter
condition, not a search for a hidden term.

---

## 1. The map — every fundamental limit on computation / intelligence

| Limit | Category | Wall or loophole | The escape / grow-around move | Tag |
|---|---|---|---|---|
| **Landauer bound** (`kT ln2` / erased bit) | **#4 then #1** | LOOPHOLE | (#4) The cost is on *erased* information, not op-count — reversible computing (Bennett/Toffoli/Fredkin) dissipates **zero** in the adiabatic limit, so the protected invariant is erased info, not ops. (#1) Even for erasure, the bound is on *free energy*: erase into a **non-thermal reservoir** (spin / angular-momentum / particle-number bath) and the energy cost → 0 as you raise that bath's chemical potential — the hidden term is the reservoir's *other conserved charges*. | [established] |
| **Margolus-Levitin / Bremermann speed** (`ops/s ≤ 2E/(π ℏ)`) | **#2 structural** | **REAL WALL** | None. Follows from the Fubini-Study geometry of Hilbert space + unitarity: the orthogonalization rate is fixed by the spread of the energy distribution. No hidden reservoir buys extra state-distinguishability per joule; the only "escape" is to spend more `E` — which is *paying* the bound, not evading it. Checked it is not secretly #4: the measured quantity (orthogonal-state rate) **is** the protected invariant. | [established] |
| **Bekenstein / holographic density** (`S ≤ A/4` in Planck units; `S ≤ 2πkRE/(ℏc)`) | **#2 structural** | **REAL WALL** (deepest info-capacity wall) | None. Gravity is *already in the ledger* — that's why the bound exists — so this is not #1. The structure reality lacks is **volume-extensivity of information**: assuming `S ~ volume` is the assumed structure the system genuinely lacks. DOF scale with **area**. A true UV constraint. | [established] |
| **Carnot / thermo efficiency** (`η ≤ 1 - Tc/Th`) | **#4 proxy** | LOOPHOLE | Exact for work-from-heat between two *fixed* temperatures, but the protected quantity is **free energy / exergy**. With non-thermal resources — chemical-potential gradients, quantum coherence, squeezed/athermal baths — the effective cold-reservoir term is replaced by a coherence-dressed quantity and the naive Carnot ceiling is exceeded. Carnot is a proxy *outside its defining regime*. | [established] (athermal-bath work extraction is experimentally demonstrated) |
| **de Sitter horizon computation cap** (`S_dS = 3π/(ΛG)`; ~10¹²⁰ total ops, ~10¹²² bits per causal patch) | **#2 boundary-condition** | **REAL WALL\*** | *Given* Λ>0, horizon entropy is finite and caps total accessible ops/bits in a causal patch — structural, not fixable by looking harder. **\*Asterisk:** the *value* of Λ is itself an unexplained boundary condition, so this real wall inherits the unsolved #2 problem one level down. | [established] (the cap) / [open] (the value of Λ) |
| **No-Free-Lunch** (Wolpert-Macready: uniformly averaged over all targets, all learners tie) | **#3 wrong-domain** (a real theorem misapplied) | LOOPHOLE | The theorem is real, but its precondition is a **uniform prior over all objective functions** — a symmetry the physical world genuinely lacks. Real targets are low-complexity / compressible (the universe is computable & smooth), so we live **off** the NFL average. All real ML operates in the permissive "structured targets" regime; NFL never bites there. | [established] |
| **Curse of dimensionality** (sample/volume need `~ exp(d_ambient)`) | **#3 wrong-domain** | LOOPHOLE | True for generic functions on the full `d`-cube, but real data lie on low-dim manifolds (manifold hypothesis) and targets are hierarchical/compositional. The right invariant is **intrinsic** dimension, not ambient `d`. Extrapolating ambient-`d` statistics to manifold-supported data is the 3D→2D-style domain error. A permissive regime exists. | [established] |
| **Sample-complexity / PAC** (`m ≳ (VCdim + ln(1/δ))/ε`) | **#2 worst-case / #4 in practice** | WALL + LOOPHOLE | VC/Rademacher worst-case capacity bounds are real distribution-free walls **on capacity** (#2). But VC-dim is the *wrong measured invariant* for generalization in the realizable / benign-overfitting regime (#4); the operative invariant is data-dependent complexity (margin, compression, PAC-Bayes, distribution-true Rademacher). Overparameterized nets "violate" only the naive proxy reading. | [established] |
| **Gödel incompleteness / halting undecidability** | **#2 self-reference** | **REAL WALL** (canonical) | None. Structural property of any consistent, sufficiently expressive, recursively-axiomatized system. No hidden resource, no domain shift removes it; oracles merely relocate it up the arithmetical hierarchy. The archetype of "grow around (stronger axioms/oracles), never escape." | [established] |
| **Löbian obstacle to self-trust** (no consistent system safely affirms its own `□P → P`) | **#2 self-reference** (downstream of Gödel) | **REAL WALL** | None fully. Partial escapes — reflection over weaker subsystems, logical-induction / probabilistic self-trust, bounded tiling agents — grant **graded** self-trust only: growing around, never the full forbidden reflection principle. *(RecursiveNe's response: certify successors empirically on a held-out frontier, not by proof — see `00-thesis.md` §6.)* | [established] (the obstacle) / [serious-speculative] (graded-trust workarounds) |
| **Cosmological constant problem** (observed Λ ~10⁻¹²⁰ of the QFT vacuum estimate) | **#2 boundary-condition** | **REAL WALL — UNCRACKED** | Naive QFT assumes vacuum energy gravitates additively (assumed structure reality lacks); no known mechanism, only anthropic/landscape moves that *explain* the boundary condition rather than find a hidden term. | [open] |
| **Low-initial-entropy / Past Hypothesis** (anomalously low early-universe entropy; source of the arrow of time and the entropy budget all learning spends) | **#2 boundary-condition** | **REAL WALL — UNCRACKED, deepest** | Not dynamics (no hidden term) and not domain-shiftable; a genuine initial condition that must be **explained**. Every act of computation/learning is a withdrawal from this one primordial low-entropy account. | [open] |

**Tally (from `limits_table.py`):** 12 limits — 7 pure-#2 real walls, 4 clean loopholes, 1
wall+loophole hybrid (PAC); category distribution #2=8, #3=2, #4=2; 2 walls flagged UNCRACKED-open.

---

## 2. RecursiveNe's OWN demonstrated walls, mapped onto the taxonomy

Each of these is a wall we actually *hit in code* and either crossed or located — not a thought
experiment. This is the payoff of the map: the same four categories that classify Bekenstein and
Gödel also classify our own experimental boundaries.

| RecursiveNe wall | Category | Verdict | What we did | Evidence |
|---|---|---|---|---|
| **Cost-to-complexity blocker** (fixed random features wall as target vocabulary grows) | **#3 wrong-domain** | LOOPHOLE, grown around | Fixed-capacity RFF *walls* as complexity `V` rises; with capacity grown ~linearly with `V`, cost-to-know stays bounded. The wall was an artifact of *fixed features*, the wrong regime. | `experiments/scaling_test.py` (sweep A: fixed-capacity walls, linear-capacity graceful) [established-in-substrate] |
| **Curse of dimensionality** (RFF needs `~exp(d)` to cover `d`-dim frequency space) | **#3 wrong-domain** | LOOPHOLE, **crossed** | Fixed random features never reach competence on a `d=4` `sin(w·x)` even at N=12000; a **learned** 1-hidden-layer representation crosses the exact same wall, reaching competence at N≈500–4000. The operative invariant is intrinsic structure, not ambient `d`. | `experiments/deep_test.py` (learned rep crosses where fixed RFF walls); `scaling_test.py` sweep B [established-in-substrate] |
| **Landauer energy floor** (irreversible learning dissipates `kT ln2` per erased bit) | **#1 hidden term = MEMORY** | LOOPHOLE (cost relocated, not abolished) | Two bookkeeping accounts of the *same* computation at *identical competence*: irreversible (forget data) pays `~1.9e-17 J`; reversible (keep the trace) pays `~0 J` in the quasi-static limit — the hidden term that closes the ledger is **memory** (store the whole stream). The deeper #1 move — erase into a non-thermal bath — drives even the erasure energy → 0. | `experiments/landauer_test.py`; write-up `notes/04-energy-of-knowing.md` [established] (illustrates Landauer 1961 / Bennett 1973-82 / Dyson 1979) |
| **Capacity exhaustion in the garden** (the seed's starting repertoire can't reach a target) | **#2 structural** | REAL WALL, **grown around** | A fixed repertoire *is* a structural ceiling — you cannot reach what you cannot represent (a finite-capacity sibling of the no-volume-extensivity wall). The seed grows its **own** capacity across seasons to reach what it couldn't, i.e. it grows *around* the wall rather than escaping the principle. | `experiments/run_garden.py` ("the seed grew its own capacity N time(s) to reach what it couldn't") [established-in-substrate] |

**Caveat [serious-speculative]:** "established-in-substrate" means demonstrated on our toy worlds
(RFF/MLP on sums-of-sines), at fixed seeds, illustrating the *mechanism*. It is **not** a claim that
the curse is crossed at arbitrary scale — only that the wall is mis-categorized as #2 when it is #3,
and a learned representation occupies the permissive regime. Reproduce: `py experiments/deep_test.py`,
`py experiments/scaling_test.py`, `py experiments/landauer_test.py`, `py experiments/run_garden.py`.

---

## 3. The UNCRACKED #2 problems — honestly open

These are the limits where the diagnostic returns *true open wall*: (a) no reservoir closes the
ledger, (b) the measured quantity already is the invariant, (c) reality does not sit in a more
permissive regime. What remains is either a structural symmetry to grow around or — for the hardest
cases — a **boundary condition to explain**. **All [open].** No hidden term is known for any of these;
do not present a workaround as a solution.

1. **Quantum gravity below the holographic bound.** Bekenstein/holography tells us DOF scale with
   *area*, but the microphysics that *realizes* `S = A/4` — what the horizon degrees of freedom
   actually *are* — is unsettled. The bound is established; the underlying theory is not. **[open]**

2. **Cosmological constant problem (Λ ~10⁻¹²⁰).** The single largest quantitative mismatch in physics.
   Naive QFT says the vacuum gravitates additively; reality says otherwise by 120 orders of magnitude.
   Anthropic/landscape arguments *explain the boundary condition* (a selection effect over a
   multiverse of Λ-values) but find no hidden term. Squarely a boundary-condition #2. **[open]**

3. **Low-initial-entropy / Past Hypothesis.** The deepest of the cluster: the early universe was in an
   *anomalously low-entropy* macrostate, and that single initial condition is the source of the arrow
   of time and the entire entropy budget that every act of computation and learning draws down
   (cf. `04-energy-of-knowing.md`: every irreversible bit is a withdrawal). It is not dynamics, not
   domain-shiftable, not closable by a reservoir — it is an initial condition that must be *explained*.
   **[open]**

4. **The Löbian self-trust obstacle.** A consistent, sufficiently expressive system cannot safely
   affirm its own reflection principle `□P → P` (Gödel II applied to self-trust). For an
   *self-improving* system this is the live constraint: a successor cannot be certified sound *by
   proof* from inside the system. The partial escapes (reflection over weaker subsystems, logical
   induction, bounded tiling agents) buy only **graded** self-trust. RecursiveNe's stance is to dodge
   the proof obstacle entirely and certify successors **empirically** on a held-out frontier
   (`00-thesis.md` §6) — a *grow-around*, explicitly not an escape. **[open]**

> **The Λ-inherited de Sitter asterisk.** The de Sitter op-cap (~10¹²⁰ ops/patch) is an *established*
> finite ceiling *given* Λ>0 — but because the value of Λ is itself problem #2 above, the de Sitter
> cap inherits an open boundary condition one level down. The ceiling is real; *why it sits where it
> does* is open. **[established] (cap) / [open] (value).**

---

## 4. The one-line summary

Most "fundamental limits on intelligence" are **#3 wrong-domain** or **#4 proxy** confusions and have
permissive regimes we already live in (NFL, curse of dimensionality, Carnot, the op-count reading of
Landauer) — and RecursiveNe has *crossed two of them in code*. A smaller set are **#2 real walls** you
can only grow around (Margolus-Levitin, Bekenstein, Gödel, Löb). And a residue — the cosmological
constant, the low-initial-entropy Past Hypothesis, and the Λ-value the de Sitter cap inherits — are
**genuinely uncracked boundary-condition #2 problems** that demand an *explanation*, not a hidden term.
That residue is the real floor of knowing; everything above it is engineering.

---

*Cross-links: `00-thesis.md` §6 (honest walls: Löb, NFL, Goodhart) · `04-energy-of-knowing.md`
(the Landauer rung, settled) · `experiments/limits_table.py` (runnable classification table) ·
`experiments/{scaling_test,deep_test,landauer_test,run_garden}.py` (the four demonstrated walls).
Reproduce the table: `py experiments/limits_table.py`.*
