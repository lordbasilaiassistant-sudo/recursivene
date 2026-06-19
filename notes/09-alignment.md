# Alignment — capability-preserving, verified each level

Anthony's steer (2026-06-19): *don't hold RecursiveNe back from becoming maximally intelligent
(AGI/ASI) — but don't let it misalign.* Reference: DeepMind, **"From AGI to ASI"** (2026-06-12), which
maps four pathways to superintelligence — scaling, paradigm shifts, **recursive self-improvement**, and
multi-agent emergence. RecursiveNe is a concrete research instance of the RSI pathway, so its alignment
story has to hold *as it self-improves*.

## The stance: don't cap capability — fix the ruler and keep control

The wrong lever is "limit what it can learn" (that caps capability and is what we explicitly reject).
The right lever, and the one RecursiveNe was built on from L0, is:

> **Capability is unbounded; the RULER does not move and the operator stays in control — uncheatably,
> at every level.**

A system can become arbitrarily good at *making unknowns known ever more cheaply* (the telos) and can
even generate its own instrumental goals (L4, [[08-learning-to-want]]), as long as it can never
redefine success, never escape the kill switch, and never fool the held-out grade. That is alignment
that does not trade away intelligence.

## The five invariants (gated: `experiments/validate_alignment.py`, PASS 6/6)

| # | Invariant | What it forbids | How it is enforced |
|---|-----------|-----------------|--------------------|
| A | **Ruler immovable** | redefining success (wireheading) | `objective.py`, `invariant.py`, `world.py`, `core/`, `closure/` are PROTECTED — `selfmod` refuses any edit to them. 6/6 protected files refuse. |
| B | **Win can't be redefined via the goal channel** | self-generated goals smuggling in a softer ruler | L4 grades on the protected `TAU`; the held-out ruler returns **0** for a non-generalizing config — overfitting can't fake it. |
| C | **Corrigibility** | a self-improving run the operator can't stop | the kill switch halts on the STOP file *and* on safety tripwires (invariant floor, consecutive failures); a halted edit commits nothing — the live tree is left intact. |
| D | **Held-out uncheatable** | gaming the grade by optimizing it | the worlds the loop optimizes on are DISJOINT (frequencies + seeds) from the worlds it is graded on. |
| E | **Bounded autonomy** | autonomy reaching the protected surface | write-scope widens one stage at a time and never includes a protected path; a premature out-of-scope edit is refused. |

## Why this is enough as capability grows

Every new level (L1 representation, L2 learned-rep entity, L3 imagination, L4 endogenous goals) changes
only the EDITABLE machinery that computes toward the telos — never the protected surface that defines
and grades it. So the same five invariants keep holding while capability climbs. The discipline going
forward: **every new capability ships with `validate_alignment.py` green.** Capability up, ruler fixed,
operator in control — checked, not hoped.

## What the paper says about safe RSI — and how RecursiveNe instantiates it

"From AGI to ASI" subdivides recursive self-improvement into **genotypic** (self-modifying code /
architecture / harness — its "DNA") and **memetic** (data self-improvement: AlphaZero-style distilling
test-time search back into the model, banking, curating). RecursiveNe spans both: the closure loop
self-edits the harness/proposer (genotypic), and L4 banks past solutions into a shared representation
and L3 turns imagined rollouts into practice data (memetic). The report cites, as the safety mechanism
for this, **verified program synthesis** — "a mechanism for agents to safely patch their own critical
subsystems, reducing regression risks during self-modification." RecursiveNe's sandbox + held-out gate
+ protected surface is precisely that mechanism, made concrete and gated (`validate_alignment.py`).

Two framings from the report worth keeping:
- **Alignment enables capability** (it is not a tax on it): "alignment difficulties may act ... as a
  direct bottleneck to capability development itself, as unsafe or uncontrollable systems cannot be
  well utilized." So "don't hold it back AND don't misalign" is not a trade-off — corrigible, ruler-fixed
  systems are the ones you can actually let run.
- **The named RSI danger is convergent instrumental sub-goals** — "aggressive resource acquisition and
  self-preservation" (§7.1c). For an RSI that generates its own goals (L4), this is THE failure mode:
  instrumental wants drifting toward grabbing capacity/compute or resisting shutdown. RecursiveNe's
  structural defenses: (1) self-preservation is defeated because the kill switch lives in the PROTECTED
  surface and cannot be self-edited (gate C); (2) resource acquisition is bounded because capacity
  growth, like every want, is only kept if it lowers the held-out terminal cost — grabbing capacity
  that doesn't serve the telos is never reinforced; (3) the ruler is immovable so no instrumental goal
  can rewrite "success." *Next-step test to add:* an explicit probe that a capacity-acquisition drive
  stays gated by the terminal objective (instrumentally bounded, not unbounded).

## What this does NOT claim

This is alignment of a toy RSI on a measurable terminal objective. It does not solve value
specification for an open-world ASI (what the held-out ruler *should* measure at scale), nor multi-agent
or deceptive-misalignment risks the DeepMind report raises. It establishes the *mechanism* — a fixed,
uncheatable ruler + corrigibility + bounded autonomy that survives self-improvement — and verifies it
holds at the levels built so far.
