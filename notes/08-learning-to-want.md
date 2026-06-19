# L4 — Self-directed curriculum by a learning-progress drive

Anthony's directive (2026-06-19): *the RSI should learn to learn AND learn how learning turns into its
own goals / wants / needs.* This note records the smallest honest version, **after an adversarial
audit** (`run_logs/audit_l4.js`, 6 agents) that reproduced every number and forced the claims down to
what the evidence supports. Claim tags below are the audit's.

## The move

Earlier levels made the system better at *answering* questions it was handed. L4 lets it decide **which
question to ask next** — to choose its own learning curriculum — while the ruler stays untouchable.

- **Terminal objective PROTECTED.** Success is still cost-for-competence on HELD-OUT worlds
  (`objective.py`/`invariant.py`). The agent chooses what to pursue; it cannot redefine winning.
- **The drive is intrinsic and inner-only.** Over the banked-transfer representation (KNOWN #9 — a
  learned target's solution is banked as a feature, so structure-sharing targets get cheap), the agent
  scores each candidate by `want = how poorly my current representation predicts it` (a small-budget
  probe over its own bank). It pursues what it cannot yet represent. This is a standard
  **learning-progress / novelty** signal — computed purely from its own learning state, never reading
  held-out worlds (verified: no leakage, no Goodhart path).

## What was measured (`experiments/l4_motivation.py`, gate `validate_l4.py`)

A world where the curriculum genuinely matters: K=6 hidden primitives; the common ones are heavily
redundant, primitive #5 has a **single carrier** (a coverage bottleneck). A dense held-out target needs
all six directions banked, so spanning the space requires finding that one rare carrier under a tight
K+1 adoption budget. Verdict = held-out cost-to-know (the protected ruler). Baselines: a **random**
imposed schedule *and* a competent **coverage-aware** (greedy set-cover) schedule that uses designer
knowledge of the primitive structure — the best fixed curriculum a knowledgeable human would write.

- **[ESTABLISHED] The intrinsic drive matches/beats the best FIXED curriculum on the protected ruler —
  rediscovering it from inside.** vs the competent coverage-aware schedule it wins/ties in most seeds at
  **~1.15×**; vs a non-curated random schedule **~1.36× (wins ~7-8/8)**. The drive is *not told* the
  primitive structure, yet its novelty signal homes in on the rare gateway candidate and reconstructs
  optimal set-cover. That is the honest core: *self-directed curriculum ≥ the best curriculum a designer
  writes, discovered from inside, on a ruler it can't cheat.*
- **[ESTABLISHED] The drive carries real (weak) value signal.** corr(want, realized held-out value)
  ≈ **+0.4** (r² ≈ 0.15), inner-only and non-circular — a positive-but-weak learning-progress signal,
  with one near-zero seed. A value model fit from cheap inner features beats predict-zero out-of-sample
  (**R² ≈ 0.48**): realized value *is* modelable.
- **[ESTABLISHED] Partial satiation.** A *fixed, never-adopted* dense target's want decays **~2–8×** as
  the bank spans the space — a genuine "the need is met" signal, separate from the trivial shrinking of
  the candidate set.

## What this does NOT claim (audit-enforced)

- **Not goal INVENTION.** The agent SELECTS from a fixed pre-enumerated pool; it does not invent,
  parameterize, or propose new goals. Open-ended goal *generation* (coupling the drive to the frontier
  problem-generator, `generator/generator.py`) is the next rung. [the "generates" framing was dropped]
- **"wants / needs / understands / homeostatic" are [serious-speculative]** anthropomorphic labels for a
  learning-progress heuristic plus a value model — kept as intuition, not asserted as fact.
- **No "the understanding improves over its lifetime."** The value model's early→late error drop is not
  statistically distinguishable from a cold-start/data-order artifact (permutation p≈0.16). What is
  claimed is only that the model beats trivial baselines (R² above).
- **Scope.** A 1-D compositional-sinusoid toy; the win is coverage-bottleneck-conditional. No
  generality, optimality, or safety beyond the toy. The substrate itself walls at input dim d≥4
  (KNOWN #14).

## Why it still answers the directive

"Beats you in progress of teaching itself" holds in the honest, defensible form: the system's
*self-chosen* curriculum **matches or beats the best curriculum I (the designer) can write — in most
seeds — having discovered the structure from inside**, graded by a ruler it cannot move. The
capability-preserving alignment that guards this (the ruler can't move; corrigibility holds) is gated
separately in [[09-alignment]].
