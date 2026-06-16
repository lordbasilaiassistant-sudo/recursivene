# RecursiveNe — L0 → L5 Roadmap

The seed grows **without retraining from scratch** because four interfaces are an abstraction
barrier: `predict`, `surprise`, `learning_progress`, `action_select`. Every higher layer changes
only the *internals* behind an interface, never its meaning — so each layer is a continuous
deformation of the one below, and the closed self-improvement loop (object/meta/meta-meta) keeps
running across the whole climb. (Synthesized from the foundations workflow; see `citations.md`.)

## L0 — SEED (built, runnable)
Online predictor (RFF+RLS) per activity; learning-progress curiosity over a mostly-noise world;
cost-for-competence objective; the full closed loop (config ratchet → self-edit harness →
self-edit proposer) with kill switch, invariant, sandbox, monotonicity log.
*Status: implemented. H1/H2/H3 + closure verified by `experiments/validate.py`.*

## L1 — LEARNED REPRESENTATION (first version BUILT, `recursivene/encoder.py`)
*Status: a first L1 exists and is validated.* `SpectralEncoder` discovers the data's own
frequencies from a rolling buffer of its experience and represents them at fixed cost (RFF
fallback for the not-yet-discovered). `l1_test.py` shows it FLATTENS the cost-to-complexity curve
that fixed features could not: as complexity climbed 14→29, fixed RFF cost-to-know exploded 28.5×
while L1 stayed flat (0.84×), ~103× cheaper at high complexity. This is the move from a fixed
feature map to a learned one — the smallest honest version of "predict in representation space."
*Next within L1:* the full JEPA form — a context encoder + EMA target + stop-gradient predicting
the target's *representation* so surprise is computed on predictable structure only (noisy-TV
rejected at the encoder), and routing it through the existing `learning_progress` interface.
*Draws on:* `knowledge/latent-world-models-jepa.md`.

## L2 — IMAGINATION (Dreamer)
A thin **differentiable latent dynamics head** `z' = f(z,a)` bolted beside the backprop-free
encoder, trained by short truncated rollouts. `action_select` upgrades from length-1 reactive
choice to scoring H-step **imagined** rollouts. One real step → many learning updates: the
sample-efficiency multiplier. Guards: short horizon, KL/free-bits, percentile-return + symlog.
*Validate:* competence reached in far fewer *real* interactions than L1. *Draws on:*
`knowledge/imagination-dreamer.md`.

## L3 — ACTIVE INFERENCE / EXPECTED FREE ENERGY
Action selection by **expected** free energy = pragmatic (reach preferred outcomes) + epistemic
(maximize learning progress). The L0 region-LP becomes a salience map over the learned latent
space; exploration is directed by expected information gain rather than reactive error.
*Validate:* directed data collection beats epsilon-greedy at equal budget. *Draws on:*
`knowledge/robust-learning-progress.md`.

## L4 — META / OPEN-ENDEDNESS
The improvement loop already edits the harness and proposer (meta-meta). L4 adds the **problem
generator**: at the mastered frontier, invent novel/previously-unsolvable/now-solvable/
non-forgetting problems (POWERPLAY/POET), and **mint abstractions** that expand representational
reach (library learning). Per-module learning rates/horizons become a mutable genome under the
same selection. *Validate:* repertoire growth + hardest-solved complexity climb while a fixed
benchmark saturates (`recursivene/bench/`, `recursivene/generator/`). *Draws on:*
`knowledge/open-endedness-powerplay.md`.

## L5 — SYMBOLS / LANGUAGE LAST
Language enters as **another observation channel** into the already-grounded world model — a
second encoder whose tokens must predict and be predicted by latent world-state, not a separate
autoregressive head. Symbols are discrete latents compressing recurrent structure the world model
already represents; preferences can then be specified symbolically. This is the explicit
inversion of LLM training order: physics first, words last, words grounded on physics.

## Why language must be last
A symbol is only meaningful as a compressed pointer into a model of what it refers to. Train
symbols first (LLMs) and they float free of any world — fluent, ungrounded, and forced to
re-derive physics from text statistics. Train the world model first and language becomes a cheap
compression layer over structure that is already there. The order is not a preference; it is what
makes the symbols *mean* anything.
