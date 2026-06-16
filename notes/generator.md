# Open-Endedness: a frontier-following problem generator

**Module:** `recursivene/generator/` · **Owner:** open-endedness specialist
**Status:** built, runs against the live substrate, tests pass in 26s.

## The problem it solves

A fixed test saturates and then lies. The objective is cost-for-competence on a *fixed*
frequency ladder (`_INNER_W = 0, 1.5, 3, 6, 9, 12`). Once the learner aces that ladder,
`samples-to-competence` flatlines — every policy looks equally good against a ceiling it
already hit, and "smarter over time" stops being measurable (Goodhart). The cure is to keep
*moving* the test: when the repertoire is full, **invent a new activity at the learner's own
learnable frontier** and add it. That is open-endedness (Schmidhuber PowerPlay; Wang POET;
Lehman/Stanley novelty search).

## What it does

`propose_problems(learner_state, world)` reads the learner's per-activity competence and
proposes a new sine frequency that passes **four gates**, each with attached evidence so a
reviewer can audit the decision (no bare claims):

| Gate | Meaning | How it's tested |
|------|---------|-----------------|
| (a) **novel** | far from every existing frequency | `min |w − w_present| ≥ novelty_gap` |
| (b) **unsolved** | learner can't do it yet | fresh 1-arm probe at a *tiny* budget (60 samples) still has err > τ |
| (c) **now-solvable** | but it *becomes* solvable with more samples | same probe at a large budget (2500) reaches err ≤ τ, with a real downward `lp = err_small − err_large` |
| (d) **non-forgetting** | adding it leaves mastered arms mastered | re-score mastered arms in the world with the new arm appended; Δ must be ~0 |

Gate (c) is the key honest part. It is the **noisy-TV rejection lifted to the level of
problem generation**: a frequency too high for the model's capacity flatlines near the
`E[sin²]=0.5` floor regardless of samples, exactly like a noise arm sitting at its variance.
Proposing it would be proposing noise. The generator refuses it — the same way the LP policy
refuses the noisy TV inside a run.

Gate (d) is provably ~0 here because the substrate gives **each activity its own model**
(`learner.models[r]`); appending a higher arm never touches a mastered arm's weights or its
ground truth. We *measure* the delta instead of asserting it, so a future change that breaks
that isolation would be caught, not silently trusted.

`mint_abstraction(history)` compresses recurrent mastered structure into a reusable directed
RFF basis feature, or returns `None` when there is no dense recurrent band (≥3 mastered
frequencies within a band) — it never fabricates an abstraction.

## The empirical frontier (measured, not assumed)

Single-arm solvability for the default model (`gamma=8, D=96`), final grid-MSE vs frequency
and sample budget — this is the structure every gate threshold is grounded in:

```
 w  : err@60  err@250  err@2500
13.5: 0.065   0.010    0.000     <- frontier rung (unsolved tiny / solved big)
15.0: 0.080   0.012    0.001     <- frontier rung
16.5: 0.169   0.043    0.003     <- frontier rung
18.0: 0.208   0.052    0.009     <- frontier rung
19.5: 0.293   0.096    0.016     <- frontier rung (last solvable)
21.0: 0.464   0.313    0.198     <- capacity-MARGINAL: never reaches τ -> gate (c) rejects
24.0: 0.529   0.441    0.358     <- capacity-NOISE: flatlines -> gate (c) rejects
30.0: ~0.53   ~0.45    0.454     <- capacity-NOISE -> refused (test T2)
```

So the genuine learnable ladder for this model is **13.5 → 15 → 16.5 → 18 → 19.5**, then a
hard capacity wall at ~20. The generator climbs that ladder one rung at a time and then
**honestly reports the frontier exhausted** — it does not manufacture fake harder problems.

## The demonstration (real numbers)

`python recursivene/generator/run_generator.py` — 6 rounds, 34.4s, seed 0:

```
round 0: repertoire= 6 hardest_w= 12.0  fixed_cost=2.00e+07  ->  propose w=13.5
round 1: repertoire= 7 hardest_w= 13.5  fixed_cost=2.00e+07  ->  propose w=15.0
round 2: repertoire= 8 hardest_w= 15.0  fixed_cost=2.00e+07  ->  propose w=16.5
round 3: repertoire= 9 hardest_w= 16.5  fixed_cost=2.00e+07  ->  propose w=18.0
round 4: repertoire=10 hardest_w= 18.0  fixed_cost=2.00e+07  ->  propose w=19.5
round 5: repertoire=11 hardest_w= 19.5  fixed_cost=2.00e+07  ->  propose w=none (frontier exhausted)
```

**The contrast — the whole point of open-endedness:**

| metric | round 0 → 5 | behavior |
|--------|-------------|----------|
| FIXED benchmark cost (FLOPs) | `19,997,952` every round | **SATURATED** (flatline) |
| open-ended repertoire size | `6 → 7 → 8 → 9 → 10 → 11` | **CLIMBS** |
| open-ended hardest-solved freq | `12 → 13.5 → 15 → 16.5 → 18 → 19.5` | **CLIMBS** |

The fixed frequency-ladder test stopped measuring progress after the learner mastered it; the
open-ended repertoire-growth and hardest-solved-complexity metrics kept climbing. That gap is
exactly why a saturating test lies and why open-endedness is needed.

Minted abstraction this run: a directed basis feature at **w=13.5**, distilled from the
recurrent mastered band `[12.0, 13.5, 15.0]`. Standalone, grafting that single directed
feature into a fresh model at the band-center frequency cut samples-to-mastery from a mean of
~649 to ~430 (8/8 vs 7/8 seeds reaching τ) — a real but minimal win; aggressive
multi-feature grafting is fragile (it can remove random coverage the model still needed), so
the honest version grafts one directed feature and leaves D unchanged.

## What would falsify the claims

- If the proposed frequencies reached τ at the *tiny* budget, they'd be busywork not frontier
  work — gate (b) (`err_small > τ`) would have to fail. It doesn't (T3).
- If a capacity-noise frequency (w≥24, flatlines at ~0.45) were ever proposed, the noisy-TV
  rejection would be broken — gate (c) (`err_large ≤ τ`) would have to leak. It doesn't (T2:
  w=30 plateaus at 0.454 ≫ τ and is refused).
- If appending an arm changed a mastered arm's competence, non-forgetting would be false —
  gate (d) measures Δ and asserts <1e-9 (T4: 6 arms unchanged).
- If the repertoire failed to grow while the fixed test moved, the contrast would collapse —
  T1 asserts repertoire and complexity both strictly increase while the fixed cost is
  constant.

## Reproduction

```
cd <repo root>
python recursivene/generator/test_generator.py     # 5 assertions, ~26s, ALL PASS
python recursivene/generator/run_generator.py       # demo, ~34s, writes the report
```

Artifacts: `run_logs/generator_report.json` (per-round rows + proposal evidence +
saturation contrast + minted abstraction). Seed 0, numpy-only, CPU.

## Contract friction

None blocking. Two notes for the record:
- `make_world(extra_w=...)` inserts the new sine **before** the noise block, so noise indices
  shift when an arm is added. Harmless here (we rebuild the world each round and re-read
  `noise_indices`), but worth knowing for anyone caching arm indices across `extra_w` changes.
- The model's capacity wall (~w20 at `gamma=8, D=96`) caps the demo at ~6 rounds. To climb
  further the *substrate* would need more features or higher bandwidth — which is the
  open-ended loop's correct signal: "to invent harder problems, the learner must first get a
  bigger model." That hands the baton to the RSI harness, exactly as intended.

Cross-link: see `notes/PROGRESS.md` item "real continuous task generator (frontier-following),
not a fixed enum" — this module is that item.
