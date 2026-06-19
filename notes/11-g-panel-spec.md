# The g-panel — canonical facet spec (honesty-audited)

Synthesized by the design+anti-woo workflow (2026-06-19, 5 agents). The g-panel measures HOW SMART the
entity is across orthogonal facets, on a laptop, with each facet carrying a CONTROL that VOIDs it if it
can be gamed. The **facet VECTOR is the primary artifact** — a single g-score destroys the orthogonality
that is the whole point. If a scalar is demanded: **geometric mean over non-N/A facets** (dominated by
the weakest facet, so you can't buy g by over-optimizing one cheap axis — explicitly NOT arithmetic mean).

## Honesty rules (load-bearing)
- Score on a PROTECTED held-out set the optimizer never touches (`HELDOUT_SEEDS=(101,102,103)` /
  `_HELDOUT_W` with phase_seed+777 / disjoint combos & referents). Add no new editable surface.
- **MEDIAN over seeds, not mean** (cost-to-know is heavy-tailed — learner-build-lessons).
- Every facet ships a **control AS CODE** that MUST collapse to chance for the headline to count
  (lookup table, orthogonal arm, persistence model, fixed-RFF arm, shuffled symbols, frozen operator).
  A facet whose control does not collapse is **N/A — VOID**, shown visibly, excluded from g (never a win).
- **No fabrication**: facets score 0 when nothing is minted/established; never read a cached/editable
  JSON as a live score (the static `race_6knob.json` self-improvement read was the canonical violation —
  re-measure live each run); FLOPs-to-tau is first-class, not just samples.
- Label everything TOY-SCALE, CPU, pure-numpy, NOT human-comparable. Honest negatives are first-class
  (if out-of-family is low, the finding is "it's a sine-fitter at this scale").

## Canonical facets (each: probe · held-out · control)
- **F1 sample_efficiency** — samples AND FLOPs to tau on held-out (min of both sub-scores). Control: a
  naive fixed-budget RFF must not beat it.
- **F2 curiosity_discipline** — noise_fraction on the 12-noise-arm world AND reached; LP vs novelty trap.
- **F3 transfer** — banking shared primitives makes new unknowns cheaper. Control: ORTHOGONAL world must
  stay ~1.0x (no free lunch) or the facet is VOID.
- **F4 compositional_generalization** — zero-shot recombination of frozen primitive heads. Control: a
  lookup table (no entry for held-out tuples) must score ~chance.
- **F5 abstraction** — mint a reusable feature, causal A/B graft ablation; 0 if nothing minted.
- **F6 curriculum_self_direction** — endogenous goals vs the COMPETENT coverage baseline (honest ~parity,
  "rediscovers coverage from inside; does not invent goals"); gate on corr(want,realized)>0.25.
- **F7 world_model_fidelity** — rollout error vs a persistence (s'=s) control.
- **F8 imagination_planning** — model-based vs real planning at FIXED K,H; cap if task perf regresses;
  VOID if F7 rollout error is high (hallucinated planning).
- **F9 dimension_reach** — TWO-ARM gap: fixed RFF (walls ~d=2-3) vs learned deep encoder (crosses);
  score the measured d_max gap, never assert a prior result.
- **F10 generalization_gap** — inner vs held-out competence+cost gap (the anti-Goodhart keystone).
- **F11 self_improvement_rate** — LIVE from `monotonicity.jsonl` accept-ratchet (slope/halflife) +
  plateau-break demo; N/A unless source=='monotonicity_accepts' & n_points>=4. NEVER the static JSON.
- **F12 capacity_growth_autonomy** — forced walls → grows representation → reaches on held-out; cost bounded.
- **F13 grounded_language_first_communication** — samples-to-first-comm; control: shuffled-symbol → chance.
- **F14 grounded_language_compositional** — zero-shot symbol recombination; control: lookup → chance.
  Honest scope: zero-shot over novel COMBINATIONS of ATTESTED atoms, not novel atoms (rank diagnosis).
- **F15 out_of_family_robustness** *(the biggest missing anti-woo axis)* — train on the sine family,
  test sawtooth/square/Gaussian-bump AND input-distribution shift (x outside training support). A pure
  sine/periodic-fitter scores LOW (correctly). Likely the lowest facet — that IS the finding.
- **F16 retention_no_forgetting** — re-score season-1 held-out targets after season-N; measures decay.
- **F17 calibration** — RLS posterior covariance vs actual held-out error (does it know what it doesn't know).

## Build order (phases)
0 geometric-mean + median aggregate · 1 kill the two inflated facets (live self-improvement, two-arm
dimension-reach) · 2 re-anchor curriculum to coverage · 3 wire cheap existing probes (F2/F7/F8/F10) ·
4 controls AS CODE · 5 mint+graft (F5), capacity-growth replay (F12) · 6 the NEW axes (F15/F16/F17) ·
7 L5 facets + controls · 8 budget guard (<~10 min CPU), write run_logs/g_panel.json with raw+score+control_ok.
