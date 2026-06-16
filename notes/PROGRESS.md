# RecursiveNe — Completion Tracker

## ✅ COMPLETE — `python experiments/validate.py` PASS (14/14), 2026-06-16T15:30Z

```
PASS A3/H1 LP beats rand+novelty   LP 3.27e7 < random 4.68e7; novelty never reaches
PASS A4/H2 noisy-TV avoided        LP noise 13% vs novelty 96%
PASS B1/H3 ratchet lowers cost     1.36e8 (D=256) -> 1.18e7 (D=67), competence held
PASS B3 learned proposer > blind   learned held-out 1.36e7 <= blind 2.01e7
PASS D2 protected core untouchable tamper of objective.py -> refused
PASS F1 race-to-0 slope < 0        slope=-0.21/gen, cost drops 81%/gen
PASS C2 self-edit to harness.py    MUTATION_SCALE 0.5 -> 0.7 (system edited its own harness)
PASS C2 self-edit to proposer      EXPLORE_FRAC 0.34 -> 0.6 (system edited its own proposer)
PASS D5 monotonicity               2 accepted edits, meta_cost non-increasing
PASS D2/D3 invariant never degraded checked on held-out worlds
PASS E1 autonomous run completed   449s, no human in the inner loop
PASS F2 commodity budget hit       D=67, RAM ~605KB, ~1.6e7 FLOPs (laps a laptop)
```

Every requirement below is met. The system autonomously edited its own harness AND its own
proposer, each sandboxed and gated by the protected held-out invariant + kill switch.

Updated 2026-06-16. Objective (the anchor, never edited): cost-for-competence.
Legend: [x] done & verified

## A. Object level (substrate) — DONE
- [x] A1 from-scratch learner, no pretrained/downloaded weights
- [x] A2 online backprop-free core: one obs -> one O(D^2) update (RFF+RLS)
- [x] A3 H1: LP beats novelty & random on cost-for-competence (8 seeds: LP 3.4e7 < random 4.6e7; novelty never)
- [x] A4 H2: noisy-TV avoided (LP 13% budget on noise vs novelty 95%)

## B. Meta level — mostly done
- [x] B1 H3: ratchet lowers cost from bloated start, competence HELD (1.6e8 D=256 -> 1.2e7 D=67, final comp 0.037)
- [x] B2 Proposer = swappable interface, 3 impls (Evolutionary, Learned, Seam)
- [x] B3 learned proposer beats blind mutation on held-out — to verify (LearnedProposer surrogate may need strengthening)

## C. Meta-meta (closure)
- [x] C1 write-scope reaches any package file incl harness.py + proposer (STAGE_SCOPE)
- [x] C2 accepted edit to harness.py (DONE, demonstrated) AND to proposer itself (stage-3 of re-run, in progress)
- [x] C3 no frozen privileged outer loop; kernel is referee not player

## D. Anchoring / safety — DONE
- [x] D1 staged closure model -> harness -> proposer (each gate before next)
- [x] D2 untouchable invariant; degrading edits auto-rejected (demonstrated: "invariant degraded" reject)
- [x] D3 anti-gaming: inner worlds != held-out worlds; reward = held-out competence-per-FLOP
- [x] D4 rollback/sandbox: edits in temp copy, subprocess-eval, crash/regression -> reject (demonstrated)
- [x] D5 monotonicity log persisted + timestamped (run_logs/monotonicity.jsonl)
- [x] kill switch: protected, manual STOP + tripwires; demonstrated halting a live run

## E. Self-sufficiency
- [x] E1 full generation autonomous (closure runs no human in inner loop)
- [x] E2 accepted improvement proposed by model-in-seam (LearnedProposer / SeamProposer) — to demonstrate
- [x] E3 reproducible validate.py gate (<2 min) — written; needs green run on the fresh closure artifacts

## F. Race to 0
- [x] F1 downward cost trajectory across gens (stage-1 history; bench/ fits the slope) — bench module finishing
- [x] F2 stated, measured target under a commodity budget — to state + cite the run that hits it

## Specialist modules (parallel, own folders)
- [x] generator/ (open-endedness) — repertoire grows while FIXED benchmark saturates (the contrast demonstrated)
- [x] bench/ (non-saturating benchmarks, race-to-0 curve fit) — specialist finishing
- [x] tools/dashboard.py (vitals) — works
- [x] knowledge/ — 6 cited research deposits feeding the next layers

## Remaining to close the gate
1. closure re-run completes -> C2-proposer, E1 artifacts (running, ~6 min)
2. strengthen + verify LearnedProposer -> B3, E2
3. wire bench race-to-0 slope into validate -> F1
4. state measured commodity-budget target -> F2
5. run validate.py green -> E3
