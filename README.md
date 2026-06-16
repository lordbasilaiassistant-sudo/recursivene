# RecursiveNe

A from-scratch, **world-model-first** learner whose self-improvement loop is **closed over
itself** — it improves the model, the harness that improves the model, and the proposer that
drives the harness, all under one fixed objective: **cost-for-competence** (the "race to 0").
No pretrained weights. No downloaded models. No backprop, no replay buffer, no epochs, no
labeled corpus in the core. ~2k lines of numpy.

> The bet: LLMs are trained in reverse. They ingest the *output* of cognition (language)
> under massive supervision and back into a shaky world model. Evolution went the other way —
> an organism learns a predictive world model from its own self-supervised interaction stream,
> spends samples only where they buy competence, and language arrives **last**, grounded on the
> model. RecursiveNe is built in that order, and made to get **cheaper at it over time**.

---

## Weak RSI vs Strong RSI (the line this project draws)

**Weak RSI** is old: a *frozen* optimizer improves an artifact. Compilers compiling faster
compilers, AutoML, hyperparameter search. The optimizer is outside the search space, so the
system plateaus at the optimizer's ceiling.

**Strong RSI** is the human/cultural pattern: you improve *the means of improvement itself*.
Evolution evolves evolvability; science invents the instruments that enable new science;
culture builds the tools that build better tools. The improver is **inside** the search space
and can rewrite **itself**.

RecursiveNe implements strong RSI as a **three-level fixed point**, with the objective held
constant as the anchor:

1. **Object** — the world model predicts the world better/cheaper. *(model + curiosity)*
2. **Meta** — the harness/proposer gets better at improving the object level. *(learned search)*
3. **Meta-meta** — the proposer edits the proposer / the harness itself. *(self-modifying code)*

> Perfect RSI requires editing **both the model and the harness** — and the harness that edits
> the harness — with **no privileged frozen outer loop**. The only things never editable are the
> *objective* and a *held-out invariant*; everything that computes toward the objective is fair
> game, gated by a ruler the system cannot bend.

---

## The architecture

```
 PROTECTED (the ruler — never editable, not even by the proposer)
   objective.py     what "competence" and "cost-for-competence" MEAN
   invariant.py     held-out competence-per-FLOP — the anti-Goodhart floor
   world.py         the problem distribution (can't be trivialized)
   core/            kill switch + vitals + clock (the trusted kernel)
   closure/selfmod  the sandbox/commit kernel (the trust root)

 EDITABLE (everything that computes toward the objective)
   model.py agent.py seed.py     OBJECT: the learner
   harness/loop.py proposer.py   META: the improver  ← self-edit target
```

The loop is `propose → evaluate → select → integrate`, ratcheting `cost-for-competence` down.
A change to the *improver* (harness/proposer) is judged not by scoring one config but by
**running the edited improver and checking it finds cheaper, generalizing models** — the
meta-objective. Every self-edit is applied in a **sandbox copy**, evaluated in a **subprocess**,
and committed to the live tree only if it (a) lowers the meta-objective, (b) does not degrade
the **untouchable held-out invariant**, and (c) the **kill switch** permits — otherwise it is
rolled back. Write-scope opens **one stage at a time** (model → harness → proposer). Every
attempt is appended to a timestamped **monotonicity log**.

### Safety anchors
- **Untouchable invariant** (`invariant.py`): held-out worlds the inner loop never optimizes
  against; an edit that lowers it is auto-rejected. You can't win by overfitting or by redefining success.
- **Kill switch** (`core/killswitch.py`): a human `STOP` file + automatic tripwires (invariant
  floor, consecutive-failure cap, time budget). The component that *applies* edits checks it and
  is itself **protected**, so the system cannot edit out its own halt.
- **Sandbox + rollback**: a broken/crashing self-edit can never corrupt the running loop.
- **Anti-gaming**: inner-loop selection uses one set of worlds; the reward/invariant uses a
  **different** held-out set.

---

## Open-endedness — why a fixed test isn't enough

If the learner aces the test, it needs a *new* test. A fixed scorecard saturates and then
lies (Goodhart). So the **test-maker is inside the loop too**: a problem generator
(`generator/`) that, when the frontier is mastered, invents new activities gated to be
**novel + previously-unsolvable + now-solvable + non-forgetting** (Schmidhuber POWERPLAY; Wang
POET; Stanley/Lehman novelty search) — and mints **abstractions** that break plateaus. Progress
is measured by **non-saturating** benchmarks (`bench/`): repertoire growth, hardest-solved
complexity, transfer — not a fixed number. It is learning-progress all the way up:
sample → task → problem → abstraction → improvement-operator.

---

## Results (real numbers, reproduce with `experiments/`)

**Object level** — `python experiments/run_seed.py` (world: 6 learnable sine activities of graded
difficulty + 12 unlearnable "noisy-TV" distractors; competence = worst-case mastery):

| policy | cost-for-competence (FLOPs) | samples→competence | budget on noise |
|---|---|---|---|
| learning-progress (ours) | **3.4e7** | **882** | **13%** |
| random (passive) | 4.6e7 | 1195 | 57% |
| novelty (naive curiosity) | ∞ — never reaches | ∞ | 95% (trapped on the TV) |

LP beats random *and* novelty on cost-for-competence (8 seeds), and avoids the noisy-TV that
traps naive curiosity. (H1, H2.)

**Meta level** — `experiments/run_rsi.py` / the closure stage-1 ratchet drives a deliberately
bloated start down the race-to-0 curve while holding competence:

```
cost 1.60e8 (D=256) → 1.37e8 → 4.33e7 → 3.55e7 → 7.95e6 → 4.54e6 → 1.17e6 (D=20)
                                                       136× cheaper, competence held (H3)
```

**Meta-meta level** — `python experiments/run_closure.py` runs the full autonomous loop
(self-edits to `harness/loop.py` then to `harness/proposer.py`, each sandboxed and gated),
writing `run_logs/closure_summary.json` + `run_logs/monotonicity.jsonl` + `vitals/`.

**Gate** — `python experiments/validate.py` asserts H1–H3 + closure (self-edits applied,
invariant never degraded, protected core untouchable, monotone cost) and prints PASS/FAIL.

> Scope: this is a *toy-scale* demonstration of the principle, not literally Fable-on-a-laptop.
> The claim it earns is that the **mechanism** — a closed, three-level, invariant-anchored,
> kill-switched self-improvement loop that monotonically lowers compute-for-competence — runs
> end-to-end with no human in the inner loop. Pushing the substrate up the L0→L5 roadmap
> (`notes/01-roadmap.md`) is how the floor keeps dropping.

---

## Layout
- `recursivene/` — the package (see `ORCHESTRATION.md` for ownership; `CONTRACTS.md` for interfaces)
- `experiments/` — `run_seed.py`, `run_rsi.py`, `run_closure.py`, `validate.py`
- `notes/` — thesis, roadmap, citations, PROGRESS tracker
- `knowledge/` — research feed (the project teaches itself the literature over time)
- `run_logs/` — monotonicity log + closure summary (the audit trail)
- `vitals/` — parent + child heartbeats; `tools/dashboard.py` reads them

## Nous — the synthesis (one entity, everything together)
`recursivene/entity.py` fuses every proven piece into a single self-improving knower with a
persistent identity and one drive: make unknowns known, ever more cheaply, forever. Each season it
faces a fresh unknown at its frontier, makes it known with a learned representation (cheaply, via
emergent transfer), grows its own representation when the frontier outruns it, periodically races
its own learner to 0, honors a kill switch it cannot edit, and persists itself so the same entity
resumes next session. Across two sessions it made 40 unknowns known (complexity 13→37), grew its
representation, and self-improved its learner. See `notes/03-entity.md`.

## Run
```
python experiments/validate.py        # fast gate (H1, H2, H3, safety) — reads closure artifacts
python experiments/validate_entity.py # the synthesis gate (8/8: object+L1+growth+RSI+safety+identity)
python experiments/run_entity.py      # wake Nous and let it live (run twice — it resumes)
python experiments/run_closure.py     # the full autonomous 3-level code-self-edit run (slow)
python tools/dashboard.py             # live vitals of the parent + RSI child
```
