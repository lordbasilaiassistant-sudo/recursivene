---
ts: 2026-06-16T14:18Z
topic: open-endedness-powerplay
sources:
  - Schmidhuber 2011 "PowerPlay: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem" (arXiv:1112.5309)
  - Wang Lehman Clune Stanley 2019 "Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions" (arXiv:1901.01753)
  - Lehman & Stanley 2011 "Abandoning Objectives: Evolution through the Search for Novelty Alone" (Evol. Comput.)
  - Mouret & Clune 2015 "Illuminating Search Spaces by Mapping Elites" (arXiv:1504.04909)
  - Stanley & Lehman 2015 "Why Greatness Cannot Be Planned: The Myth of the Objective" (Springer)
  - Hughes et al. (DeepMind) 2024 "Open-Endedness is Essential for Artificial Superhuman Intelligence" (arXiv:2406.04268)
---

# Open-Endedness and Self-Generated Curricula: a Non-Gameable Problem Generator

## TL;DR (3 sentences)
An open-ended learner needs a *problem generator* that proposes tasks which are neither already solved (trivial) nor currently unlearnable (impossible), and a *gate* that only admits a task once the learner provably improves on it WITHOUT regressing on any prior task. POWERPLAY does this with a hard "simplest still-unsolvable" search plus a no-forgetting validation pass; POET/MCC do it with a bounded fitness band (minimal criterion) plus cross-population transfer; QD/MAP-Elites do it by keeping an *archive of diverse elites* so progress is measured against the whole frontier, not a single objective. The anti-gaming property comes from the gate, not the generator: a self-invented problem only counts if a held-out, retrospective check confirms genuine learnable progress, which closes the loophole where the generator and solver collude to fake easy "wins."

## The core idea / key equation

**POWERPLAY (Schmidhuber 2011).** Search the joint space of `(new_task T, solver_modification δ)`. Accept the pair iff:
1. **Novelty / non-triviality:** the *old* solver `s` cannot solve `T` (and ideally not any previously-unsolved interesting task) within budget.
2. **Learnability:** the *modified* solver `s' = δ(s)` solves `T` within budget.
3. **No catastrophic forgetting (the hard gate):** `s'` still solves *every* previously mastered task `T_k (k<i)` within its original time/space bound. Implemented efficiently via per-task validation lists `L_ik` so the Correctness Demonstration only re-tests tasks plausibly affected by `δ`.
Among all admissible pairs, pick the one **found fastest** under a Levin/universal-search ordering (Algorithm: double the time limit until a sufficiently-likely program is found) — this operationalizes "*simplest* still-unsolvable problem." Variant I = strict no-forgetting; Variant II relaxes to allow forgetting only if total cost strictly drops by `ε` (`c_i* - c_i > ε`).

**POET / Minimal-Criterion Coevolution (Wang 2019).** Coevolve a population of `(environment E, agent θ)` pairs. A child environment `E'` (mutation of a parent) is admitted only if some current agent scores inside a band:
```
MC_low  ≤  fitness(θ, E')  ≤  MC_high
```
`fitness < MC_low` ⇒ too hard (unlearnable now) → reject. `fitness > MC_high` ⇒ too easy (trivial) → reject. Plus two engines: (a) **optimize** each paired agent on its env; (b) **transfer** — periodically test every agent on every env and reassign if a foreigner beats the incumbent. Transfer is what makes it open-ended rather than parallel curricula: a stepping stone discovered in env A unlocks env C.

**Novelty search & QD (Lehman/Stanley 2011, Mouret/Clune 2015).** Drop the objective; reward *behavioral novelty* = mean distance to k-nearest neighbors in a behavior-descriptor space `B(x)`. MAP-Elites generalizes: discretize `B` into a grid of bins, keep the single best (`elite`) per bin; a child only replaces an elite if it lands in a bin and beats (or fills) it. The archive is the curriculum — diversity is preserved by construction, so the search can't collapse onto one deceptive optimum. Thesis of *Why Greatness Cannot Be Planned*: ambitious objectives are deceptive; collecting *stepping stones* (novelty/diversity) reaches further than optimizing the target directly.

**Formal frame (Hughes 2024).** From an observer with a predictive model, a system is open-ended iff its output stream is simultaneously **novel** (artifacts become *less* predictable over any fixed model) AND **learnable** (conditioning on more history makes them *more* predictable). Random = novel but not learnable; a static dataset = learnable but not novel. This is exactly the (1)+(2) POWERPLAY band stated in information terms, and it is observer-relative — which maps cleanly onto RecursiveNe's per-region learner being the observer.

## Minimal numpy-implementable recipe (<40 lines described)
A POWERPLAY-flavored generator with a learnability band and a no-forget gate, sitting on top of RecursiveNe's existing `RegionLearner`:

1. State: `mastered` = list of accepted task params; `learner` = the world-model ensemble; `rng`.
2. `propose_task()`: mutate a *task descriptor* `t` (e.g. a frequency/phase/noise vector that parameterizes a world activity) off a random mastered parent, OR sample fresh — return a candidate `t`.
3. `pre_error(t)` = current 1-step prediction error of `learner` on a short rollout of `t` (no weight update). This is the novelty/triviality probe.
4. **Triviality gate:** reject if `pre_error(t) < tau_master` (already solved).
5. **Unlearnability gate:** reject if `pre_error(t) > noise_floor` AND short-burst learning makes no dent (see step 6) — the noisy-TV reject already implemented in `agent.py`.
6. **Learnability probe (the POWERPLAY accept test):** clone learner, run `min_lp` updates on `t`, compute `lp = pre_error - post_error` minus the per-task noise floor `c*std(resid)`. Require `lp > lp_floor`.
7. **No-forget gate (the hard part, cheap version):** after committing those `min_lp` updates, re-check held-out error on a fixed validation rollout from EACH `mastered` task. Accept only if `max_k val_err_k(after) ≤ val_err_k(before) + δ_tol`. This is POWERPLAY's `L_ik` correctness pass.
8. On accept: append `t` to `mastered`, keep the updated weights. On reject of (7): roll the weights back (snapshot before step 6) — this is the EWC-free, exact-replay version of catastrophic-forgetting protection.
9. **Curriculum ordering:** among accepted candidates in a batch, pick the one with the *smallest learnability-positive* `lp` that still clears `lp_floor` (cheapest genuine win) → "simplest still-unsolvable."
10. Maintain a MAP-Elites-style archive keyed by a 2-D task descriptor so the generator keeps proposing across the whole frontier instead of one corner.

`numpy` only: snapshot = `copy.deepcopy(learner)` or save/restore the RLS `(P, w)` matrices; everything else is dot products already in `model.py`.

## How this informs RecursiveNe's next layer (specific)
RecursiveNe today generates *solver-config* mutations (`recursivene/harness/space.py` `SEARCH_SPACE`, mutated by `harness/proposer.py`), gated by the protected objective. It does NOT yet generate *tasks/worlds*. The open-endedness layer is the missing dual: a **TaskProposer** that mutates `world.py` activity descriptors instead of solver knobs.

Concrete upgrades:
- **New file `recursivene/harness/task_space.py`** mirroring `space.py`: a `TASK_SPACE` of bounds over `world.py` activity parameters (freq, phase, noise, dimensionality), with `mutate_task()` reusing the exact log-normal mutation already in `space.py`.
- **New `TaskProposer(Proposer)` in `harness/proposer.py`** implementing steps 2–9 above. It already has the right seam: `propose(best, history, rng)` becomes "propose next *task*," and the existing protected gate in `harness/loop.py` is reused unchanged as POWERPLAY's acceptance check.
- **Reuse `agent.py RegionLearner` as the learnability oracle.** Its `lp` machinery (learning progress minus `noise_floor*std(resid)`, `tau_master`, `probe_min`) IS the (1) triviality + (2) unlearnability band — no new math. Wire `pre_error/post_error/lp` (steps 3–6) directly to the existing per-region LP fields; the `noisy-TV` reject is already correct.
- **Add the no-forget validation pass** as a method `RegionLearner.holdout_regression(mastered_tasks)` returning `max_k Δval_err` — this is the only genuinely new code and is ~15 lines (snapshot `(P,w)`, re-predict on stored validation rollouts, restore on reject). This upgrades the LP curriculum from "what to sample next" to "what task to *admit to the world* next, provably without regression."
- **Archive:** a `MAP-Elites` dict keyed by a 2-D `(difficulty, descriptor)` bin so `TaskProposer` illuminates the frontier (anti-collapse). Store the best solver weights per bin → free POET-style **transfer**: before optimizing a new task, seed from the nearest-bin elite.

Net: this is the L1 → open-ended-L2 step. The object level (world model) and meta level (config search) already exist; this adds the *environment-generation* level that makes the system its own teacher, with the no-forget gate inherited straight from POWERPLAY so "never forgetting old skills" is a proof obligation, not a hope.

## Pitfalls (top 2)
1. **Generator/solver collusion (gaming).** If the same gradient/objective both proposes tasks and scores success, the generator learns to emit tasks that are *trivially* inside the learnability band — fake progress. Defense: the accept test must use a **held-out, retrospective** check (the no-forget validation rollouts + a learnability *probe on a cloned learner*, not the live one). Make the descriptor space coarse (MAP-Elites bins) so "novelty" can't be faked by infinitesimal perturbations. POWERPLAY's strict no-forget gate is the non-gameable core; POET's MC band alone is gameable without transfer.
2. **The noisy-TV / unlearnable trap.** A task whose error is high but irreducible (pure noise) passes the triviality gate forever and starves real progress. RecursiveNe already mitigates this in `agent.py` via `noise_floor` + `probe_min` (confirm an above-floor arm is noise before chasing it) — the TaskProposer MUST route its learnability probe through that same logic, not raw prediction error, or open-endedness degenerates into noise-seeking. Symmetric failure: `MC_high`/`tau_master` set too loose admits near-trivial tasks and the curriculum stalls; calibrate the band on the seed worlds before running open-ended.

## Citations
- Schmidhuber, J. (2011). *PowerPlay: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem.* arXiv:1112.5309 / Front. Psychol. 4:313 (2013).
- Wang, R., Lehman, J., Clune, J., Stanley, K.O. (2019). *Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions.* arXiv:1901.01753 (GECCO 2019).
- Lehman, J., Stanley, K.O. (2011). *Abandoning Objectives: Evolution through the Search for Novelty Alone.* Evolutionary Computation 19(2):189-223. (Also: NSLC, "Evolving a diversity of virtual creatures," GECCO 2011.)
- Mouret, J-B., Clune, J. (2015). *Illuminating Search Spaces by Mapping Elites (MAP-Elites).* arXiv:1504.04909.
- Stanley, K.O., Lehman, J. (2015). *Why Greatness Cannot Be Planned: The Myth of the Objective.* Springer.
- Hughes, E. et al. (2024). *Open-Endedness is Essential for Artificial Superhuman Intelligence.* arXiv:2406.04268. (Formal novel+learnable, observer-relative definition.)
- Related recent (2023-2025): ACES "Generating Diverse Programming Puzzles with Autotelic Generative Models" (arXiv:2310.10692); OMNI "Open-endedness via Models of human Notions of Interestingness" (ICLR 2024); MAGELLAN "Metacognitive predictions of learning progress" (arXiv:2502.07709).
