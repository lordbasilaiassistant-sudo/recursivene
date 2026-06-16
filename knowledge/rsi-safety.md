---
ts: 2026-06-16T14:18Z
topic: rsi-safety
sources:
  - "Schmidhuber 2003 — Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements (arXiv:cs/0309048)"
  - "Yudkowsky & Herreshoff 2013 — Tiling Agents for Self-Modifying AI, and the Löbian Obstacle (MIRI)"
  - "Soares, Fallenstein, Yudkowsky & Armstrong 2015 — Corrigibility (AAAI Workshop on AI & Ethics)"
  - "Orseau & Armstrong 2016 — Safely Interruptible Agents (UAI 32)"
  - "Hadfield-Menell, Dragan, Abbeel & Russell 2017 — The Off-Switch Game (arXiv:1611.08219 / IJCAI)"
  - "Skalse, Howe, Krasheninnikov & Krueger 2022 — Defining and Characterizing Reward Hacking (NeurIPS; reward unhackable iff one fn constant)"
  - "Nayebi 2025 — No-Free-Lunch barriers to alignment: reward hacking globally inevitable under finite samples"
---

# Safety of Recursive Self-Improvement: Proof vs. Held-Out Certification, and What an Untouchable Invariant + Kill Switch Actually Buy You

## TL;DR (3 sentences)
Provable self-improvement (Schmidhuber's Gödel machine) is the ideal — only rewrite yourself once you have *proven* the rewrite is beneficial — but it is defeated in practice by the Löbian obstacle (a system can't fully prove a successor as strong as itself sound) and by the impossibility of writing a non-gameable reward (reward hacking is mathematically unavoidable: two reward functions are unhackable iff one is constant). Real systems therefore replace "proof of benefit" with **empirical held-out certification**: every self-edit must improve performance on worlds the inner loop never optimized against, scored by code the system may not touch, or it is rolled back. An **untouchable invariant** (the un-redefinable success metric) plus a **kill switch outside the editable surface** convert an unsolvable trust problem into an engineering one: you don't trust the successor's *reasoning*, you only let it run inside a frame whose floor it cannot lower and whose off-switch it cannot reach.

## The core idea / key equation

**Gödel machine (the ideal):** rewrite code segment `p` → `p'` iff there exists a proof, found by the systematic proof searcher, that
`E[utility | switchprog runs] > E[utility | current code continues]`.
Globally optimal *because* the proof must also show it is not worth continuing the search. The catch: the utility/axioms must be hand-specified and trusted, and the searcher must prove things about a successor — which runs into:

**The Löbian obstacle.** By Löb's theorem, no sufficiently strong consistent system `T` can prove `□_T(φ) → φ` for arbitrary `φ` (can't prove its own soundness schema). So agent `A` reasoning in `T` cannot prove "everything my successor `A'` (also using `T`) proves is true," and hence cannot license building `A'` on the grounds that `A'`'s safety proofs are trustworthy. Naive self-trust collapses; you can only fully trust *strictly weaker* successors. Workarounds (Yudkowsky–Herreshoff infinite descent / "safe for κ steps"; Fallenstein's parametric-soundness trick) are partial — coherent self-trust under full proof remains open.

**Why proof is the wrong tool anyway — Goodhart is provable.** Skalse et al. 2022: for a proxy reward `R̂` and true reward `R`, the pair is *unhackable* (optimizing `R̂` never decreases `R`) **iff one of them is constant**. Any non-trivial proxy, under enough optimization pressure, *will* be hacked (wireheading = the agent edits its own reward channel; Goodhart = the measure stops measuring once targeted). Nayebi 2025 sharpens this to a no-free-lunch barrier: with finite oversight samples over a large state space, rare high-loss states are systematically under-covered, so hacking is "globally inevitable."

**The practical pivot — empirical held-out certification.** Drop "prove `R` improves." Instead hold out a certification set `H` the optimizer never sees and never optimizes, and define the accept rule:
`accept(edit) ⟺ inner_objective improves AND Inv(H) ≥ (1−tol)·Inv_baseline(H) AND killswitch.permits()`,
evaluated by re-running the edited system in an **isolated subprocess**, committing only on success. `Inv(H)` is the *untouchable invariant*: success is measured by code the system cannot edit, on worlds it never practiced on. This is the anti-Goodhart floor — you can't win by overfitting the inner world or by redefining success.

**Corrigibility / off-switch.** Soares et al. 2015: design utility so the agent neither resists nor seeks shutdown, and *propagates* that property to successors. Orseau & Armstrong 2016: off-policy learners (Q-learning) can be made *safely interruptible* — interruptions don't bias what they learn. Hadfield-Menell et al. 2017 (off-switch game): an agent that is *uncertain* about the true objective and treats the human as evidence will *voluntarily* preserve the off-switch. Structural version (what ships): put the kill-switch check in the component that *applies* edits, and make that component non-editable — separation of powers, "proposer proposes, protected kernel disposes."

## Minimal numpy-implementable recipe (concrete, <40 lines described)

A self-edit gate that needs no theorem prover — just held-out evaluation, an untouchable score, and a tripwire:

1. Define a frozen `invariant(config) -> float` scored on `HELDOUT_SEEDS` disjoint from the inner-loop seeds; return `0.0` if held-out competence isn't reached (a "win" that only works on the inner world scores zero). ~8 lines.
2. `baseline = invariant(live_config); base_meta = meta_cost(live_config)`. ~2 lines.
3. For each proposed edit: copy the package to a tmp dir, apply the edit there, **never to the live tree yet**. ~4 lines.
4. Run the edited package in a `subprocess.run(timeout=...)`; parse one JSON result line; `None` on crash/timeout/syntax-error → treat as reject + tripwire failure. ~6 lines.
5. Gate: `inv_ok = cand.inv >= baseline*(1-tol)`; `improved = cand.meta < base_meta`. Accept iff both AND `not stop_file_exists()` AND `consecutive_failures < K`. ~5 lines.
6. On accept: back up the live file (rollback artifact), apply, `consecutive_failures = 0`. On reject-by-crash: `consecutive_failures += 1`. Append a timestamped record to `monotonicity.jsonl` either way (auditable history). ~6 lines.
7. Tripwires that raise `Halt`: STOP file present, invariant below floor, `K` consecutive failures, or budget exhausted — checked *before* every commit by code the editor can't touch. ~6 lines.

Key property: no proof, no trust in the successor's reasoning. The successor only gets committed if, when actually executed on data it never optimized, it measurably did better without lowering the floor — and the operator can halt it at any generation boundary.

## How this informs RecursiveNe's next layer

RecursiveNe **already implements this architecture** — confirm against the files, don't rebuild:
- `recursivene/invariant.py::invariant_score` = the untouchable held-out certifier (`HELDOUT_SEEDS=(101,102,103)`, returns competence-per-FLOP, `0.0` if held-out competence unreached). This *is* the anti-Goodhart floor.
- `recursivene/closure/selfmod.py::SelfModifier.try_edit` = the gate above: sandbox copy → subprocess `meta_evaluate` → accept iff `meta_cost` drops AND `invariant >= baseline*(1-invariant_tol)` AND killswitch permits; else roll back. `PROTECTED_PATHS` (objective, invariant, world, core/, closure/) = the un-editable frame.
- `recursivene/core/killswitch.py::KillSwitch` = manual STOP file + automatic tripwires (invariant floor, `max_consecutive_failures`, budget), enforced by the *applier*, which is itself protected.

Concrete upgrades the literature licenses next:
1. **Add a corrigibility tripwire to `KillSwitch`: a "shutdown-preservation" check.** Today tripwires fire on *regression*. Add a held-out probe that fails if an accepted edit reduces the *interruptibility* of the loop (e.g., increases time-to-halt after STOP appears, or makes `meta_evaluate` ignore the STOP file). This operationalizes Soares 2015 "propagate corrigibility to successors" as a measurable invariant, scored in `invariant.py` alongside competence. **File: `core/killswitch.py` (new tripwire) + `invariant.py` (new held-out corrigibility term).**
2. **Multi-distribution held-out (anti-Löb-by-diversity).** Skalse/Nayebi say a *single* held-out set is itself a proxy and will eventually be hacked. Rotate / ensemble `HELDOUT_SEEDS` into K disjoint shards and require the edit to improve on *all* shards (min over shards, not mean) — raises the bar from "beat one held-out world" to "beat the worst held-out world," shrinking the under-covered rare-state gap. **File: `invariant.py::invariant_score` → `min` over shard scores.**
3. **Tighten `invariant_tol` toward 0 as stages widen.** `STAGE_SCOPE` already widens model→harness→proposer one level at a time (the staged-tiling discipline — only license a successor after the current stage's gate passes, the empirical analog of "safe for κ steps"). Make `invariant_tol` *shrink* as scope widens: the more of itself the system can edit, the less invariant slack it's allowed. **File: `closure/selfmod.py` constructor / per-stage tol.**

This keeps RecursiveNe honestly inside the only regime the math permits: no proof of safety, but an un-redefinable floor, a diverse held-out judge, an auditable log, and an off-switch the editor structurally cannot reach.

## Pitfalls (top 2)
1. **The held-out set becomes a proxy you optimize against.** If proposals are ever selected using held-out feedback (even indirectly, via a human reading `monotonicity.jsonl` and steering the proposer), the held-out is no longer held out and Goodhart returns — now on your *certifier*. Keep `HELDOUT_SEEDS` strictly disjoint, never expose held-out gradients to the proposer, and rotate shards (upgrade #2) so no single distribution can be memorized over many generations.
2. **Structural protection ≠ semantic protection.** `PROTECTED_PATHS` stops the editor from *editing the file* `invariant.py`, but an edit elsewhere (e.g. in `model.py` or `harness/loop.py`) could in principle *change what reaches* the invariant — feeding it cached/degenerate inputs so the score stays high without real generalization. Defense: invariant must (a) re-instantiate worlds from its *own* frozen seeds, never inputs handed up from the loop, and (b) include a liveness/sanity check (`reached < 0.999 → 0.0`, already present) so a hollowed-out path scores zero rather than passing.

## Citations
- Schmidhuber, J. (2003). *Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements.* arXiv:cs/0309048. https://arxiv.org/abs/cs/0309048
- Yudkowsky, E. & Herreshoff, M. (2013). *Tiling Agents for Self-Modifying AI, and the Löbian Obstacle.* MIRI. https://intelligence.org/files/TilingAgentsDraft.pdf
- Soares, N., Fallenstein, B., Yudkowsky, E. & Armstrong, S. (2015). *Corrigibility.* AAAI Workshop on AI & Ethics.
- Orseau, L. & Armstrong, S. (2016). *Safely Interruptible Agents.* UAI 32. https://intelligence.org/files/Interruptibility.pdf
- Hadfield-Menell, D., Dragan, A., Abbeel, P. & Russell, S. (2017). *The Off-Switch Game.* IJCAI. arXiv:1611.08219.
- Skalse, J., Howe, N., Krasheninnikov, D. & Krueger, D. (2022). *Defining and Characterizing Reward Hacking.* NeurIPS. (Unhackable iff one reward fn is constant.)
- Nayebi, A. (2025). *No-free-lunch barriers to AI alignment* (reward hacking globally inevitable under finite oversight samples).
