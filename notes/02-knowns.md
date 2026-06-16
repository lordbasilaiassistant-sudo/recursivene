# RecursiveNe — Unknowns Made Known

The directive: *make unknowns known*. Not by argument — by experiment. This is the ledger of
questions that were open or speculative, and what running the system turned them into. Each
KNOWN cites the run that established it. Each REMAINING UNKNOWN is named honestly, because a
gardener who pretends the garden is finished is lying.

## Made KNOWN (measured, reproducible)

1. **Learning-progress curiosity is the right attention rule — not novelty.** In a world that is
   67% unlearnable noise, the LP learner reaches competence ~30% cheaper than random and spends
   13% of its budget on noise; naive novelty spends 96% and NEVER reaches competence (trapped on
   the noisy TV). *Evidence: validate.py H1/H2, 6 seeds.*

2. **The race to 0 is real.** Starting deliberately bloated, the harness drove cost-for-competence
   from 1.6e8 (D=256) to 1.2e7 (D=67), 136× cheaper, with competence held. *Evidence: H3, bench
   slope −0.83/gen.*

3. **A learned proposer beats blind search at improving the learner.** The model of the
   improvement landscape found cheaper, generalizing configs than mutation (held-out 1.36e7 vs
   2.01e7). The meta level genuinely gets better at its job. *Evidence: B3.*

4. **Strong RSI runs end-to-end.** The system autonomously edited its own harness
   (MUTATION_SCALE 0.5→0.7) AND its own proposer (EXPLORE_FRAC 0.34→0.6), each sandboxed and kept
   only because it survived a held-out test it could not see or edit. No human in the inner loop.
   *Evidence: closure stages 2+3, monotonicity log.*

5. **The safety anchors hold under test.** Tampering with the objective is refused (protected);
   every accepted self-edit left the untouchable invariant intact; the kill switch halted a live
   run on command. *Evidence: validate D2/D3, the STOP-file halt.*

6. **The loop COMPOUNDS — the seed grew itself.** Tended for 22 seasons, RecursiveNe made a
   continuous stream of harder unknowns known, climbing from frequency w=12 to w=45 (≈3.75× harder)
   and repertoire 6→28. Four times the frontier became *unknowable* at its current capacity and the
   system GREW its own perceptual bandwidth (gamma 8→22.8) and capacity (D 96→235) until that
   unknown became knowable. It never faked a result: when a frequency was capacity-unlearnable it
   reported "unreachable" rather than inventing mastery. *Evidence: run_garden.py, run_logs/garden_rings.jsonl.*

7. **Growth keeps the cost of new knowledge BOUNDED.** As the frontier advanced 3.75×, cost-to-know
   per rung stayed in the 120–700-sample band (with occasional random-feature-misalignment spikes
   that the next growth resolved). Growing capacity to meet ambition kept each new unknown
   affordable instead of letting cost explode. *Evidence: the cost-to-know sparkline.*

8. **Cost-to-know CAN be made decreasing — but only with shared structure.** A learner that banks a
   small library of shared primitives makes each new *compositional* unknown cheaper: cost-to-know
   fell **3.7× (74→20 samples)** as the library filled, bottoming out at "just fit the coefficients."
   In a world of near-orthogonal tasks (pure single frequencies — the garden) the same banking gives
   **zero transfer (1.00×)**: cost stays flat. So the garden's bounded-flat cost was not a weakness —
   it is the *correct* behavior when tasks share no structure (no free lunch). Decreasing cost-to-know
   — the true open-ended race-to-0 — is real, and its precise lever is **shared structure + an
   abstraction that banks it.** *Evidence: transfer_test.py.*

9. **Abstraction/transfer EMERGES — the system finds structure in its own past work.** Given only
   a stream of compositional targets (primitives hidden), banking past solutions as features made
   each new unknown **4.6× cheaper** (123→27 samples) as the bank came to span the hidden subspace;
   a naive learner stayed flat. No frequency detector — the span of past solutions IS the discovered
   abstraction. *Evidence: discover_test.py. (First design failed honestly — dense correlated bank
   smears salience, smooth targets already cheap — and the fix is recorded in that file.)*

10. **The phenomena survive in 2-D — and taught a real lesson.** With a 2-D world model, LP still
    beats random on cost-for-competence and still avoids the noisy TV (2.7% vs novelty's 100%). But
    getting there exposed a genuine bug: mastery judged by TRAINING error lets a high-capacity model
    "master" an arm by OVERFITTING before it generalizes, so LP abandons it and loses to random.
    The curiosity/mastery signal must track GENERALIZATION, not training fit. (And the race-to-0 —
    shrinking D — is itself a regularizer against this trap.) *Evidence: exp_2d.py.*

11. **Transfer gives a level discount, but does NOT bound open-ended HARDER growth.** Running growth
    + emergent transfer together: banking kept every new unknown ~30% cheaper, but as complexity rose
    (14→29) cost-to-know still climbed ~1.5× for both naive and banked. Transfer amortizes REUSED
    structure; it cannot make genuinely HARDER new structure cheap. Bounding the open-ended race-to-0
    needs a representation whose cost does not grow with complexity — i.e. L1. *Evidence: capstone.py.*

12. **L1 (a learned representation) FLATTENS the cost-to-complexity curve — the blocker is removed.**
    Same open-ended stream as #11, but the learner uses a shared SpectralEncoder that DISCOVERS the
    data's frequencies from its own buffer and represents them at fixed cost (RFF fallback for the
    not-yet-discovered). As complexity climbed 14→29, fixed RFF cost-to-know exploded **28.5×**
    (high frequencies effectively unlearnable at fixed bandwidth) while L1 stayed **flat (0.84×)**
    and ended **~103× cheaper**. A learned representation keeps new unknowns affordable as
    complexity climbs — so the open-ended race-to-0 (#11's blocker) is reachable. This also answers
    the meta-acquisition question (#B' below): the encoder gets BETTER at acquiring new structure by
    discovering it once and banking it into the representation. *Evidence: encoder.py, l1_test.py.*

13. **It learns a genuinely SENSORY field.** Given only random coordinate→intensity samples from a
    2-D image (blobs + gradient + a high-frequency ring), the substrate reconstructs the scene to
    **99.7% of its variance** (MSE 0.0003) — visibly, in ASCII. Too-low feature bandwidth blurs the
    detail; the right bandwidth captures it (the L1 lesson, in pixels). It is a general smooth-field
    learner, not a sinusoid trick. *Evidence: sensory_test.py, generality_test.py.*

14. **Scaling has a measured shape: graceful in complexity, a hard wall in dimension.** Cost-to-know
    stayed BOUNDED as components grew 2→32 (graceful, with ~linear capacity — and the garden grows
    capacity autonomously). But input DIMENSION hit a hard wall: d=1→25, d=2→175, d=3→725, **d≥4 →
    unreachable** — the curse of dimensionality, because fixed random features need ~exp(d) to cover
    d-dim frequency space. The next wall is now LOCATED, not asserted. *Evidence: scaling_test.py.*

15. **A learned representation CROSSES that wall.** On the exact d=4 target that walled fixed
    features (MSE stuck at 0.70 with 12000 samples), a 1-hidden-layer learned representation reaches
    competence at **N=500** (MSE 0.014 → 0.001 at N=2000). Learned deep representations defeat the
    curse of dimensionality where random features cannot — the empirical foundation of modern ML,
    re-derived on RecursiveNe's own wall. This is exactly L2's job. *Evidence: deep_test.py.*

16. **L2 is BUILT and gated — the learned representation is now a reusable rung.** `deep_encoder.py`:
    the dimension-wall-crosser (1-hidden-layer tanh MLP, Adam) packaged as a clean L2 backend with the
    project's fit/predict/cost_to_know contract. Demonstrated: it makes known **4/4** of a stream of
    **d=4 compositional unknowns** (N=500–1000 each) where fixed features wall **0/4**, AND learns a
    real **2-D sensory image to 99% recovery** (relative MSE 0.8%) from raw coordinates. The entity's
    reach is no longer 1-D frequencies. *Evidence: run_l2.py, validate_l2.py (2/2 PASS).*

17. **L2 is wired into a LIVING entity — Nous-L2 lives in multi-D worlds.** `entity_l2.py`: the learned
    representation (SharedDeepBackend, persistent + growable) as the entity's world-model backend, with
    the full scaffolding (identity, vitals, kill switch, race-to-0, capacity growth). Nous-L2 made **15+
    unknowns known in a d=3 world and got 12.7× cheaper over its life** (held-out MSE 0.0060→0.0005) as
    its body accumulated the world's shared structure — learned transfer past the curse of
    dimensionality — while tuning its own learning rate and persisting identity across sessions.
    *Evidence: run_l2_entity.py, run_entity_l2.py, validate_l2_entity.py (3/3 PASS).*

## REMAINING — the experimental program has reached its fixpoint

Every question raised has been answered with code+output, and where an answer REQUIRED building
(the L1 SpectralEncoder, a sensory image task, a learned deep net), it was built and run — not
deferred. The chain of walls is fully grounded: fixed features wall on complexity → transfer gives
only a discount → a learned spectral representation flattens complexity → dimension still walls →
a learned deep representation crosses the dimension wall. There is no remaining question about the
*current system* that a focused experiment can answer; what is left is ENGINEERING on principles now
demonstrated:

- **Build L2** — fold the learned deep representation (#15) into the closed loop as the world-model
  encoder, so the whole system (curiosity + race-to-0 + open-ended growth + meta-closure) runs on a
  representation that beats the curse. The pieces are proven individually; integration is the work.
- **Real high-dim sensory + scale** — apply L2 to genuine perception. The principle (#13, #15) holds;
  the open part is degree, and degree is settled by scale, not by a one-file experiment.

"Scales to general intelligence" remains unprovable — but it is no longer a shrug: every wall on the
path so far has been located by experiment AND shown crossable by a learned representation. The bet
now rides on building L2 and pushing scale, with each prior rung demonstrated.
