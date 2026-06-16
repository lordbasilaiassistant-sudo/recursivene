---
ts: 2026-06-16T14:18Z
topic: latent-world-models-jepa
sources:
  - Assran et al. 2023 — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA), arXiv:2301.08243
  - LeCun 2022 — A Path Towards Autonomous Machine Intelligence (H-JEPA position paper), OpenReview BZ5a1r-kVsf
  - Bardes, Ponce & LeCun 2022 — VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, arXiv:2105.04906
  - Grill et al. 2020 — Bootstrap Your Own Latent (BYOL): EMA target + stop-gradient without negatives, arXiv:2006.07733
  - Anonymous 2025 — Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring (LPM), arXiv:2509.25438
  - Mavor-Parker et al. 2021 — How to Stay Curious while Avoiding Noisy TVs using Aleatoric Uncertainty Estimation, arXiv:2102.04399
---

# Moving Surprise into Representation Space: JEPA Targets for Noise-Proof Curiosity

## TL;DR (3 sentences)
JEPA computes prediction error in a *learned representation space* produced by an encoder whose target copy is an EMA of the online encoder with a stop-gradient, so the predictor is trained to match abstract structure rather than raw pixels — and because an encoder is free to *drop* aleatoric (unpredictable) content, the noisy-TV's variance is encoded to a near-constant and registers ~zero surprise. The single thing that makes this work instead of collapsing to a trivial constant is the asymmetry invariant: gradient flows ONLY through the online/predictor side, the target side is EMA + stop-grad, and a variance/covariance regularizer (VICReg) forbids the encoder from solving prediction by emitting a constant. For RecursiveNe this means surprise/learning-progress should eventually be measured on `encoder(obs)` not `obs`, which moves the noise rejection from a hand-tuned `noise_floor` scalar into a *learned* representation that the noise simply cannot occupy.

## The core idea / key equation
Replace "predict the observation, measure error in observation space" with "predict the *representation* of the target, measure error in representation space." Three networks:

- **Online (context) encoder** `f_theta` — trainable, gradient flows.
- **Target encoder** `f_xi` — same architecture, NO gradient. Parameters track the online encoder by EMA: `xi <- m*xi + (1-m)*theta`, with momentum `m` ramped `0.996 -> 1.0`.
- **Predictor** `g_phi` — trainable, maps context representation (+ a position/conditioning code `z`) to the predicted target representation.

Per step (context `x_c`, target `x_t`, conditioning `z`):

    s_hat = g_phi( f_theta(x_c), z )          # prediction lives in rep space
    s     = stopgrad( f_xi(x_t) )             # EMA target, NO gradient
    L_pred = SmoothL1( s_hat, s )             # I-JEPA uses smooth-L1 / L2 in latent space

This alone collapses (encoder learns `f ≡ const`, making every prediction trivially exact). The **collapse-prevention invariant** is what forbids that. EMA+stop-grad (BYOL/I-JEPA) is empirically enough in vision; the *explicit* and auditable version is VICReg added on the online embeddings of a batch `Z = f_theta(X)` (d-dim):

    var_loss = mean_j  max(0, gamma - sqrt(Var(Z_:,j) + eps))     # gamma=1  -> every dim must spread
    cov_loss = sum_{i!=j} Cov(Z)_ij^2 / d                          # decorrelate dims (no redundant collapse)
    L = lambda*L_pred + mu*var_loss + nu*cov_loss                  # lambda=25, mu=25, nu=1 (VICReg defaults)

**Invariant, stated once:** the predictor minimizes distance to a target it cannot influence (stop-grad), and the encoder is held off the constant solution by `var_loss` (per-dim std forced >= gamma). Break either half and the system finds the noisy-TV-free-lunch: a constant representation that is perfectly predictable. The noise rejection is then automatic — pixels whose only signal is aleatoric carry no gradient that survives the variance constraint, so the encoder allocates capacity to *predictable* structure and the noisy-TV maps to a near-constant latent => near-zero `L_pred` => no curiosity reward.

## Minimal numpy-implementable recipe (concrete, <40 lines described)
Linear-encoder JEPA on the existing RFF feature space — validates the mechanism with no torch.

1. `W = rng.normal(size=(d, D)) * scale` — online encoder weights, `d` latent dims, `D` input/RFF dims. Encoder = `f(x) = W @ phi(x)` (reuse `RFFOnlineRegressor`'s `phi`).
2. `Wt = W.copy()` — target encoder; `P = rng.normal(size=(d, d))*scale` — predictor (conditioning `z` appended as extra input cols if you have multiple target blocks).
3. Per observation pair (context block, target block) from the SAME world step:
   - `s_hat = P @ (W @ phi(x_c))`
   - `s = Wt @ phi(x_t)`  ; `# stop-grad: never backprop through Wt`
   - `e = s_hat - s` ; `L_pred = 0.5*(e@e)` — THIS scalar is the representation-space surprise.
4. Gradients (closed form, online side only): `dP`, `dW` from `L_pred`; add VICReg grads on a small running batch `Z` of `W@phi(x)`: variance hinge pushes each `Z_:,j` std up to `gamma=1`, covariance pushes off-diagonals of `cov(Z)` to 0. SGD step on `W,P` only.
5. EMA: `Wt = m*Wt + (1-m)*W` with `m=0.99` (small-net value; ramp toward 1).
6. Curiosity signal = a windowed *downward trend* of `L_pred` (reuse the exact `learning_progress` slope+significance-floor logic from `agent.py`) — now on latent error, not observation error.

20-line falsification test (write FIRST, in `experiments/`): build a 2-arm world — arm A a learnable `sin`, arm B pure Gaussian noise. Run this latent JEPA. **Falsifier:** if mean `L_pred` on arm B does NOT decay to within noise of arm A's *mastered* error (i.e. the encoder fails to absorb the noise into a constant latent), the representation-space claim is false for this world and you stay in observation space. **Collapse check (second falsifier):** if `var_loss -> 0` while `min_j std(Z_:,j) << gamma`, the encoder collapsed — invariant violated, recipe rejected. Log `std(Z)` every step.

## How this informs RecursiveNe's next layer (specific)
Today curiosity is computed in OBSERVATION space: `agent.py::RegionLearner.observe(r,x,y)` calls `models[r].update(x,y)->err`, squares it into `err_hist[r]`, and `learning_progress(r)` / `choose("lp")` read that. Noise is rejected by two *hand-set scalars*: `noise_floor=0.7` and `tau_master`. That is brittle — it only works because the toy world's noise variance (~1.0) happens to sit above learnable error (<=0.5); it will not survive richer observations where signal and noise overlap in raw space.

Upgrade path, contract-safe (the `observe`/`learning_progress`/`recent_error` interface is the stated invariant that "survives at higher layers"):

- **New file `recursivene/encoder.py`** (or an L1 module folder) exposing `LatentEncoder` with `encode(x)->z`, `ema_step()`, and a `predict_target(z_ctx, cond)->z_hat`. Keep `model.py`/`agent.py` signatures frozen.
- In `RegionLearner.observe`, route through the encoder: the regressor predicts `z_t = encode(x_t)` from `z_c = encode(x_c)`; `err` becomes the *latent* smooth-L1, fed into the SAME `err_hist`/LP machinery unchanged. Downstream (`choose`, `objective.competence`) needs no edit because it only reads the bookkeeping interface.
- **Retire the `noise_floor` scalar** in favor of the learned invariant: once the encoder absorbs aleatoric content, noisy arms produce low latent error naturally, so the `errs < noise_floor` gate in `choose("lp")` can be loosened/removed and replaced by "latent LP trend > significance floor" — exactly the existing `learning_progress` test, now noise-robust by construction. This is precisely the LPM 2025 result (arXiv:2509.25438): reward model *improvement*, not error, and measure it where noise has been encoded out.

Net: RecursiveNe's noise rejection moves from a tuned constant (a confound the bench specialist could rightly flag) to a learned, falsifiable representation — and it generalizes the noisy-TV defense beyond the toy variance gap.

## Pitfalls (top 2)
1. **Silent collapse masquerading as success.** EMA+stop-grad WITHOUT the variance term can collapse to a constant encoder; latent error goes to ~0 for everything, including genuinely learnable arms, and the agent thinks it has mastered the world. ALWAYS log `min_j std(Z_:,j)` and assert it stays >= ~gamma; treat a drop as a hard failure, not a hyperparameter to tune away. The 20-line test's second falsifier exists for exactly this.
2. **Encoding away the signal you wanted (representational shortcut / over-rejection).** A too-aggressive encoder (or `lambda` too low relative to `mu`) can drop *reducible* structure along with the noise, so a learnable hard arm also maps to a near-constant latent and never earns curiosity — the noisy-TV fix silently becomes a "timid-on-hard" regression, the bug `agent.py` already fought once in observation space. Guard with the first falsifier (learnable arm's latent error must still *start high and decay*, not start low) and keep `lambda=25 >> nu=1` so prediction pressure dominates decorrelation.

## Citations
- Assran et al. 2023 — I-JEPA, arXiv:2301.08243 — predict target-block representations from context in latent space; EMA target encoder + smooth-L1; masking strategy drives semantics.
- LeCun 2022 — A Path Towards Autonomous Machine Intelligence (H-JEPA), OpenReview BZ5a1r-kVsf — hierarchical JEPA world model, intrinsic motivation, energy regularized against collapse.
- Bardes, Ponce & LeCun 2022 — VICReg, arXiv:2105.04906 — explicit variance (hinge, gamma=1), covariance (off-diag^2/d), invariance (MSE); defaults lambda=25, mu=25, nu=1.
- Grill et al. 2020 — BYOL, arXiv:2006.07733 — EMA target + stop-gradient prevents collapse without negative pairs (the asymmetry I-JEPA inherits).
- Anonymous 2025 — Beyond Noisy-TVs: Learning Progress Monitoring, arXiv:2509.25438 — reward model improvement not prediction error; latent learning-progress rejects unlearnable transitions. Direct theoretical backing for RecursiveNe's `lp` policy.
- Mavor-Parker et al. 2021 — Aleatoric Uncertainty Estimation, arXiv:2102.04399 — separating aleatoric from epistemic uncertainty as the principled noisy-TV defense.
