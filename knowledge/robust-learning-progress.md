---
ts: 2026-06-16T14:18Z
topic: robust-learning-progress
sources:
  - "Oudeyer, Kaplan & Hadsell 2007 — Intrinsic Motivation Systems for Autonomous Mental Development (R-IAC/competence-progress lineage)"
  - "Baranes & Oudeyer 2013 — Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration (SAGG-RIAC)"
  - "Portelas, Colas, Hofmann & Oudeyer 2020 — Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments (ALP-GMM), PMLR v100; arXiv:1910.07224"
  - "Klink et al. 2023 — Self-Paced Absolute Learning Progress as a Regularized Approach to Curriculum Learning, arXiv:2306.05769"
  - "Anonymous 2025 — Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring (LPM), arXiv:2509.25438"
  - "Schmidhuber 2010 — Formal Theory of Creativity, Fun, and Intrinsic Motivation (reward = compression/learning progress)"
---

# Robust Estimation of Learning Progress: Separating Reducible Signal from Irreducible Noise

## TL;DR (3 sentences)
The hard problem — a slow-but-learnable task is indistinguishable from a high-amplitude noisy-TV over a short window — is *not* solved by any error-based or single-window estimator, because both produce high, slowly-changing prediction error. The robust fix is to estimate learning progress as a **statistically-significant negative trend in error** (regression slope minus k standard errors of the slope), and the 2025 SOTA (LPM) sharpens this by measuring **expected error reduction across model *iterations*** — noise yields zero model improvement by construction, so it is rejected analytically rather than empirically. ALP-GMM remains the best *curriculum sampler* on top of whatever LP estimate you trust, but ALP's raw nearest-neighbor reward delta is itself noise-fragile and must be fed a denoised estimate.

## The core idea / key equation

The four estimator families, ordered by noise-robustness:

1. **Window/novelty (R-IAC raw):** `LP = mean(err)` or `Δerr`. Chases the noisy-TV forever — error stays maximal where nothing is learnable. **Do not use.**

2. **Two-EMA (SAGG-RIAC / classic ALP):** maintain fast and slow EMAs of error; `ALP = |EMA_fast − EMA_slow|`. The *absolute* value is the key insight (progress can be negative = forgetting, still informative), but `E[|EMA_fast − EMA_slow|] > 0` under pure noise — a positive bias that leaks the noisy-TV straight back in. Cheap, O(1), but biased.

3. **Regression-slope-with-significance-floor (what RecursiveNe already does, and the right default):**
   ```
   slope  = Σ(t−t̄)(e_t−ē) / Σ(t−t̄)²            # OLS slope of err vs time
   σ²     = Σ resid² / (n−2)
   se     = sqrt(σ² / Σ(t−t̄)²)                 # standard error of the slope
   LP     = −slope·n − k·se·n                    # k = lp_floor (≈1.0)
   ```
   This is the principled separation: a **learnable** task has clear negative slope with small `se` → LP > 0; a **noisy-TV** has slope≈0 with LARGE residual variance → `se` large → the `−k·se·n` floor drives LP < 0 → rejected; a **mastered** task has slope≈0 and tiny residuals → LP≈0 → harmless. The `se` term is exactly a t-test on the slope: it asks "is the downward trend bigger than the noise?" — which is the noisy-TV question stated correctly.

4. **LPM — Learning Progress Monitoring (arXiv:2509.25438, SOTA 2025):** monitor *model improvement across update steps*, not error magnitude.
   ```
   ε_t^(τ)   = log( (1/d)·‖o_{t+1} − f_θ^(τ)(o_t,a_t)‖² )   # log-MSE of CURRENT model
   r_t^i     = g_φ^(τ)(o_t,a_t) − ε_t^(τ)                    # expected-prev-error − current-error
   ```
   `g_φ` is a small auxiliary net trained (on a replay queue, d≈100–128) to predict the error the *previous* model iteration would have made. The intrinsic reward is the **drop** from previous-model error to current-model error. Under irreducible noise both models are equally bad → drop = 0 → no reward (Theorem 4.2: this expectation keeps a monotonic link to true information gain). Under learnable-but-stochastic dynamics the new model is measurably better → positive reward. This is the cleanest known answer to "slow-learnable vs high-noise."

**Curriculum layer (orthogonal to estimator choice): ALP-GMM (Portelas 2020).** Keep a history of `(task_param, ALP)`; to score a freshly-sampled task, find its nearest neighbor in param-space and take `|reward − reward_nn|`. Periodically fit a GMM over `(param ‖ ALP)` space and sample new tasks ∝ the ALP-weighted components, with ρ≈0.2 uniform exploration. Few hyperparameters, scales to high-D continuous param spaces, no MDP needed. **But its per-task ALP is a 2-point delta = maximally noise-fragile**; feed it a windowed/regressed ALP, not the raw delta.

## Minimal numpy-implementable recipe (<40 lines described)

Upgrade path with two tiers — tier A is a 3-line hardening of the *current* estimator; tier B is the LPM idea adapted to RecursiveNe's per-region online regressors.

- **Tier A (drop-in, robust default).** Keep the existing regression-slope estimator in `agent.py:learning_progress` but make the noise floor *self-calibrating* instead of a fixed `lp_floor`: estimate per-region residual std from the regression itself (already computed as `sqrt(σ²)`), and additionally subtract the **median absolute deviation** of the error window to resist outlier spikes from heavy-tailed noise:
  ```
  mad = median(|y − median(y)|) * 1.4826
  LP  = −slope·n − lp_floor·se·n          # significance test (have this)
  LP -= 0.5 * mad                          # NEW: heavy-tail spike guard
  ```
  ~5 lines. Kills the residual leak where a few large noise spikes inflate slope estimates on a short window.

- **Tier B (LPM-style, the real upgrade).** For each region keep a **frozen snapshot** of its world model from `T` steps ago (`prev_model`). LP becomes the *measured* error reduction of the live model vs. the snapshot on the SAME recent inputs:
  ```
  # on observe(r, x, y): live model updates as today; snapshot does NOT.
  e_live = (y − model[r].predict(x))**2
  e_prev = (y − prev_model[r].predict(x))**2
  lp_inst = e_prev - e_live                 # >0 iff the model genuinely improved
  err_hist[r].append(lp_inst)               # store the DROP, not the error
  LP = mean(err_hist[r])                    # noise -> e_prev≈e_live -> ~0 by construction
  every T steps: prev_model[r] = deepcopy(model[r])   # re-snapshot
  ```
  ~12 lines on top of `RFFOnlineRegressor`. This needs no significance floor and no `noise_floor`/`tau_master` hand-tuning, because noise produces ~0 by identity, not by threshold. The RFF regressors are tiny so the deepcopy snapshot is cheap.

Validation (20-line test, do this BEFORE adopting): a 3-arm bandit — arm0 = mastered (flat low error), arm1 = slow-learnable (error decays at 0.5%/step under additive N(0,σ) noise), arm2 = noisy-TV (error ~ N(0.8, 0.4), no trend). The estimator passes iff `LP(arm1) > LP(arm2)` for σ up to where the slow signal and noise amplitude are equal. The current regression estimator should already pass for moderate σ; Tier B should extend the crossover further. Report the σ at which each estimator's ranking flips — that single number is the head-to-head result.

## How this informs RecursiveNe's next layer

- **Direct upgrade target: `recursivene/agent.py`, `RegionLearner.learning_progress(self, r)` (lines 86–113).** It already implements estimator family #3 correctly — this is good and ahead of most published baselines. Do Tier A (the MAD spike-guard) immediately; it is a strict robustness win with no interface change.
- **Snapshot interface for Tier B:** add `prev_models` alongside `self.models` in `RegionLearner.__init__` (line 50) and a re-snapshot every `T` calls in `observe` (line 66). This changes what `err_hist` stores (the per-step error *drop* rather than squared error), so the `lp` policy's `recent_error`/`noise_floor`/`tau_master` logic in `choose` (lines 169–183) must be revisited — under Tier B `noise_floor` and `probe_min` largely become unnecessary, which is a simplification, not a complication. Gate this behind a config flag and A/B it against the current estimator on the existing harness seeds before flipping the default.
- **L1 latent layer:** when "activity" becomes a learned latent region (noted as the surviving invariant in `agent.py:36`), the LPM snapshot approach is the one that *transfers*, because it is defined purely in terms of "did my predictor on this region's inputs get better" — it needs no per-region noise-floor constant, which would be impossible to set per-latent. **Build the L1 region encoder against the Tier-B (snapshot-delta) interface, not the error-magnitude interface.** That is the load-bearing design decision.
- **ALP-GMM as the future continuous sampler:** RecursiveNe currently samples discrete arms (`argmax error` among eligible). When the activity space becomes continuous/parameterized, replace the argmax with an ALP-GMM sampler in `choose` — but feed it the Tier-B LP, never a raw 2-point reward delta.

## Pitfalls (top 2)

1. **The positive-bias trap (`E[|fast−slow|] > 0`).** Any estimator using an absolute difference of two noisy quantities — two-EMA ALP, and ALP-GMM's nearest-neighbor reward delta — has a *strictly positive* expectation under pure noise. On a short window this bias is large enough to make the noisy-TV the highest-LP arm. The significance floor (family #3) and the cross-iteration expectation (LPM) both exist specifically to cancel this; do not "simplify" them away. The adversarial restatement: `max(0, e_slow − e_fast)` looks like progress but leaks the TV back in.

2. **Window length is a reducible-vs-irreducible tradeoff with no free lunch.** Short window → slow-learnable trend buried in noise (false negative, the timid-on-hard bug). Long window → stale; a region that just became learnable is missed, and FLOPs/RAM grow. There is no window length that solves a high-amplitude slow-learnable task with a *trend* estimator — which is the structural reason to prefer LPM's snapshot-delta, where the signal is the model-improvement itself and window length only affects variance of the *mean*, not whether the signal is detectable at all.

## Citations
- Portelas, Colas, Hofmann & Oudeyer 2020 — ALP-GMM. arXiv:1910.07224 ; PMLR v100. Code: github.com/flowersteam/teachDeepRL
- Baranes & Oudeyer 2013 — SAGG-RIAC / absolute learning progress with goal exploration.
- Oudeyer, Kaplan & Hadsell 2007 — R-IAC, competence/learning-progress intrinsic motivation.
- Klink et al. 2023 — Self-Paced Absolute Learning Progress (regularized ALP). arXiv:2306.05769
- Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring (LPM) 2025. arXiv:2509.25438
- Schmidhuber 2010 — Formal Theory of Creativity/Fun (reward = learning progress).
