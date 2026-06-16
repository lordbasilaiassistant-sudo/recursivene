---
ts: 2026-06-16T14:18Z
topic: imagination-dreamer
sources:
  - "Ha & Schmidhuber 2018, World Models (arXiv:1803.10122)"
  - "Hafner et al. 2023, Mastering Diverse Domains through World Models / DreamerV3 (arXiv:2301.04104, Nature 2025)"
  - "Hafner et al. 2020, Dream to Control: Learning Behaviors by Latent Imagination / Dreamer (arXiv:1912.01603)"
  - "Oudeyer & Kaplan 2007, What is Intrinsic Motivation? A Typology of Computational Approaches"
  - "Hansen et al. 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control (arXiv:2310.16828)"
  - "Robine et al. 2025-survey thread; Discrete Codebook World Models (arXiv:2503.00653)"
---

# Model-Based Imagination: Turning One Real Step into Many Learning Updates

## TL;DR (3 sentences)
A learned latent dynamics model lets you roll the world forward *in imagination* (no environment calls), so a single real transition added to a replay buffer can drive dozens-to-hundreds of cheap gradient updates on the policy/value — this is the entire sample-efficiency win of World Models (2018) and DreamerV3 (2023). The trick only stays stable because of four guards: a SHORT rollout horizon (errors compound), a KL/free-bits floor on the latent posterior-vs-prior (keeps the prior usable for imagination without collapsing), symlog + twohot targets (scale-free regression across wildly different reward/value magnitudes), and percentile-based return normalization (one fixed hyperparameter set works across domains). For RecursiveNe the actionable move is to bolt a small *differentiable* one-step latent-dynamics head beside the existing backprop-free RFF/RLS encoder, then generate imagined LP-bearing rollouts to amortize real samples — without touching the closed-form encoder that gives us the FLOP-cheap surprise signal.

## The core idea / key equation
You separate "modeling the world" from "acting in it" and let the model fabricate experience.

World model (RSSM, DreamerV3) over latent state `s_t = (h_t, z_t)` (deterministic recurrent `h`, stochastic discrete `z`):
- Sequence/recurrent: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`
- Prior (the *imagination* generator): `ẑ_t ~ p(z_t | h_t)`
- Posterior (training only, uses the real obs): `z_t ~ q(z_t | h_t, x_t)`
- Heads: decoder `x̂_t`, reward `r̂_t`, continue `ĉ_t`, all regressed with **symlog**.

World-model loss (per step):
```
L = L_pred(decoder,reward,continue; symlog/twohot)
  + β_dyn · max(1 nat, KL[ sg(q) ‖ p ])      # train the PRIOR toward the posterior
  + β_rep · max(1 nat, KL[ q ‖ sg(p) ])      # train the POSTERIOR toward the prior
```
`sg` = stop-gradient; `β_dyn=0.5, β_rep=0.1`; the `max(1 nat, ·)` is **free bits** — below 1 nat of KL the term is switched off so the latent never collapses and the prior stays a faithful rollout engine.

Behavior learning happens ENTIRELY on imagined trajectories. From each replayed real state, unroll the prior for **H=15** steps using the actor `a_t ~ π(a|s_t)`, never querying the environment. Train a λ-return critic on symlog-twohot targets, and the actor by maximizing **percentile-normalized returns**:
```
Â_t = (R^λ_t − offset) / max(1, Per95(R) − Per5(R))
```
The denominator (5th–95th percentile spread of returns in the batch, floored at 1) is the single normalization that lets one hyperparameter set span sparse and dense rewards. The amortization: `(#imagined updates per real step) = train_ratio × H`, often 100s of policy updates per real transition.

Symlog (used on every continuous target): `symlog(x) = sign(x)·ln(1+|x|)`, inverse `symexp(x)=sign(x)·(exp(|x|)−1)`. Continuous targets are predicted as a **twohot** distribution over fixed exponentially-spaced bins (classification-as-regression) for robust, scale-free fitting.

Lineage: Ha & Schmidhuber 2018 first trained a controller *entirely inside* an MDN-RNN "dream" (V=VAE encoder, M=MDN-RNN dynamics, C=tiny controller) and transferred it back. DreamerV3 industrialized this into a single robustly-tuned recipe. TD-MPC2 (2024) is the contrasting design point: it skips reconstruction, learns a *reward/value-aligned* latent and does short-horizon MPC instead of long imagined policy rollouts — relevant because our world is reward-free/curiosity-driven.

## Minimal numpy-implementable recipe (concrete, <40 lines described)
Goal: a tiny *differentiable* latent-dynamics head usable for imagination, deliberately kept separate from the RFF/RLS encoder so the backprop-free surprise path is untouched.

1. **Latent (z, 8-d).** Reuse the existing RFF map as a fixed encoder: `z = phi(x)` projected to 8 dims by a fixed random matrix (no training of the encoder — preserves A2 "backprop-free encoder").
2. **Differentiable dynamics head `g_θ`.** A single hidden-layer MLP (8→16→8) with weights `θ` as numpy arrays; predicts next latent: `ẑ' = g_θ([z, a_onehot])`. Plus a linear surprise/return head `r̂ = u·ẑ` (here "reward" = predicted *learning-progress proxy*, see next section). ~12 params-blocks, all numpy.
3. **Manual backprop.** Forward stores activations; backward is two matmuls + a tanh' — ~10 lines, no autograd needed (this is the only differentiable piece in the system).
4. **Real step.** Pick activity `r` via existing `choose(...)`; observe `(x,y)`; push `(z, a, z', surprise)` to a small deque buffer (cap 512).
5. **Imagination loop (the payoff).** For `train_ratio` (e.g. 8) iterations: sample a buffer state `z0`; roll `g_θ` forward for `H` steps (start `H=3`, the SHORT-horizon guard) producing imagined `(z_t, r̂_t)`; compute a λ-return; one SGD step on `θ` to fit symlog'd targets. => `train_ratio × H = 24` updates per real step.
6. **Guards, ported to our scale:** (a) horizon `H=3` and grow only if 1-step imagined error < real error; (b) **free-bits analog**: if the dynamics head's 1-step MSE is below a floor `τ`, freeze its update (don't overfit a mastered transition — mirrors KL free bits and our `tau_master`); (c) symlog all targets; (d) normalize advantages by `max(1, Per95−Per5)` over the buffer.
7. **Validation test (20 lines):** on the sine ladder, does adding imagination reach competence in FEWER REAL `world.step` calls than the current real-only loop, at equal FLOPs counted by the meter? Report real-steps-to-competence with/without imagination, fixed seed. Falsifier: imagination gives NO real-sample reduction, or its compounding error degrades held-out competence.

## How this informs RecursiveNe's next layer (specific)
RecursiveNe today is *prediction-only and real-sample-only*: `RegionLearner.observe()` (recursivene/agent.py:66) consumes exactly one real `world.step` per model update, and `choose()` (agent.py:134) allocates those real samples by learning progress. There is no imagination — every learning update costs a real environment interaction. That is precisely the inefficiency Dreamer removes.

Concrete upgrade path, smallest first:
1. **New file `recursivene/dynamics.py`** holding `LatentDynamicsHead` (the §recipe MLP). It sits BESIDE `RFFOnlineRegressor` (model.py) — the encoder/surprise path stays closed-form and backprop-free (protects invariant A2); only this head is differentiable. It must expose the same `flops`, `n_params()`, `ram_floats()` meters so cost-for-competence stays measured.
2. **Upgrade the curiosity signal, not just the predictor.** The head's predicted "reward" should be the **imagined learning-progress** at a latent state — i.e. give `RegionLearner.learning_progress()` (agent.py:86) an imagined counterpart `imagined_lp(z)` so `choose()` can rank activities by *anticipated* LP from rollouts instead of only the realized error-slope it sees now. This is the single highest-leverage change: it lets the LP engine look ahead instead of purely backward.
3. **Amortize in the harness loop.** Wherever the experiment driver calls `observe` once per real step (experiments/run_*.py), wrap it: 1 real `observe` -> push to buffer -> `K_imag` imagined updates to the dynamics head. Keep the held-out invariant measurement on REAL data only (world.py inner/held-out split, D3) so imagination cannot game competence.
4. **Latent regions = the "L1 encoder" hook.** The module docstring already anticipates "activity -> learned latent region" (agent.py:35). A trained latent `z` from this head is exactly that L1 encoder: discover regions as clusters in imagined latent space rather than the hand-given activity index `r`.

Net: this is the mechanism that converts RecursiveNe from "one update per real sample" to "many updates per real sample" while keeping the FLOP meter honest and the backprop-free surprise core intact.

## Pitfalls (top 2)
1. **Compounding model error over horizon — the dominant failure.** Imagined rollouts train the policy on the model's *hallucinations*; small per-step latent errors compound, so a long horizon optimizes against fiction (the classic MBRL "exploit the model" trap). Mitigation is non-negotiable: keep `H` short (start 3), grow it only when measured 1-step imagined error stays below real error, and ALWAYS score competence on real held-out data. For us the extra hazard is the **noisy-TV inside imagination**: a dynamics head can hallucinate spurious learning-progress on the noise arms, re-leaking the exact failure mode agent.py was built to kill — so the imagined-LP signal needs the same noise-floor/significance gate as the real one.
2. **Latent collapse / free-bits mis-set.** Without the `max(1 nat, KL)` floor the posterior collapses to a point and the prior becomes a useless rollout engine (you imagine garbage); with it set too high the model underfits. In our minimal port the analog is the freeze-below-`τ` guard plus symlog targets — skip symlog and a single large-magnitude surprise (a high-frequency sine or a noise spike) will dominate the dynamics-head gradient and destabilize the whole imagined loop.

## Citations
- Ha, D. & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122. (V=VAE, M=MDN-RNN, C controller; train controller entirely inside the RNN dream, transfer back.)
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023). *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104; Nature (2025). (RSSM, symlog, twohot, free bits = 1 nat, KL balancing β_dyn=0.5/β_rep=0.1, H=15, percentile-return normalization.)
- Hafner, D. et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer). arXiv:1912.01603. (Differentiable latent imagination + λ-returns; the imagination-as-policy-training mechanism.)
- Oudeyer, P-Y. & Kaplan, F. (2007). *What is Intrinsic Motivation? A Typology of Computational Approaches.* (Learning-progress intrinsic motivation — the signal we want to imagine forward.)
- Hansen, N., Su, H., Wang, X. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* arXiv:2310.16828. (Reconstruction-free, reward/value-aligned latent + short-horizon planning — the contrasting design point for a reward-free curiosity world.)
- *Discrete Codebook World Models for Continuous Control* (2025). arXiv:2503.00653. (Recent line showing discrete-latent world models remain the strong sample-efficient default.)
