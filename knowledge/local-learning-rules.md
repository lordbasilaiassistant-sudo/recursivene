---
ts: 2026-06-16T14:18Z
topic: local-learning-rules
sources:
  - "Rao & Ballard 1999 — Predictive coding in the visual cortex (Nat. Neurosci. 2:79)"
  - "Whittington & Bogacz 2017 — An Approximation of Backprop in a Predictive Coding Network with Local Hebbian Plasticity (Neural Computation 29:1229)"
  - "Millidge, Tschantz & Buckley 2022 — Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs (Neural Computation 34:1329; arXiv:2006.04182)"
  - "Hinton 2022 — The Forward-Forward Algorithm: Some Preliminary Investigations (arXiv:2212.13345)"
  - "Scellier & Bengio 2017 — Equilibrium Propagation: Bridging Energy-Based Models and Backpropagation (Front. Comput. Neurosci. 11:24)"
  - "Gandhi et al. 2025 — Mono-Forward: Backprop-Free Local Cross-Entropy (arXiv:2501.09238)"
---

# Backprop-Free Local Learning Rules for a One-Observation-One-Update Latent Encoder

## TL;DR (3 sentences)
RecursiveNe's substrate (`model.py`) is already a backprop-free, closed-form, one-sample online learner (RFF + Recursive Least Squares), but it is a *single fixed nonlinear layer* — its features `phi(x)` never adapt to the data. The family of local rules — predictive coding (PC), Forward-Forward (FF), equilibrium propagation (EP), target prop — all let you stack and *train* multiple layers using only quantities physically available at each layer (local prediction errors, local goodness), with no global backward pass and no replay/epochs, which is exactly the A2 "least-training-intensive" constraint. The single most actionable rule is **predictive coding with the Fixed-Prediction Assumption (FPA)**: it gives an adaptive deep encoder whose per-step update is local Hebbian, costs a handful of forward relaxation steps, and provably approximates backprop gradients — making it a drop-in upgrade for the static `phi(x)` feature map feeding RLS.

## The core idea / key equation
All four methods replace "compute a global gradient by chaining the backward pass" with "each layer minimizes a *locally available* error/energy." The unifying object (Millidge 2022; Scellier & Bengio 2017 show PC, EP and contrastive Hebbian are the same rule at the infinitesimal-inference limit) is an energy of squared local prediction errors:

  E = Σ_l ½ · ‖ε_l‖²,   where  ε_l = z_l − f(W_l · z_{l-1})

- `z_l` = activity (value node) at layer l; `z_0` = input x, `z_L` clamped to target y (supervised) or free (self-supervised/generative).
- **Inference (fast, inner loop):** relax activities by gradient descent on E with errors fixed-pointing locally:
  `ε_l = z_l − f(W_l z_{l-1})`
  `dz_l ∝ −ε_l + (W_{l+1}ᵀ ε_{l+1}) ⊙ f'(·)`   — only needs the layer above/below.
- **Learning (slow, one outer step per observation):** purely Hebbian, no chain rule:
  `ΔW_l ∝ ε_l · z_{l-1}ᵀ`
- **FPA shortcut (Millidge 2022b):** initialize each `z_l` to its feedforward value so `ε_l=0` at t=0; then *one* error-propagation sweep makes `ΔW_l` equal to the backprop gradient to first order. This collapses the inner relaxation loop to ~1–5 iterations and removes most of PC's cost gap vs SGD.

Forward-Forward (Hinton 2022) is the cheapest cousin: no error feedback at all. Each layer has a scalar **goodness** `G_l = Σ z_{l,j}²`; train W_l to push `G_l` high on real ("positive") inputs and low on corrupted/mismatched ("negative") inputs via local logistic loss `σ(G_l − θ)`. Mono-Forward (2025) shows replacing FF's contrastive goodness with a *local* cross-entropy head per layer beats vanilla FF and matches backprop at ~31% of its memory.

## Minimal numpy-implementable recipe (concrete, <40 lines described)
A 2-layer PC+FPA encoder that adapts features, then hands them to the existing RLS head. Per observation `(x, y)`:

1. Params: `W1 (H×1)`, `W2 (D×H)`, activation `f=tanh`, `f'=1−tanh²`. (`D` = current RFF budget so the RLS head is unchanged.)
2. **Feedforward init (FPA):** `a1 = f(W1·x)`; `a2 = f(W2·a1)`. Set value nodes `z1=a1`, `z2=a2`.
3. **One error sweep** (top error driven by the head's own surprise, see integration below): top error `ε2 = z2 − a2_target` where `a2_target` is `z2` nudged by the head residual; `ε1 = z1 − a1` (≈0 at init), then `ε1 += (W2ᵀ ε2) ⊙ f'(W1·x)`.
4. **Local Hebbian updates:** `W2 += η · ε2 · z1ᵀ`; `W1 += η · ε1 · xᵀ`. (Optionally 2–3 relaxation iterations of step 3 before updating for stability.)
5. Output adapted feature vector `phi_adapt = z2` (length D), feed to the unchanged `RFFOnlineRegressor.update`/`predict` as the feature map.

That is ~25 numpy lines, all O(H·D) (no D² unless you keep RLS), no minibatch, no epoch, no stored history — one obs, one update. FF variant is even smaller: drop `W2ᵀε2` feedback entirely and train each W_l on `σ(Σz_l² − θ)` against one negative sample (a shuffled x), ~15 lines.

## How this informs RecursiveNe's next layer (specific)
**Target interface: the feature map `phi(x)` in `recursivene/model.py`.** Today `RFFOnlineRegressor.phi` is `scale·cos(W·x+b)` with `W,b` *frozen random* — it cannot specialize to a region's structure, which caps competence on any world whose useful basis differs from random Fourier. Wrap RLS with the PC/FPA encoder above as a new optional component **without touching the frozen `model.py` contract**: expose it as `recursivene/L1encoder.py` (or under a specialist folder) providing `encode(x)->phi_adapt` and `learn(x, residual)`, and have the agent feed `phi_adapt` where `phi(x)` is consumed.

Concrete wiring against existing contracts:
- The PC top-level error can be driven directly by the quantity RLS already returns: `err = update(x,y)`. That surprise IS `ε` at the head — so the encoder and the RLS head share one error signal computed once, preserving the "learning signal = intrinsic reward, computed once" property the `model.py` docstring and `agent.py` curiosity rely on. No second loss.
- This gives `recursivene/agent.py`'s `RegionLearner` a *learnable* representation per region, so `learning_progress`/LP becomes richer (features improve, not just the linear head), strengthening the curiosity/`choose` policy without changing `POLICIES`.
- It directly serves the RSI "race to 0" knob: an adaptive encoder should reach target competence at **smaller D** than frozen RFF (fewer random features needed once features are tuned), which the FLOP meter will register as lower cost-for-competence — the exact metric `objective.evaluate` and `harness/loop.search` optimize.

**Sample-efficiency argument vs backprop+SGD.** Backprop+SGD's sample efficiency on a stationary i.i.d. dataset comes from *epochs* — reusing each sample many times. RecursiveNe forbids replay/epochs (A2), so SGD here is one noisy gradient step per sample: high variance, slow. The local rules win in *this* regime for three reasons: (1) PC's update equals the backprop gradient to first order (Whittington & Bogacz 2017; Millidge 2022) so per-step it is no worse than online SGD, while needing no backward graph; (2) RLS on the head remains an *exact* closed-form one-shot fit (infinite effective learning rate in feature space) — strictly more sample-efficient than any SGD head; (3) keeping the head closed-form and only the encoder Hebbian means the hard-to-fit part is solved exactly and the encoder only has to track a slowly-changing basis. Net: same gradient direction as backprop, but compatible with one-pass online learning and a closed-form head, so fewer observations to a given competence than backprop+SGD under the no-replay constraint. **HYPOTHESIS — must be validated** with the 20-line test: frozen-RFF+RLS vs PC-encoder+RLS, same seed, same world, compare competence-at-fixed-D and D-at-fixed-competence on `make_world`.

## Pitfalls (top 2)
1. **The first-order gradient match is only first-order, and only under FPA.** If you run too few relaxation iterations *without* FPA, PC updates diverge from backprop and can be unstable; if you run too many, you lose the cheapness that justified it. Pin FPA + 1–3 inner steps and treat the inner-step count as a metered knob, not "more is better." (Infinite-width stability also matters — see arXiv:2411.02001 on parameterization; use small `η` and tanh, watch for activity blow-up.)
2. **Local rules optimize a local objective, not the task objective.** FF's goodness and per-layer losses can learn representations that are locally good but globally suboptimal (greedy layer-wise myopia); PC avoids this *only* when the error genuinely propagates from a task-relevant top error. So you MUST drive the encoder's top error from the RLS head's real surprise (above), not from a free-floating reconstruction loss — otherwise you get a pretty autoencoder that does not lower prediction error, the only metric `competence` rewards.

## Citations
- Rao, R.P.N. & Ballard, D.H. (1999). *Predictive coding in the visual cortex.* Nature Neuroscience 2:79–87.
- Whittington, J.C.R. & Bogacz, R. (2017). *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity.* Neural Computation 29:1229–1262.
- Millidge, B., Tschantz, A. & Buckley, C.L. (2022). *Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs.* Neural Computation 34:1329–1368. arXiv:2006.04182.
- Hinton, G. (2022). *The Forward-Forward Algorithm: Some Preliminary Investigations.* arXiv:2212.13345.
- Scellier, B. & Bengio, Y. (2017). *Equilibrium Propagation: Bridging Energy-Based Models and Backpropagation.* Frontiers in Computational Neuroscience 11:24.
- Gandhi et al. (2025). *Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors.* arXiv:2501.09238.
- (background) *Local Loss Optimization in the Infinite Width: Stable Parameterization of PCNs and Target Propagation.* arXiv:2411.02001.
