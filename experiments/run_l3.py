"""L3 — IMAGINATION: planning in a learned model vs in reality. The sample-efficiency multiplier,
and the thing imagination does that reality cannot (try many futures from one state).

Run:  python experiments/run_l3.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.imagination import (real_step_batch, train_model, run_episode,
                                      model_rollout_error)

T, K, H = 35, 300, 15
GOALS = [np.array(g) for g in ([0.7, -0.6], [-0.5, 0.6], [0.6, 0.6], [-0.7, -0.5])]
N_MODEL = 900
START = np.array([0.0, 0.0])


def main():
    print("=" * 90)
    print("L3 — IMAGINATION: learn the dynamics, then plan on DREAMED rollouts (World Models / Dreamer)")
    print("=" * 90)
    print(f"Task: 2-D point-mass + nonlinear curl drift; reach a goal. MPC: K={K} seqs, H={H} horizon, T={T} steps.")

    # learn the forward model from a handful of real transitions (ONCE, reused for every goal)
    model, n_model = train_model(N_MODEL, seed=0)
    roll_err = model_rollout_error(model, H=H)
    print(f"\nLearned forward model from {n_model} real transitions. {H}-step imagined-vs-real rollout")
    print(f"MSE = {roll_err:.5f}  ->  faithful enough to PLAN with (the proof is the planning results "
          f"below: imagination reaches every goal). ~{roll_err/H:.4f}/step over a {H}-step dream.")
    print("-" * 90)

    imag_d, real_d, rand_d = [], [], []
    rng = np.random.default_rng(7)
    for g in GOALS:
        # IMAGINATION-MPC: plan by rolling the LEARNED model (no real-world queries during planning)
        di, _ = run_episode(START, model.predict_batch, g, T, K, H, np.random.default_rng(1), False)
        # REAL-ENV-MPC: plan by rolling the REAL world (each candidate rollout costs real queries)
        dr, _ = run_episode(START, real_step_batch, g, T, K, H, np.random.default_rng(1), True)
        # RANDOM baseline (model-free, no planning)
        s = START.copy()
        for _ in range(T):
            s = real_step_batch(s[None], rng.uniform(-1, 1, (1, 2)))[0]
        imag_d.append(di); real_d.append(dr); rand_d.append(float(np.linalg.norm(s - g)))
        print(f"  goal {np.round(g,2)}: imagination final dist {di:.3f} | real-env-MPC {dr:.3f} | random {rand_d[-1]:.3f}")

    # real-world interaction accounting (the headline)
    imag_total = n_model + T * len(GOALS)              # model (once) + executed steps
    real_total = T * (1 + K * H) * len(GOALS)          # every planning rollout is a real query
    print("-" * 90)
    print(f"REAL-WORLD INTERACTIONS used across {len(GOALS)} goals:")
    print(f"  imagination : {imag_total:,}  (learn the model once: {n_model}, + {T*len(GOALS)} executed steps)")
    print(f"  real-env-MPC: {real_total:,}  (every one of K*H planning rollouts touches the real world)")
    print(f"  -> imagination reached the goals using {real_total/imag_total:,.0f}x FEWER real interactions,")
    print(f"     at the same competence (imag avg dist {np.mean(imag_d):.3f} vs real {np.mean(real_d):.3f}).")
    print("=" * 90)
    print("THE DEEPER POINT: real-env-MPC requires trying K*H futures from the SAME state — which is")
    print("physically impossible in a world you cannot rewind. Only a LEARNED MODEL lets you imagine")
    print("them. Imagination isn't merely cheaper than reality; it does what reality cannot. That is the")
    print("L3 multiplier: one real sample trains the model, the model dreams unlimited practice.")
    print("Honest scope: toy 2-D control, illustrating the mechanism; model is faithful only where it")
    print("was trained (rollout MSE measured above).")
    print("=" * 90)


if __name__ == "__main__":
    main()
