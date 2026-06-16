"""L3 gate: imagination (planning in a learned model) reaches goals using far fewer real-world
interactions than planning in reality, at equal competence, with a faithful learned model.
Run:  python experiments/validate_l3.py
"""

import sys
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.imagination import real_step_batch, train_model, run_episode, model_rollout_error

T, K, H = 30, 250, 15
GOALS = [np.array(g) for g in ([0.7, -0.6], [-0.5, 0.6], [0.6, 0.6])]
START = np.array([0.0, 0.0])
R = []


def check(name, ok, detail):
    R.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name:32s} {detail}")


def main():
    model, n_model = train_model(900, seed=0)
    roll = model_rollout_error(model, H=H)
    check("learned model is faithful", roll < 0.06, f"{H}-step rollout MSE {roll:.4f} (~{roll/H:.4f}/step)")

    imag, real = [], []
    for g in GOALS:
        di, _ = run_episode(START, model.predict_batch, g, T, K, H, np.random.default_rng(1), False)
        dr, _ = run_episode(START, real_step_batch, g, T, K, H, np.random.default_rng(1), True)
        imag.append(di); real.append(dr)
    ai, ar = float(np.mean(imag)), float(np.mean(real))
    check("imagination reaches the goals", ai < 0.2, f"avg final distance {ai:.3f} (< 0.2)")
    check("imagination ~= reality competence", ai <= ar * 1.5, f"imag {ai:.3f} vs real-env-MPC {ar:.3f}")

    imag_total = n_model + T * len(GOALS)
    real_total = T * (1 + K * H) * len(GOALS)
    ratio = real_total / imag_total
    check("imagination = sample-efficiency multiplier", ratio > 50,
          f"{real_total:,} real interactions in reality vs {imag_total:,} with imagination = {ratio:,.0f}x fewer")

    ok = all(R)
    print(f"\n{'='*60}\n{'PASS — L3 verified: imagination multiplies sample efficiency.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
