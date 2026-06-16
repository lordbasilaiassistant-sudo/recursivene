"""L3 — IMAGINATION. Learn the world's DYNAMICS, then plan on DREAMED rollouts instead of real
experience (Ha & Schmidhuber 2018 'World Models'; Hafner DreamerV3).

Everything before this made a STATIC unknown known (y = f(x)). L3 adds TIME: a world with state,
actions, and transitions s' = f(s,a). The agent learns a forward model f-hat from a handful of real
transitions, then PLANS by rolling f-hat forward (imagining), choosing actions without touching the
real world. The payoff is the sample-efficiency multiplier — and something stronger: planning needs
to try MANY futures from the SAME state, which is physically impossible in reality (you cannot rewind
the world) but trivial in a learned model. Imagination is not just cheaper than reality; it does what
reality cannot.

Task: a 2-D point-mass with a nonlinear position-dependent drift (a curl field), action = a 2-D push,
goal = reach a target. Forward model: RFF + multi-output RLS predicting the state delta. Planner:
model-predictive control (random shooting) that rolls EITHER the learned model (imagination) or the
real world.
"""

import numpy as np

DT, PUSH, DRIFT = 0.15, 1.0, 0.05
BOUND = 1.2


def real_step_batch(S, A):
    """True dynamics, batched. S,A: (K,2). Nonlinear curl drift the model must learn."""
    A = np.clip(A, -1, 1)
    drift = DRIFT * np.stack([np.sin(3 * S[:, 1]), np.cos(3 * S[:, 0])], axis=1)
    return np.clip(S + DT * PUSH * A + drift, -BOUND, BOUND)


def real_step(s, a):
    return real_step_batch(s[None], a[None])[0]


class ForwardModel:
    """Learns f-hat: (s,a) -> s' via RFF features + multi-output RLS (one shared inverse-covariance,
    a weight column per output dim). Online, closed-form — the substrate primitive, applied to dynamics."""

    def __init__(self, in_dim=4, out_dim=2, D=240, gamma=2.5, ridge=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, gamma, (D, in_dim))
        self.b = rng.uniform(0, 2 * np.pi, D)
        self.scale = np.sqrt(2.0 / D)
        self.Wt = np.zeros((D, out_dim))
        self.P = np.eye(D) / ridge
        self.D = D

    def _phi(self, Z):
        return self.scale * np.cos(Z @ self.W.T + self.b)

    def update(self, s, a, s_next):
        z = np.concatenate([s, a])
        p = self._phi(z[None])[0]
        err = (s_next - s) - p @ self.Wt
        Pp = self.P @ p
        k = Pp / (1.0 + p @ Pp)
        self.Wt = self.Wt + np.outer(k, err)
        self.P = self.P - np.outer(k, Pp)

    def predict_batch(self, S, A):
        Z = np.concatenate([S, A], axis=1)
        return S + self._phi(Z) @ self.Wt


def mpc_act(s0, step_batch, goal, K, H, rng):
    """Random-shooting MPC: sample K action sequences of horizon H, roll `step_batch` forward, pick the
    first action of the best sequence. Returns (action, real_world_queries_used_for_planning)."""
    A = rng.uniform(-1, 1, (K, H, 2))
    S = np.tile(s0, (K, 1)).astype(float)
    total = np.zeros(K)
    for h in range(H):
        S = step_batch(S, A[:, h, :])
        total += -np.linalg.norm(S - goal, axis=1)
    return A[int(np.argmax(total)), 0, :], K * H


def run_episode(s0, planner_step_batch, goal, T, K, H, rng, planning_is_real):
    """Run T real steps; at each, plan with planner_step_batch (learned model or real world). Returns
    (final_distance, real_world_interactions). Execution always costs 1 real step; planning costs
    K*H real queries IF the planner rolls the real world, 0 if it imagines."""
    s = np.array(s0, float)
    real_interactions = 0
    for _ in range(T):
        a, planning_real = mpc_act(s, planner_step_batch, goal, K, H, rng)
        s = real_step(s, a)
        real_interactions += 1 + (planning_real if planning_is_real else 0)
    return float(np.linalg.norm(s - goal)), real_interactions


def train_model(n_transitions, seed=0):
    """Learn f-hat from random real transitions. Returns (model, n_transitions)."""
    rng = np.random.default_rng(seed)
    m = ForwardModel(seed=seed)
    for _ in range(n_transitions):
        s = rng.uniform(-BOUND, BOUND, 2)
        a = rng.uniform(-1, 1, 2)
        m.update(s, a, real_step(s, a))
    return m, n_transitions


def model_rollout_error(model, H=15, n=200, seed=1):
    """H-step imagined-vs-real rollout MSE: is the dream faithful or hallucinated?"""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        s_real = rng.uniform(-0.9, 0.9, 2)
        s_imag = s_real.copy()
        acts = rng.uniform(-1, 1, (H, 2))
        for h in range(H):
            s_real = real_step(s_real, acts[h])
            s_imag = model.predict_batch(s_imag[None], acts[h][None])[0]
        errs.append(np.sum((s_real - s_imag) ** 2))
    return float(np.mean(errs))
