"""The seed loop: perceive -> learn from surprise -> estimate learning progress -> act.

This is EDITABLE machinery (the proposer may rewrite it). It binds the world, the
world-model primitive, and the curiosity engine into one online loop with no training
phase, no dataset, no labels — the learner generates its own data by choosing what to
do next. What it computes toward (competence, cost) is defined in the protected
objective.py and is NOT editable.
"""

import numpy as np

from .world import make_world
from .agent import RegionLearner, choose
from .objective import competence, EVAL_EVERY

# The learner's entire hyperparameter surface — the proto-genome the harness mutates.
DEFAULT_CONFIG = {
    "policy": "lp",
    "n_features": 96,   # D — model's whole parameter budget; the 'race to 0' knob
    "gamma": 8.0,       # RFF bandwidth (must be high enough to fit the hard activities)
    "ridge": 1.0,       # RLS prior precision
    "forget": 1.0,      # RLS forgetting factor
    "hist": 24,         # learning-progress window (max)
    "min_lp": 8,        # samples before LP is estimable (caps warmup waste)
    "epsilon": 0.1,     # exploration rate
    "ensemble": 1,      # heads per activity (>1 enables disagreement policy)
    "lp_floor": 1.0,    # noise-floor multiplier for learning progress
    "tau_master": 0.04, # per-arm mastery threshold (stop sampling below this)
    "noise_floor": 0.7, # error above this is treated as unlearnable (noise rejection)
}


def _make_learner(world, cfg, seed):
    return RegionLearner(
        world.K,
        n_features=cfg["n_features"], gamma=cfg["gamma"], ridge=cfg["ridge"],
        forget=cfg["forget"], hist=cfg["hist"], ensemble=cfg.get("ensemble", 1),
        lp_floor=cfg.get("lp_floor", 1.0), min_lp=cfg.get("min_lp", 8),
        tau_master=cfg.get("tau_master", 0.04), noise_floor=cfg.get("noise_floor", 0.7),
        seed=seed,
    )


def run(config=None, world=None, steps=3000, seed=0, eval_every=EVAL_EVERY):
    """Run the seed loop on `world`. Returns (world, learner, log)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if world is None:
        world = make_world("inner", seed=seed)
    rng = np.random.default_rng(seed + 1)
    learner = _make_learner(world, cfg, seed)

    log = {"steps": [], "competence": [], "flops": [],
           "region_seq": np.empty(steps, dtype=int)}
    for t in range(steps):
        r = choose(cfg["policy"], learner, world, rng, epsilon=cfg["epsilon"])
        x = world.sample_x()
        y = world.step(r, x)
        learner.observe(r, x, y)
        log["region_seq"][t] = r
        if (t + 1) % eval_every == 0:
            log["steps"].append(t + 1)
            log["competence"].append(competence(world, learner))
            log["flops"].append(learner.total_flops())

    log["visits"] = learner.visits.copy()
    log["noise_indices"] = world.noise_indices
    log["names"] = world.names
    log["n_params"] = learner.n_params()
    log["ram_floats"] = learner.ram_floats()
    return world, learner, log
