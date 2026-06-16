"""The configuration search space — the proto-genome the meta level mutates.

Bounds keep every mutation a valid, runnable config. The 'race to 0' lives here: the
search is free to lower n_features (model size) and tighten the loop as long as the
protected objective still certifies competence on held-out worlds."""

import numpy as np

# key -> (lo, hi, kind). kind 'int' or 'float'.
SEARCH_SPACE = {
    "n_features": (16, 256, "int"),    # D — the race-to-0 knob (model parameter budget)
    "gamma": (4.0, 12.0, "float"),     # RFF bandwidth (must stay high enough to fit hard activities)
    "ridge": (0.01, 10.0, "float"),
    "hist": (12, 96, "int"),
    "min_lp": (4, 32, "int"),
    "epsilon": (0.0, 0.3, "float"),
    "lp_floor": (0.0, 2.0, "float"),
}
# forget is intentionally NOT searched: the world is stationary, so forget=1.0 (full memory)
# is both correct and numerically stable. Letting the search lower it caused RLS covariance
# windup and divergence (configs that touched tau then blew up).
CONFIG_KEYS = list(SEARCH_SPACE.keys())


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mutate(config, rng, scale=0.4, rate=0.5):
    """Return a mutated copy. Each knob is perturbed with prob `rate` by a log-normal
    step of width `scale`. (`scale`/`rate` are the editable meta-knobs — see loop.py.)"""
    child = dict(config)
    for key, (lo, hi, kind) in SEARCH_SPACE.items():
        if rng.random() >= rate:
            continue
        cur = child.get(key, 0.5 * (lo + hi))
        val = cur * float(np.exp(rng.normal(0.0, scale)))
        val = clamp(val, lo, hi)
        child[key] = int(round(val)) if kind == "int" else float(val)
    return child


def vec(config):
    """Normalized config vector in [0,1]^d for the learned proposer's surrogate."""
    out = []
    for key, (lo, hi, _) in SEARCH_SPACE.items():
        out.append((config.get(key, 0.5 * (lo + hi)) - lo) / (hi - lo))
    return np.asarray(out, dtype=float)
