"""The world the learner lives in: a playground of activities of graded learnability.

Each learnable activity is a band-limited function  y = amp * sin(w*x + phase)  on
x in [-1, 1]. The frequency w is the difficulty knob: w=0 is a constant (mastered in
one sample); high w needs many samples AND enough model capacity to fit. One activity
is pure noise (y ~ N(0, s)), independent of x — the unlearnable "noisy TV".

Two design properties matter, and both answer the adversarial critique of the v0 seed:

  1. HARD activities exist (high w). Competence is defined over ALL learnable activities,
     so reaching it REQUIRES concentrating samples on the hard ones. That makes
     "samples-to-competence" a real allocation problem (H1), not a formality that every
     policy satisfies instantly.

  2. INNER vs HELD-OUT activity sets. The improvement loop optimizes against the inner
     set; the untouchable invariant and the reward are measured on a DIFFERENT held-out
     set (different frequencies + phases). Optimizing the inner score cannot trivially
     game the held-out one — the anti-gaming anchor (D3).

The frequency is a continuous parameter, so a frontier-following generator can push it
upward as the learner masters the current band (the open-ended hook; see harness).
"""

import numpy as np


class Activity:
    """One activity: either a band-limited sine (learnable) or pure noise."""

    __slots__ = ("kind", "w", "phase", "amp", "noise_std")

    def __init__(self, kind, w=0.0, phase=0.0, amp=1.0, noise_std=1.0):
        self.kind = kind          # "sine" | "noise"
        self.w = float(w)
        self.phase = float(phase)
        self.amp = float(amp)
        self.noise_std = float(noise_std)

    @property
    def learnable(self):
        return self.kind != "noise"

    def truth(self, x):
        """Noise-free ground truth (used only for competence scoring, never shown)."""
        if self.kind == "noise":
            return 0.0
        return self.amp * np.sin(self.w * x + self.phase)


class World:
    """A list of activities + an i.i.d. observation-noise level on the learnable ones."""

    def __init__(self, activities, obs_noise=0.02, seed=0):
        self.activities = activities
        self.obs_noise = float(obs_noise)
        self.rng = np.random.default_rng(seed)
        self.K = len(activities)
        self.names = [self._name(i, a) for i, a in enumerate(activities)]
        self.learnable = np.array([a.learnable for a in activities])
        self.noise_indices = [i for i, a in enumerate(activities) if not a.learnable]

    @staticmethod
    def _name(i, a):
        if a.kind == "noise":
            return "noise"
        if a.w == 0.0:
            return "const"
        return f"sine{a.w:g}"

    def truth(self, r, x):
        return self.activities[r].truth(x)

    def step(self, r, x):
        """Sample one observation from activity r at knob x."""
        a = self.activities[r]
        if a.kind == "noise":
            return float(a.noise_std * self.rng.standard_normal())
        return float(a.truth(x) + self.obs_noise * self.rng.standard_normal())

    def sample_x(self):
        return float(self.rng.uniform(-1.0, 1.0))


# --- canonical activity ladders ----------------------------------------------------
# Inner set: the frequency ladder the improvement loop trains/optimizes against.
_INNER_W = (0.0, 1.5, 3.0, 6.0, 9.0, 12.0)
# Held-out set: DIFFERENT frequencies + phase offsets. Used by the untouchable
# invariant and the reward, so inner-loop optimization cannot game it (D3).
_HELDOUT_W = (0.75, 2.25, 4.5, 7.5, 10.5)


def _phases(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=n)


def make_world(which="inner", seed=0, n_noise=12, obs_noise=0.02, extra_w=()):
    """Build the inner or held-out world. `extra_w` lets a frontier-following generator
    append harder activities at runtime. `n_noise` unlearnable distractors make the world
    mostly-noise — the realistic case where attention/allocation actually matters: a
    uniform sampler wastes most of its budget, a learning-progress sampler does not."""
    ladder = (_INNER_W if which == "inner" else _HELDOUT_W) + tuple(extra_w)
    phase_seed = seed + (0 if which == "inner" else 777)
    ph = _phases(len(ladder), phase_seed)
    acts = [Activity("sine", w=w, phase=ph[i], amp=1.0) for i, w in enumerate(ladder)]
    for _ in range(n_noise):
        acts.append(Activity("noise", noise_std=1.0))
    return World(acts, obs_noise=obs_noise, seed=seed + 1)
