"""L1 — a learned representation whose per-step cost does not grow with complexity.

capstone.py isolated the blocker to an unbounded open-ended race-to-0: FIXED random Fourier
features must densely cover all of frequency space, so as the signal's frequencies climb they
get more expensive no matter what past work is banked. The fix is a representation that adapts
to the data: a SHARED spectral encoder that DISCOVERS the frequencies actually present (from a
rolling buffer of its own experience) and places its features exactly there. Once a frequency is
discovered it is represented at O(1) cost forever, regardless of how high it is — so a target
that reuses discovered structure is cheap even at high complexity. An RFF fallback covers
not-yet-discovered structure (expensive once), so nothing is unrepresentable; the first time a
new frequency appears it costs, then it is banked into the encoder and amortized.

This is the JEPA move in miniature: a shared learned encoder + a cheap per-task readout, with the
encoder trained self-supervised on the agent's own stream.
"""

import numpy as np


class SpectralEncoder:
    def __init__(self, n_freqs=20, fmax=40.0, rff_dim=64, gamma=8.0, buffer=800, min_sep=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_freqs
        self.min_sep = min_sep
        self.freqs = rng.uniform(1.0, 6.0, n_freqs)        # start LOW / under-provisioned
        self.fmax = float(fmax)
        self.candidates = np.linspace(1.0, fmax, 240)
        self.bx, self.by = [], []
        self.buffer = buffer
        # fixed RFF fallback so not-yet-discovered structure is still representable
        self.Wr = rng.normal(0.0, gamma, rff_dim)
        self.br = rng.uniform(0.0, 2 * np.pi, rff_dim)
        self.sr = np.sqrt(2.0 / rff_dim)
        self.rff_dim = rff_dim

    def observe(self, x, y):
        self.bx.append(x); self.by.append(y)
        if len(self.bx) > self.buffer:
            self.bx.pop(0); self.by.pop(0)

    def discover(self):
        """Re-place the encoder's features at the dominant frequencies in the buffer."""
        if len(self.bx) < 60:
            return
        X = np.asarray(self.bx); Y = np.asarray(self.by); Y = Y - Y.mean()
        S = np.sin(np.outer(self.candidates, X)) @ Y
        C = np.cos(np.outer(self.candidates, X)) @ Y
        energy = S * S + C * C
        # greedy peak-pick with non-max suppression
        picked, used = [], np.zeros(len(self.candidates), bool)
        order = np.argsort(energy)[::-1]
        for i in order:
            if used[i]:
                continue
            picked.append(self.candidates[i])
            used |= np.abs(self.candidates - self.candidates[i]) < self.min_sep
            if len(picked) >= self.n:
                break
        if picked:
            self.freqs = np.array(picked)
            self.n = len(self.freqs)        # keep the count in sync with the actual array

    def phi(self, x):
        disc = np.concatenate([np.sin(self.freqs * x), np.cos(self.freqs * x)])
        rff = self.sr * np.cos(self.Wr * x + self.br)
        return np.concatenate([disc, rff])

    def dim(self):
        return 2 * len(self.freqs) + self.rff_dim     # authoritative: derived from the real array

    def grow(self, add=8, fmax_mul=1.2, fmax_cap=80.0, seed=0):
        """Widen the representation: more feature slots + higher reach, so the encoder can hold
        and discover structure at complexity it could not before (the garden's capacity-growth,
        at the representation level)."""
        rng = np.random.default_rng(seed)
        self.n += int(add)
        self.fmax = min(fmax_cap, self.fmax * fmax_mul)
        self.candidates = np.linspace(1.0, self.fmax, 240)
        self.freqs = np.concatenate([self.freqs, rng.uniform(1.0, self.fmax, int(add))])
        return self.dim()

    def state(self):
        return {"freqs": [float(f) for f in self.freqs], "n": self.n, "fmax": float(self.fmax)}

    def load_state(self, s):
        self.freqs = np.array(s["freqs"]); self.n = s["n"]; self.fmax = s["fmax"]
        self.candidates = np.linspace(1.0, self.fmax, 240)
