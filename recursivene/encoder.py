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

import math

import numpy as np


class SpectralEncoder:
    def __init__(self, n_freqs=20, fmax=40.0, rff_dim=64, gamma=8.0, buffer=800, min_sep=1.0,
                 bic_mult=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_freqs
        self.min_sep = min_sep
        self.bic_mult = float(bic_mult)   # parsimony strength: higher prunes harder (extrapolation),
        #                                   lower keeps more components (rich-world capacity)
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
            # REFINE each grid-picked frequency by a fine local search (continuous structure ID, not
            # just the coarse candidate grid). A frequency that is even slightly off drifts in phase as
            # x grows, which is exactly what kills EXTRAPOLATION (KNOWN #24); sharpening the discovered
            # law is what lets the encoder EXTEND it beyond the data, not just fit within it.
            step = (self.fmax - 1.0) / (len(self.candidates) - 1)
            ones = np.ones_like(X)
            refined = []
            for f0 in picked:
                # Refine by the LEAST-SQUARES RESIDUAL of a [sin,cos,1] fit (orthogonality-corrected),
                # not the raw periodogram dot-product which is biased on a short window. This is the
                # system-ID objective that pins the true frequency, so the discovered law EXTENDS beyond
                # the data (the extrapolation lever, KNOWN #24). Wide window to beat grid-quantization.
                grid = np.linspace(max(0.2, f0 - 2.5 * step), f0 + 2.5 * step, 121)
                best_f, best_r = f0, np.inf
                for f in grid:
                    A = np.stack([np.sin(f * X), np.cos(f * X), ones], axis=1)
                    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                    r = float(np.mean((A @ coef - Y) ** 2))
                    if r < best_r:
                        best_r, best_f = r, float(f)
                refined.append(best_f)
            # Keep the FULL refined set as the representation: over-capacity helps in-support fitting of
            # rich multi-frequency worlds (the L1 cost-to-complexity flattening relies on it). PARSIMONY
            # is a SEPARATE objective (Occam/MDL), exposed by law() below for EXTRAPOLATION — because
            # representation (max capacity) and law-extraction (min description) genuinely trade off, so
            # the entity keeps both rather than collapsing them into one phi.
            self.freqs = np.array(refined)
            self.n = len(self.freqs)        # keep the count in sync with the actual array (derived)

    def law(self):
        """The parsimonious LAW: the FEW frequencies (a sparse subset of the discovered ones) that
        genuinely explain the buffer, by BIC forward selection — each added (sin,cos) pair must lower
        N*ln(RSS) by more than its description cost ~bic_mult*ln(N). This is the MDL/Occam reading of the
        data: the shortest law that fits, which is what EXTENDS beyond the data (KNOWN #24 — extrapolation
        is structure discovery, and needs sharp AND SPARSE). Representation (phi) stays rich; the law is
        what you extrapolate with. Returns an array of frequencies (>=1)."""
        if len(self.bx) < 60 or len(self.freqs) == 0:
            return np.array(self.freqs)
        X = np.asarray(self.bx); Y = np.asarray(self.by); Y = Y - Y.mean(); ones = np.ones_like(X)
        N = len(Y)
        def rss(freqs):
            cols = []
            for f in freqs:
                cols += [np.sin(f * X), np.cos(f * X)]
            A = np.stack(cols + [ones], axis=1)
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            return float(np.mean((A @ coef - Y) ** 2))
        cand = list(dict.fromkeys(float(f) for f in self.freqs))
        active, cur = [], float(np.mean(Y ** 2)) + 1e-12
        pen = self.bic_mult * math.log(max(N, 2))
        while cand and len(active) < len(self.freqs):
            r, f = min(((rss(active + [f]), f) for f in cand), key=lambda t: t[0])
            if N * math.log(max(r, 1e-12)) + pen < N * math.log(cur):
                active.append(f); cand.remove(f); cur = r
            else:
                break
        return np.array(active if active else [float(self.freqs[0])])

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
