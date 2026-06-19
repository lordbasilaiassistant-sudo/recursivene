"""L5 — GROUNDED SYMBOLS. Language as another observation channel, grounded on the world model.

The roadmap's principle: a symbol is only meaningful as a compressed pointer into a model of what it
refers to (words last, grounded on physics — not floating like an LLM). Here a symbol is a discrete
token; its MEANING is a world-STATE the substrate perceives from raw samples. We learn a BIDIRECTIONAL
map between symbol space and the world-model latent, trained ONLY on (scene, symbol) pairs:

  production    : perceive a scene -> NAME it (emit the right symbol(s))     latent -> symbol
  comprehension : be told a symbol -> IMAGINE it (reconstruct the scene)     symbol -> latent -> scene

Grounding, not a lookup table: the symbol predicts the world-model LATENT (RFF coefficients fit from
samples), and the latent reconstructs the actual scene on a grid. Because scenes are COMPOSITIONAL (sums
of hidden primitives) and symbols are multi-hot over primitives, the test of real understanding is
ZERO-SHOT COMPOSITION: name / imagine NOVEL symbol combinations never seen paired. That is systematic
generalization — the hallmark of intelligence (and a known weakness of pattern-matchers).

Benchmarks: samples-to-first-communication (how few pairs until it round-trips held-out concepts), and
zero-shot compositional accuracy. numpy only; runs on a laptop in seconds.
"""

import numpy as np

GRID = np.linspace(-1.0, 1.0, 81)          # the "retina" a scene is read out on


class RFF:
    """Fixed random-feature perceiver: x -> R^D. The substrate's sensory front end."""
    def __init__(self, D=128, gamma=12.0, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, gamma, D); self.b = r.uniform(0, 2 * np.pi, D)
        self.s = np.sqrt(2.0 / D); self.D = D

    def __call__(self, x):
        return self.s * np.cos(self.W * x + self.b)

    def phi_grid(self):
        return np.stack([self(x) for x in GRID])    # (|GRID|, D)


def perceive(scene_fn, rff, n=120, ridge=1.0, noise=0.02, seed=0):
    """Read a scene from n noisy samples -> latent theta in R^D (the world-model's representation of it).
    This is the only access the namer/imaginer has to meaning — symbols are grounded on THIS, not truth."""
    rng = np.random.default_rng(seed)
    P = np.eye(rff.D) / ridge; w = np.zeros(rff.D)
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0); f = rff(x); y = scene_fn(x) + noise * rng.standard_normal()
        Pp = P @ f; k = Pp / (1.0 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
    return w


def reconstruct(theta, phi_grid):
    return phi_grid @ theta                          # scene values on GRID from a latent


class LanguageGround:
    """Bidirectional symbol<->latent maps, refit in closed form from accumulated (scene, symbol) pairs.

    V symbols, D-dim latent. Wp: latent->symbol logits (production). Wc: symbol->latent (comprehension).
    Ridge-regularized least squares — cheap, online-friendly, and (crucially) it can GENERALIZE to
    unseen symbol combinations because the maps are linear over the primitive basis, so a novel multi-hot
    decodes to the sum of its parts' latents (compositionality falls out, it isn't memorized)."""

    def __init__(self, V, D, ridge=1.0):
        self.V, self.D, self.ridge = V, D, ridge
        self.S, self.T = [], []                      # symbols (V), latents (D)
        self.Wp = np.zeros((V, D)); self.Wc = np.zeros((D, V))

    def observe(self, theta, sym):
        self.S.append(np.asarray(sym, float)); self.T.append(np.asarray(theta, float))

    def fit(self):
        if len(self.S) < 2:
            return
        S = np.asarray(self.S); T = np.asarray(self.T)
        # production: T (N,D) -> S (N,V).  Wp (V,D):  Wp^T = (T^T T + lam I)^-1 T^T S
        self.Wp = np.linalg.solve(T.T @ T + self.ridge * np.eye(self.D), T.T @ S).T
        # comprehension: S (N,V) -> T (N,D).  Wc (D,V):  Wc^T = (S^T S + lam I)^-1 S^T T
        self.Wc = np.linalg.solve(S.T @ S + self.ridge * np.eye(self.V), S.T @ T).T

    def name(self, theta, k=1):
        """Production: latent -> the k highest-scoring symbols (k = number of active primitives)."""
        logits = self.Wp @ theta
        return set(np.argsort(logits)[-k:].tolist())

    def imagine(self, sym):
        """Comprehension: symbol(s) -> predicted latent."""
        return self.Wc @ np.asarray(sym, float)
