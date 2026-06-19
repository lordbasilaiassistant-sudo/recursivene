"""PROGRAM INDUCTION — discover the generating LAW over a grammar of primitives, not just frequencies.

The g-panel's named boundary (KNOWN #24): the SpectralEncoder's law class is sinusoidal, so it cannot
discover or EXTEND non-periodic structure (a localized bump, a trend). This module widens the hypothesis
class to a small GRAMMAR of closed-form primitives — constant, polynomial, sinusoid, gaussian, exponential
— and finds the SHORTEST combination (MDL/Occam, greedy orthogonal matching pursuit + BIC) that explains
the data. Because every primitive is a closed-form law, the induced program EXTRAPOLATES beyond the data.

This is the Universal-AI / Solomonoff move that the whole project bets on: intelligence = finding the
generating program (the shortest law), not approximating the function. It is the step from "fit within
the data" toward "discover the law and extend it" — across function families, not one.

numpy only; laptop seconds.
"""

import math

import numpy as np


def _dictionary(fmax=15.0):
    """A grammar of closed-form basis atoms: (name, f(x)->value, complexity). Each EXTENDS beyond any
    data range. Complexity is an Occam/MDL cost so that, when two atoms fit comparably, the SIMPLER law
    wins (a linear trend over a high-frequency sinusoid that merely mimics it in-window)."""
    atoms = [("1", lambda x: np.ones_like(np.asarray(x, float)), 0.0)]
    for k in (1, 2, 3, 4):                                   # polynomial trend
        atoms.append((f"x^{k}", (lambda k: lambda x: np.asarray(x, float) ** k)(k), float(k)))
    for w in np.linspace(0.5, fmax, 80):                     # sinusoids (periodic) — pricier at high freq
        atoms.append((f"sin{w:.2f}", (lambda w: lambda x: np.sin(w * np.asarray(x, float)))(w), 1.5 + 0.15 * w))
        atoms.append((f"cos{w:.2f}", (lambda w: lambda x: np.cos(w * np.asarray(x, float)))(w), 1.5 + 0.15 * w))
    for mu in np.linspace(-1.0, 1.0, 17):                    # gaussian bumps (localized) — pricier when narrow
        for sg in (0.10, 0.18, 0.30, 0.5):
            atoms.append((f"g({mu:.2f},{sg:.2f})",
                          (lambda mu, sg: lambda x: np.exp(-((np.asarray(x, float) - mu) ** 2) / (2 * sg * sg)))(mu, sg),
                          2.5 + 0.3 / sg))
    for a in np.linspace(-2.0, 2.0, 9):                      # exponential trend
        if abs(a) > 1e-6:
            atoms.append((f"e^{a:.2f}x", (lambda a: lambda x: np.exp(a * np.asarray(x, float)))(a), 2.0 + 0.5 * abs(a)))
    return atoms


def induce(X, Y, max_terms=8, fmax=15.0):
    """Induce the generating law of Y=f(X) as a sparse combination of grammar atoms, selected by
    EDGE-VALIDATION: fit on the interior of the data and greedily add the atom that most improves the
    fit toward the BOUNDARY. This picks programs that EXTEND (a sinusoid that fits the interior but
    diverges at the edge is rejected; an x or e^ax that extends is kept) — the right inductive bias for
    discovering a law that extrapolates, rather than the in-support-greedy OMP that grabs sinusoids to
    approximate any trend. Returns (law_fn, terms); law_fn(x) evaluates the induced program at ANY x."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); N = len(X)
    atoms = _dictionary(fmax)
    vals = [a[1](X) for a in atoms]
    cx = [a[2] for a in atoms]
    order = np.argsort(np.abs(X)); ncore = max(8, int(0.70 * N))
    core, edge = order[:ncore], order[ncore:]
    yscale = float(np.var(Y[edge])) + 1e-12

    def edge_mse(idxs):
        Ac = np.stack([vals[j][core] for j in idxs], axis=1)
        c, *_ = np.linalg.lstsq(Ac, Y[core], rcond=None)
        Ae = np.stack([vals[j][edge] for j in idxs], axis=1)
        return float(np.mean((Ae @ c - Y[edge]) ** 2))

    active = [0]                                              # const (offset) always present
    cur = edge_mse(active)
    for _ in range(max_terms):
        best = None
        for j in range(1, len(atoms)):
            if j in active:
                continue
            em = edge_mse(active + [j])
            score = em + 0.01 * cx[j] * yscale               # Occam: penalize complex atoms (MDL)
            if best is None or score < best[0]:
                best = (score, em, j)
        if best is None or best[1] >= cur * 0.98:            # chosen atom must improve edge fit by >2%
            break
        active.append(best[2]); cur = best[1]

    Aall = np.stack([vals[j] for j in active], axis=1)        # final refit on ALL data
    coef, *_ = np.linalg.lstsq(Aall, Y, rcond=None)

    def law(x):
        x = np.asarray(x, float)
        return sum(coef[i] * atoms[active[i]][1](x) for i in range(len(active)))
    terms = [(atoms[active[i]][0], float(coef[i])) for i in range(len(active))]
    return law, terms
