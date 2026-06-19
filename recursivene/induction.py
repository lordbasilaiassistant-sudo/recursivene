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


def _greedy(vals, target, core, edge, allowed, cx, max_terms, seed_active=()):
    """Greedily add atoms (from `allowed`) that most improve EDGE-validation fit of `target`: fit coefs on
    the interior (core), score residual toward the boundary (edge). Occam penalty breaks near-ties toward
    simpler atoms. Returns the selected atom indices."""
    yscale = float(np.var(target[edge])) + 1e-12
    active = list(seed_active)

    def edge_mse(idxs):
        if not idxs:
            return float(np.mean(target[edge] ** 2))
        Ac = np.stack([vals[j][core] for j in idxs], axis=1)
        c, *_ = np.linalg.lstsq(Ac, target[core], rcond=None)
        Ae = np.stack([vals[j][edge] for j in idxs], axis=1)
        return float(np.mean((Ae @ c - target[edge]) ** 2))

    cur = edge_mse(active)
    for _ in range(max_terms):
        best = None
        for j in allowed:
            if j in active:
                continue
            em = edge_mse(active + [j])
            score = em + 0.01 * cx[j] * yscale
            if best is None or score < best[0]:
                best = (score, em, j)
        if best is None or best[1] >= cur * 0.98:
            break
        active.append(best[2]); cur = best[1]
    return active


def induce(X, Y, max_terms=8, fmax=15.0):
    """Induce the generating law of Y=f(X) over a grammar, by STRUCTURED edge-validated search:
      1) DETREND — find the smooth extending part (polynomial / exponential / const) that best fits
         toward the boundary; subtract it. (A trend masked under oscillation is captured here, not
         mimicked by a sinusoid.)
      2) OSCILLATION + LOCAL — on the residual, find sinusoids (periodic) and gaussians (localized).
      3) JOINT REFIT on all data.
    Selecting by edge-validation (interior->boundary) keeps atoms that EXTEND, and separating trend from
    oscillation fixes the superposition failure of plain greedy. Every atom is closed-form, so the
    induced law EXTRAPOLATES. Returns (law_fn, terms); law_fn(x) evaluates the program at ANY x."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); N = len(X)
    atoms = _dictionary(fmax)
    vals = [a[1](X) for a in atoms]; cx = [a[2] for a in atoms]
    order = np.argsort(np.abs(X)); ncore = max(8, int(0.70 * N))
    core, edge = order[:ncore], order[ncore:]
    trend_idx = [i for i, a in enumerate(atoms) if a[0] == "1" or a[0].startswith("x^") or a[0].startswith("e^")]
    osc_idx = [i for i, a in enumerate(atoms) if a[0].startswith(("sin", "cos", "g("))]

    # 1) detrend: the smooth extending part (poly/exp/const), then 2) oscillation+localized on residual.
    # (Separating trend from oscillation lets a trend be CAPTURED rather than mimicked by a sinusoid; it
    # cracks polynomial/exponential/cubic families. The remaining hard case — a mid-frequency sinusoid
    # superposed on a gentle linear trend — resists greedy grid search and is named as the frontier.)
    trend = _greedy(vals, Y, core, edge, allowed=trend_idx, cx=cx, max_terms=4, seed_active=[0])
    At = np.stack([vals[j] for j in trend], axis=1)
    ct, *_ = np.linalg.lstsq(At[core], Y[core], rcond=None)
    osc = _greedy(vals, Y - At @ ct, core, edge, allowed=osc_idx, cx=cx, max_terms=max_terms)

    active = list(dict.fromkeys(trend + osc))
    Aall = np.stack([vals[j] for j in active], axis=1)        # 3) joint refit on ALL data
    coef, *_ = np.linalg.lstsq(Aall, Y, rcond=None)

    def law(x):
        x = np.asarray(x, float)
        return sum(coef[i] * atoms[active[i]][1](x) for i in range(len(active)))
    terms = [(atoms[active[i]][0], float(coef[i])) for i in range(len(active))]
    return law, terms
