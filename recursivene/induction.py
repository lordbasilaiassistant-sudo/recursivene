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


def extract_laws(X, Y, Kmax=4, starts=6, seed=0):
    """CONSISTENT sinusoid-frequency extraction by GLOBAL multi-start joint NLS + BIC model selection —
    the extractor that exploits a wider observation window (KNOWN #26: short windows can't resolve
    superposed frequencies; greedy induce() fragments even on wide data). For each K it runs `starts`
    random restarts of joint coordinate-descent refinement and keeps the best residual, then picks K by
    BIC. Returns the discovered frequencies (sorted). Recovers true primitives consistently given enough
    data RANGE — which is what makes banked structure-reuse (compounding) REAL rather than capacity."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); N = len(Y); rng = np.random.default_rng(seed)

    def rss(freqs):
        cols = []
        for f in freqs:
            cols += [np.sin(f * X), np.cos(f * X)]
        A = np.stack(cols + [np.ones_like(X)], axis=1)
        c, *_ = np.linalg.lstsq(A, Y, rcond=None); return float(np.mean((A @ c - Y) ** 2))

    def joint(freqs):
        freqs = list(freqs)
        for _ in range(4):
            for i in range(len(freqs)):
                br, best = rss(freqs), freqs[i]
                for w in np.linspace(max(0.5, freqs[i] - 1.0), freqs[i] + 1.0, 41):
                    t = list(freqs); t[i] = float(w); r = rss(t)
                    if r < br:
                        br, best = r, float(w)
                freqs[i] = best
        return freqs, rss(freqs)

    best = None
    for K in range(1, Kmax + 1):
        bk = None
        for _ in range(starts):
            f, r = joint(rng.uniform(2.0, 16.0, K))
            if bk is None or r < bk[1]:
                bk = (sorted(f), r)
        bic = N * math.log(bk[1] + 1e-12) + (2 * K) * math.log(N)
        if best is None or bic < best[0]:
            best = (bic, bk[0])
    return best[1]


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
    # 3) JOINT continuous refinement of the sinusoid frequencies (coordinate descent on the FULL-program
    # residual) — converges each frequency to its EXACT value regardless of what else is superposed, so
    # discovery is precise AND consistent across data (the unified lever for extrapolation precision and
    # transfer; KNOWN #26). Non-sinusoid atoms (poly/exp/gaussian/const) stay fixed.
    fixed = [(atoms[j][0], atoms[j][1]) for j in active if not atoms[j][0][:3] in ("sin", "cos")]
    sinf = sorted({round(float(atoms[j][0][3:]), 4) for j in active if atoms[j][0][:3] in ("sin", "cos")})

    def build(freqs):
        cols = [fn(X) for _, fn in fixed] + [g(w * X) for w in freqs for g in (np.sin, np.cos)]
        return np.stack(cols, axis=1) if cols else np.ones((len(X), 1))

    def resid(freqs):
        A = build(freqs); c, *_ = np.linalg.lstsq(A, Y, rcond=None); return float(np.mean((A @ c - Y) ** 2))

    for _ in range(3):
        for i in range(len(sinf)):
            br, best = resid(sinf), sinf[i]
            for w in np.linspace(max(0.2, sinf[i] - 0.5), sinf[i] + 0.5, 81):
                trial = list(sinf); trial[i] = float(w); r = resid(trial)
                if r < br:
                    br, best = r, float(w)
            sinf[i] = best

    def design(fix, freqs):
        cols = [fn(X) for _, fn in fix] + [g(w * X) for w in freqs for g in (np.sin, np.cos)]
        return np.stack(cols, axis=1) if cols else np.ones((len(X), 1))

    coef, *_ = np.linalg.lstsq(design(fixed, sinf), Y, rcond=None); nf = len(fixed)
    # PRUNE negligible-amplitude components -> a CLEAN, parsimonious law (the same primitive discovered
    # in different data lands on the same few atoms, so laws BANK consistently and learning COMPOUNDS).
    fa = [abs(coef[i]) * float(np.linalg.norm(fixed[i][1](X))) for i in range(nf)]
    sa = [math.hypot(coef[nf + 2 * j], coef[nf + 2 * j + 1]) * float(np.linalg.norm(np.sin(sinf[j] * X)))
          for j in range(len(sinf))]
    amax = max([1e-9] + fa + sa)
    fixed = [fixed[i] for i in range(nf) if fixed[i][0] == "1" or fa[i] >= 0.06 * amax]
    sinf = [sinf[j] for j in range(len(sinf)) if sa[j] >= 0.06 * amax]
    Afull = design(fixed, sinf)
    coef, *_ = np.linalg.lstsq(Afull, Y, rcond=None); nf = len(fixed)

    def law(x):
        x = np.asarray(x, float)
        out = sum(coef[i] * fixed[i][1](x) for i in range(nf))
        for j, w in enumerate(sinf):
            out = out + coef[nf + 2 * j] * np.sin(w * x) + coef[nf + 2 * j + 1] * np.cos(w * x)
        return out
    terms = [(fixed[i][0], float(coef[i])) for i in range(nf)] + \
            [(f"sin{w:.3f}", float(coef[nf + 2 * j])) for j, w in enumerate(sinf)]
    # CONFIDENCE: fit coefs on the INTERIOR only and predict the held-out EDGE — does the law EXTEND to
    # unseen points, or only fit in-window? Low confidence flags a FAKE-FIT (e.g. a cubic standing in for
    # two sinusoids fits the window but collapses at the edge). A caller can reject low-confidence laws.
    cc, *_ = np.linalg.lstsq(Afull[core], Y[core], rcond=None)
    pe = Afull[edge] @ cc; ev = float(np.var(Y[edge])) + 1e-12
    conf = float(max(0.0, 1.0 - np.mean((pe - Y[edge]) ** 2) / ev))
    return law, terms, conf
