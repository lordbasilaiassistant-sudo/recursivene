"""EXTRAPOLATION — the g-panel's weakest link, diagnosed: approximation fails, structure-discovery crosses.

The honesty-audited g-panel (KNOWN #23) flagged out-of-family / EXTRAPOLATION as the entity's weakest
facet. This experiment diagnoses WHY, and finds the real lever — the honest scientific payoff of a panel
that surfaces its own boundary.

Task: learn sin(w*x) from samples on [-1,1], then predict on [1.0,1.6] — OUTSIDE the training support.
Three learners on the identical data:

  FIXED features (RFF+RLS)       — a function APPROXIMATOR. Fails (periodic features don't extend).
  LEARNED features (deep MLP)    — a bigger approximator. Fails WORSE (tanh saturates off-support).
  STRUCTURE DISCOVERY (system ID)— estimate the generative frequency w from the data, fit amplitude+phase,
                                   then EXTEND the law. Extrapolates ~perfectly.

The lesson (KNOWN #24): generalizing beyond the data is not approximation, it is discovering the
generating STRUCTURE and extending it. Scaling the approximator does not cross this wall; discovering
the law does. This is exactly the SpectralEncoder direction (KNOWN #12) — and the honest reason the
entity's hardest wall is crossable by a better KIND of representation (one that finds programs/laws),
not a bigger one.

Run:  python experiments/extrapolation.py
"""

import math
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.deep_encoder import DeepEncoder
from recursivene.encoder import SpectralEncoder

W_TRUE = 5.0
EXTRAP = np.linspace(1.0, 1.6, 60)        # outside the [-1,1] training support


def main():
    fn = lambda x: math.sin(W_TRUE * x)
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-1, 1, (2000, 1)); Ytr = np.array([fn(x[0]) for x in Xtr])
    xtr = Xtr[:, 0]; ytr_n = Ytr + 0.02 * rng.standard_normal(len(Ytr))
    te = np.array([fn(x) for x in EXTRAP]); e_mean = np.mean((te - te.mean()) ** 2)
    score = lambda mse: max(0.0, 1 - mse / e_mean)

    # 1) FIXED features (RFF + RLS)
    r = np.random.default_rng(1); D = 256; Wr = r.normal(0, 8, D); b = r.uniform(0, 2 * np.pi, D); s = math.sqrt(2 / D)
    phi = lambda x: s * np.cos(Wr * x + b); P = np.eye(D); w = np.zeros(D)
    for x, y in zip(xtr, ytr_n):
        f = phi(x); Pp = P @ f; k = Pp / (1 + f @ Pp); w = w + k * (y - f @ w); P = P - np.outer(k, Pp)
    rff = np.mean((np.array([w @ phi(x) for x in EXTRAP]) - te) ** 2)

    # 2) LEARNED features (deep MLP)
    enc = DeepEncoder(1, hidden=128, seed=0); enc.fit(Xtr, Ytr, iters=4000)
    mlp = float(np.mean((enc.predict(EXTRAP.reshape(-1, 1)) - te) ** 2))

    # 3) STRUCTURE DISCOVERY (system identification): find the generative frequency, extend the law
    best = None
    for wf in np.linspace(0.5, 12, 600):
        A = np.c_[np.sin(wf * xtr), np.cos(wf * xtr), np.ones_like(xtr)]
        coef, *_ = np.linalg.lstsq(A, ytr_n, rcond=None)
        err = np.mean((A @ coef - ytr_n) ** 2)
        if best is None or err < best[0]: best = (err, wf, coef)
    _, wf, coef = best
    pred = np.c_[np.sin(wf * EXTRAP), np.cos(wf * EXTRAP), np.ones_like(EXTRAP)] @ coef
    sid = float(np.mean((pred - te) ** 2))

    # 4a) THE ENTITY's SpectralEncoder, FULL representation (rich phi: over-capacity for in-support fit)
    enc = SpectralEncoder(n_freqs=20, fmax=20.0, seed=0)
    for x, y in zip(xtr, ytr_n): enc.observe(x, y)
    enc.discover()
    de = enc.dim(); P = np.eye(de); we = np.zeros(de)
    for x in xtr:
        f = enc.phi(x); Pp = P @ f; k = Pp / (1 + f @ Pp); we = we + k * (fn(x) - f @ we); P = P - np.outer(k, Pp)
    spec_full = float(np.mean((np.array([we @ enc.phi(x) for x in EXTRAP]) - te) ** 2))
    # 4b) THE ENTITY's parsimonious LAW (BIC-selected few frequencies): what it EXTRAPOLATES with
    law = enc.law()
    def lphi(x): return np.concatenate([np.sin(law * x), np.cos(law * x), [1.0]])
    dl = 2 * len(law) + 1; P = np.eye(dl); wl = np.zeros(dl)
    for x in xtr:
        f = lphi(x); Pp = P @ f; k = Pp / (1 + f @ Pp); wl = wl + k * (fn(x) - f @ wl); P = P - np.outer(k, Pp)
    spec_law = float(np.mean((np.array([wl @ lphi(x) for x in EXTRAP]) - te) ** 2))
    near = min(law, key=lambda f: abs(f - W_TRUE))

    print("\nEXTRAPOLATION to [1.0, 1.6] (outside training support); 1.0=perfect, 0=predict-the-mean:\n")
    print(f"  FIXED features (RFF)            : mse={rff:7.3f}   score={score(rff):.2f}")
    print(f"  LEARNED features (deep MLP)     : mse={mlp:7.3f}   score={score(mlp):.2f}")
    print(f"  ENTITY encoder, FULL phi (rich) : mse={spec_full:7.3f}   score={score(spec_full):.2f}   ({len(enc.freqs)} freqs; over-capacity pollutes off-support)")
    print(f"  ENTITY encoder, parsimonious LAW: mse={spec_law:7.3f}   score={score(spec_law):.2f}   ({len(law)} freq(s), w={near:.3f}; CROSSES the wall)")
    print(f"  STRUCTURE DISCOVERY (sys-ID)    : mse={sid:7.4f}   score={score(sid):.2f}   (one freq {wf:.3f})")
    print("\n" + "=" * 90)
    print("LESSON: extrapolation is STRUCTURE DISCOVERY, not approximation. RFF/MLP can't extend a law;")
    print("the entity's RICH representation (phi) is great IN-support but its over-capacity pollutes OFF-")
    print("support; its parsimonious LAW (BIC-selected few frequencies, enc.law()) CROSSES the wall. So the")
    print("entity keeps BOTH: max-capacity representation for fitting, min-description law for extending.")
    print("Representation vs Occam are different objectives — and intelligence needs both. (KNOWN #12, #24.)")
    print("=" * 90)


if __name__ == "__main__":
    main()
