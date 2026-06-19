"""PROGRAM INDUCTION — honest scope + calibrated confidence (post-audit rewrite).

An adversarial audit (notes #28) showed the earlier "5/5 families crossed" was a CURATED point-set:
over randomized targets the cross-rate is much lower, and worse, the inducer could FAKE-FIT in-grammar
neighbours (sin5x+sin7x -> a cubic that fits the window but collapses off-support) with NO signal. So
this experiment now reports two honest things:

  1) cross-RATE over many RANDOMIZED targets per family (not a hand-picked set), and
  2) whether induce()'s held-out CONFIDENCE is CALIBRATED — i.e. does low confidence actually flag the
     fake-fits? A learner that knows when it has NOT found a law is honest; one that fakes with no signal
     is not. The real, defensible capability is "discovers + extends a law for cleanly-separable targets,
     AND reports low confidence when it cannot."

Run:  python experiments/program_induction.py
"""

import sys
import math
import numpy as np

from _util import REPO_ROOT  # noqa: F401
from recursivene.induction import induce

EXTRAP = np.linspace(1.0, 1.6, 60)        # outside training support [-1,1]
N = 20                                     # randomized targets per family


def extrap_score(fn, law):
    te = np.array([fn(x) for x in EXTRAP]); em = np.mean((te - te.mean()) ** 2)
    return max(0.0, 1 - float(np.mean((np.asarray(law(EXTRAP)) - te) ** 2)) / max(em, 1e-9))


def main():
    rng = np.random.default_rng(0); Xtr = rng.uniform(-1, 1, 600)
    fams = {
        "single-sin ": lambda: (lambda w: (f"sin{w:.1f}", lambda x: math.sin(w * x)))(rng.uniform(2, 12)),
        "two-sin    ": lambda: (lambda a, b: ("2sin", lambda x: math.sin(a * x) + 0.8 * math.sin(b * x)))(
            rng.uniform(2, 6), rng.uniform(7, 13)),
        "trend+sin  ": lambda: (lambda s, w: ("trend+sin", lambda x: s * x + math.sin(w * x)))(
            rng.uniform(0.4, 1.0), rng.uniform(4, 10)),
        "poly       ": lambda: (lambda a, b: ("poly", lambda x: a * x ** 3 + b * x ** 2 - 0.3))(
            rng.uniform(-1, 1), rng.uniform(-1, 1)),
        "exp        ": lambda: (lambda a: ("exp", lambda x: 0.4 * math.exp(a * x)))(rng.uniform(0.5, 1.2)),
    }
    print(f"\nPROGRAM INDUCTION — cross-RATE over {N} randomized targets/family + confidence calibration:\n")
    rows = []; all_succ, all_conf = [], []
    for name, gen in fams.items():
        succ, confs = [], []
        for _ in range(N):
            _, fn = gen()
            Y = np.array([fn(x) for x in Xtr]) + 0.02 * rng.standard_normal(len(Xtr))
            law, _, conf = induce(Xtr, Y, max_terms=8)
            s = extrap_score(fn, law); succ.append(s >= 0.85); confs.append(conf)
            all_succ.append(s >= 0.85); all_conf.append(conf)
        rate = np.mean(succ)
        rows.append((name, rate, np.median(confs)))
        print(f"  {name}: cross-rate {rate*100:4.0f}%   median confidence {np.median(confs):.2f}")

    # CONFIDENCE CALIBRATION: does high held-out confidence predict real extrapolation success?
    a = np.array(all_succ, float); c = np.array(all_conf, float)
    hi, lo = c >= 0.8, c < 0.8
    succ_hi = a[hi].mean() if hi.any() else float("nan")
    succ_lo = a[lo].mean() if lo.any() else float("nan")
    sep = (succ_hi - succ_lo) if (hi.any() and lo.any()) else 0.0
    print(f"\n  CONFIDENCE CALIBRATION: when conf>=0.8 -> extrapolates {succ_hi*100:.0f}% of the time;"
          f" when conf<0.8 -> {succ_lo*100:.0f}%.  separation={sep*100:.0f} pts")
    print("  (a useful confidence SEPARATES real laws from fake-fits, so a caller can trust/reject the law.)")
    print("\n" + "=" * 90)
    print("HONEST SCOPE: induction discovers + extends a law reliably for CLEANLY-SEPARABLE single-dominant-")
    print("component targets; multi-component / superposed targets cross less often (see rates). The defensible")
    print("capability is structure-discovery WITH a calibrated 'I-don't-know' signal — not '5/5 families'.")
    ok = sep >= 0.25 and rows[0][1] >= 0.5      # confidence is informative AND single-sin works often
    print("PASS — calibrated structure discovery (confidence separates real laws from fake-fits)." if ok
          else "PARTIAL — see rates/calibration above (honest).")
    print("=" * 90)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
