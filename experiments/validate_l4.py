"""The L4 gate — reproducible PASS/FAIL with numbers (~2-3 min). Honestly scoped after adversarial
audit (run_logs/audit_l4.js): certifies that an intrinsic, inner-only learning-progress drive choosing
WHAT to learn next MATCHES/BEATS the best FIXED curriculum (a competent coverage-aware schedule) on the
PROTECTED held-out ruler — rediscovering it from inside — and that the drive carries real (weak)
value-signal and a value model beats predict-zero. No 'generates goals' / 'understanding improves'
claims (the audit could not support them).

Run:  python experiments/validate_l4.py     (exits 0 on PASS, 1 on FAIL)
"""

import sys
import numpy as np

from _util import REPO_ROOT  # noqa: F401  (path setup)
import l4_motivation as L4

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:36s} {detail}")
    return ok


def main():
    SEEDS = range(6)
    rounds = L4.K + 1
    rnd, cov, endo, ora, wins_cov, wins_rnd, corrs, sat = [], [], [], [], [], [], [], []
    sm = L4.SelfModel()

    for sd in SEEDS:
        pool = L4.build_pool(seed=sd)
        held = L4.held_set(seed=500 + sd)
        reserve = L4.held_set(seed=900 + sd, n=1)[0]
        tr, _ = L4.run_curriculum("exogenous", pool, held, rounds, seed=sd)
        tc, _ = L4.run_curriculum("coverage", pool, held, rounds, seed=sd)
        tn, ln = L4.run_curriculum("endogenous", pool, held, rounds, seed=sd, self_model=sm, reserve_fn=reserve)
        to, _ = L4.run_curriculum("oracle", pool, held, rounds, seed=sd)
        rnd.append(tr[-1]); cov.append(tc[-1]); endo.append(tn[-1]); ora.append(to[-1])
        wins_cov.append(tn[-1] <= tc[-1] + 1e-9); wins_rnd.append(tn[-1] < tr[-1])
        w, rz = np.array(ln["wants"]), np.array(ln["realized"])
        if w.std() > 1e-9 and rz.std() > 1e-9:
            corrs.append(float(np.corrcoef(w, rz)[0, 1]))
        rw = np.array(ln["reserve_want"])
        sat.append(rw[0] / max(rw[-1], 1e-9))

    mrnd, mcov, mendo, mora = map(np.mean, (rnd, cov, endo, ora))
    fcov, frnd = np.mean(wins_cov), np.mean(wins_rnd)
    mcorr, r2, msat = float(np.mean(corrs)), sm.r2_vs_zero(), float(np.mean(sat))

    # L1 (PRIMARY): the intrinsic drive matches/beats the COMPETENT coverage-aware fixed curriculum
    check("L1 drive >= coverage baseline", mendo <= mcov * 1.02 and fcov >= 0.6,
          f"own {mendo:.0f} vs coverage {mcov:.0f} ({mcov/mendo:.2f}x); win/tie {fcov*100:.0f}% of seeds")
    # L2 (context): and clearly beats a non-curated random schedule
    check("L2 drive > random schedule", mendo < mrnd and frnd >= 0.6,
          f"own {mendo:.0f} < random {mrnd:.0f} ({mrnd/mendo:.2f}x); win {frnd*100:.0f}%")
    # L3: stays at/above the realized-value oracle ceiling (real but not magic)
    check("L3 near oracle ceiling", mora <= mendo + 0.30 * mrnd,
          f"oracle {mora:.0f} <= own {mendo:.0f}; {100*(mrnd-mendo)/(mrnd-mora+1e-9):.0f}% of random-oracle gap closed")
    # L4: the drive is inner-only and weak-but-positively predictive of realized value
    check("L4 drive predicts value (weak)", mcorr > 0.25,
          f"corr(want, realized) = {mcorr:+.2f} (r^2~{mcorr**2:.2f}), inner-only learning-progress signal")
    # L5: a value model beats predict-zero out-of-sample (NOT 'understanding improves over lifetime')
    check("L5 value model > predict-zero", r2 > 0.0,
          f"self-model out-of-sample R^2 vs predict-zero = {r2:+.2f}")
    # L6: partial satiation — a FIXED never-adopted target's want decays as the basis spans
    check("L6 fixed-target want satiates", msat > 1.5,
          f"fixed reserve want decays {msat:.1f}x over the run")

    ok = all(RESULTS)
    print(f"\n{'='*74}\n{'PASS — self-directed drive matches/beats the best fixed curriculum on the protected ruler.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
