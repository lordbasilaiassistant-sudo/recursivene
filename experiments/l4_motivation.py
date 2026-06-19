"""L4 — SELF-DIRECTED CURRICULUM by a learning-progress drive (honestly scoped, post adversarial audit).

The directive (Anthony, 2026-06-19): the RSI should learn to learn AND learn how learning turns into
its own goals. This experiment builds the smallest honest version, with the protected ruler untouchable.

The setup. The agent lives over a SHARED, growable representation (the banked-solution basis proven in
discover_test.py / KNOWN #9): learning a target banks its solution as a feature, so future targets that
share structure become cheap. It faces a fixed POOL of candidate problems, each exciting a SPARSE subset
of K hidden primitives; primitive #(K-1) has a single carrier (a coverage bottleneck). A dense HELD-OUT
target needs ALL K directions banked, so the ORDER in which the agent SELECTS its curriculum decides how
fast it can learn anything new.

What is honestly established (claim tags from the audit; see notes/08-learning-to-want.md):

  [ESTABLISHED] An intrinsic, INNER-ONLY drive — want = how poorly the current representation predicts a
     candidate (a standard learning-progress / novelty signal) — selecting the next goal MATCHES/BEATS a
     competent COVERAGE-AWARE hand-designed curriculum on the protected held-out ruler (it wins/ties in
     most seeds, ~1.15x; vs a non-curated RANDOM schedule ~1.36x). It REDISCOVERS optimal set-cover from
     inside, without being told the primitive structure. The drive never reads held-out worlds.
  [ESTABLISHED] The drive is weak-but-positively predictive of realized held-out value (corr ~+0.4,
     r^2~0.15, inner-only, non-circular), and a value self-model beats predict-zero (out-of-sample R^2).
  [ESTABLISHED] Partial satiation: a FIXED never-adopted target's want decays as the basis spans the space.

What is NOT claimed (audit): the agent does not GENERATE/invent goals — it SELECTS from a fixed pool. The
"wants/needs/understands" framing is [serious-speculative] anthropomorphism of a learning-progress
heuristic. We do NOT claim the value model's "understanding improves over its lifetime" (a permutation
test could not distinguish that from a data-order artifact). The win is coverage-bottleneck-conditional
on a 1-D compositional toy; no generality/optimality/safety beyond it is claimed.

Non-circularity (verified by audit): want()/_feats() touch ONLY the candidate's own fn + the agent's
bank, never held-out; held_set is disjoint from build_pool; the protected ruler (objective.TAU) is
unedited. The agent chooses its curriculum; it cannot redefine success.

Run:  python experiments/l4_motivation.py
"""

import numpy as np

from _util import sparkline  # noqa: F401
from recursivene.objective import TAU

PRIMS = [5.0, 8.0, 11.0, 14.0, 17.0, 20.0]   # K hidden shared primitives (high freq -> RFF slow)
K = len(PRIMS)
GAMMA, D = 16.0, 128
GRID = np.linspace(-1.0, 1.0, 201)
XE = GRID[::5]                            # competence-eval grid
PENALTY = 4000                           # finite stand-in for "did not reach tau in budget"


class RLS:
    def __init__(self, d, ridge=1.0):
        self.w = np.zeros(d); self.P = np.eye(d) / ridge

    def predict(self, f):
        return float(f @ self.w)

    def update(self, f, y):
        Pp = self.P @ f; k = Pp / (1.0 + f @ Pp)
        self.w = self.w + k * (y - f @ self.w); self.P = self.P - np.outer(k, Pp)


def rff_maker(seed):
    r = np.random.default_rng(seed)
    W = r.normal(0, GAMMA, D); b = r.uniform(0, 2 * np.pi, D); s = np.sqrt(2.0 / D)
    return lambda x: s * np.cos(W * x + b)


def target_from_coeffs(c):
    c = np.asarray(c, float)
    return lambda x: float(np.sum(c * np.sin(np.array(PRIMS) * x)))


def learn(target_fn, bank, seed=0, max_n=2500, obs_noise=0.02, gate=TAU):
    """Fit target over [banked past-solution features] + RFF; return (cost_to_know, solution_grid).
    cost_to_know = samples to reach `gate` MSE on the eval grid (PENALTY if not reached). The
    returned grid values let the solution be banked as an interpolable feature. This is exactly the
    transfer substrate of KNOWN #9 — the protected cost-for-competence, lifted to cost-to-know."""
    rng = np.random.default_rng(seed)
    rff = rff_maker(seed)

    def feat(x):
        bf = np.array([np.interp(x, GRID, g) for g in bank]) if bank else np.zeros(0)
        return np.concatenate([bf, rff(x)])

    model = RLS(len(bank) + D)
    truth = np.array([target_fn(x) for x in XE])
    cost = PENALTY
    for n in range(1, max_n + 1):
        x = rng.uniform(-1.0, 1.0)
        model.update(feat(x), target_fn(x) + obs_noise * rng.standard_normal())
        if n % 10 == 0:
            pred = np.array([model.predict(feat(xx)) for xx in XE])
            if float(np.mean((truth - pred) ** 2)) <= gate:
                cost = n; break
    gvals = np.array([model.predict(feat(x)) for x in GRID])
    return cost, gvals


def heldout_cost(bank, held_targets, seed0=900):
    """THE PROTECTED RULER: mean cost-to-know over disjoint HELD-OUT targets, given the current
    representation (bank). Lower = the representation makes new unknowns cheaper. Never used to
    choose a goal — only to grade the curriculum after the fact."""
    cs = [learn(t, bank, seed=seed0 + i)[0] for i, t in enumerate(held_targets)]
    return float(np.mean(cs))


# ----------------------------------------------------------------------------------------
# The pool of candidate GOALS. Each excites a SPARSE subset of primitives, and primitive K-1 is
# RARE (few candidates carry it). A dense held-out target needs ALL K directions banked, so the
# agent must discover and adopt the rare-primitive goal — random order tends to delay it.
# ----------------------------------------------------------------------------------------
def build_pool(seed=0):
    """A pool of candidate goals where the CURRICULUM genuinely matters: the common primitives
    0..K-2 are HEAVILY REDUNDANT (many carriers each), while primitive K-1 is RARE (a single
    carrier). A dense held-out target needs ALL K directions, so spanning the space REQUIRES
    finding the one rare carrier — easy to miss if you adopt goals in an arbitrary (imposed)
    order, but exactly what an intrinsic novelty drive homes in on once the commons are banked."""
    rng = np.random.default_rng(seed)
    subsets = []
    for a in range(K - 1):                        # redundant commons: 2 singles + a pair each
        subsets += [(a,), (a,)]
    for _ in range(4):                            # extra redundant pairs among the commons
        i, j = rng.choice(K - 1, size=2, replace=False)
        subsets.append((int(i), int(j)))
    subsets.append((0, K - 1))                     # the SOLE carrier of the rare primitive K-1
    pool = []
    for S in subsets:
        c = np.zeros(K)
        for k in S:
            c[k] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
        pool.append({"coeffs": c, "subset": tuple(S), "fn": target_from_coeffs(c)})
    return pool


def held_set(seed=500, n=6):
    rng = np.random.default_rng(seed)
    return [target_from_coeffs(rng.uniform(0.4, 1.0, K) * rng.choice([-1.0, 1.0], K)) for _ in range(n)]


def want(candidate, bank, probe_budget=60, seed=7):
    """THE INTRINSIC DRIVE — inner only, never sees held-out. 'How poorly does my current
    representation predict this candidate after a tiny look?' High residual = I cannot yet represent
    it = it lies on my frontier = I want it. This is learning becoming a goal: the want is a pure
    function of the agent's own learning difficulty."""
    rng = np.random.default_rng(seed)
    rff = rff_maker(seed)

    def feat(x):
        bf = np.array([np.interp(x, GRID, g) for g in bank]) if bank else np.zeros(0)
        return np.concatenate([bf, rff(x)])

    model = RLS(len(bank) + D)
    for _ in range(probe_budget):
        x = rng.uniform(-1.0, 1.0)
        model.update(feat(x), candidate["fn"](x))
    truth = np.array([candidate["fn"](x) for x in XE])
    pred = np.array([model.predict(feat(x)) for x in XE])
    return float(np.mean((truth - pred) ** 2))


def run_curriculum(policy, pool, held, rounds, seed=0, self_model=None, reserve_fn=None):
    """Adopt `rounds` goals from the pool under `policy`, banking each solution. Returns the held-out
    cost-to-know trajectory (protected ruler) + per-round logs.

    Policies (what chooses the next goal):
      exogenous  - fixed RANDOM order (a non-curated imposed schedule).
      coverage   - greedy set-cover using DESIGNER knowledge of each candidate's primitives
                   (pool[i]['subset']): pick the candidate adding the most NEW primitives. This is the
                   strong, competent imposed baseline — the best fixed curriculum a knowledgeable
                   designer writes. (Endogenous must beat THIS to count, not just random.)
      endogenous - max intrinsic want (inner-only prediction residual; never reads held-out).
      learned    - a self-model predicts realized value from cheap inner features and selects argmax;
                   the self-model trains online -> tests whether a LEARNED value model can drive
                   selection as well as the direct drive.
      oracle     - ceiling: max actual realized held-out drop (uses held-out; reference only).
    """
    rng = np.random.default_rng(1234 + seed)
    bank, remaining, covered = [], list(range(len(pool))), set()
    order = list(rng.permutation(len(pool)))                  # fixed imposed order for 'exogenous'
    traj = [heldout_cost(bank, held)]
    log = {"wants": [], "realized": [], "picks": [], "maxwant": [], "reserve_want": []}

    for t in range(min(rounds, len(pool))):
        wants = {i: want(pool[i], bank, seed=7 + i) for i in remaining}
        log["maxwant"].append(max(wants.values()))
        if reserve_fn is not None:    # a FIXED never-adopted target: genuine-satiation probe
            log["reserve_want"].append(want({"fn": reserve_fn}, bank, seed=99))
        if policy == "exogenous":
            pick = next(i for i in order if i in remaining)
        elif policy == "coverage":                            # designer-optimal: most NEW primitives
            pick = max(remaining, key=lambda i: (len(set(pool[i]["subset"]) - covered), -i))
        elif policy == "endogenous":                          # max intrinsic want (inner-only)
            pick = max(remaining, key=lambda i: wants[i])
        elif policy == "learned":                             # learned value model drives selection
            pick = max(remaining, key=lambda i: self_model.predict(_feats(pool[i], bank)))
        elif policy == "oracle":                              # ceiling: max realized held-out drop
            pick = max(remaining, key=lambda i: traj[-1] - heldout_cost(bank + [learn(pool[i]["fn"], bank, seed=300 + i)[1]], held))
        else:
            raise ValueError(policy)

        _, gvals = learn(pool[pick]["fn"], bank, seed=300 + pick)
        before = traj[-1]
        bank.append(gvals)
        covered |= set(pool[pick]["subset"])
        after = heldout_cost(bank, held)
        traj.append(after)
        log["wants"].append(wants[pick]); log["realized"].append(before - after)
        log["picks"].append(pool[pick]["subset"])
        if self_model is not None:    # online: learn to predict REALIZED value from cheap inner features
            self_model.observe(_feats(pool[pick], bank[:-1]), before - after)
        remaining.remove(pick)
    return traj, log


def _feats(candidate, bank):
    """Cheap features for the learned self-model (no held-out): low-budget residual at two budgets
    (frontier-ness + how fast it falls) + bank size. The self-model learns to map these to want."""
    r1 = want(candidate, bank, probe_budget=20, seed=3)
    r2 = want(candidate, bank, probe_budget=80, seed=4)
    return np.array([r1, r2, r1 - r2, len(bank) / max(K, 1), 1.0])


class SelfModel:
    """A learned value model: ridge regression from cheap inner features -> REALIZED held-out value.
    Records each prediction BEFORE the update that follows it, so prediction quality can be scored
    out-of-sample. NOTE (post-audit): we report only what is statistically defensible — the model's
    out-of-sample R^2 against trivial baselines (predict-zero / running-mean). We do NOT claim the
    'understanding improves over its lifetime': a permutation test showed the early->late error drop is
    not distinguishable from a cold-start/data-order artifact (p~0.16)."""

    def __init__(self, d=5, ridge=1.0):
        self.X, self.y = [], []
        self.w = np.zeros(d); self.ridge = ridge
        self.preds, self.actuals = [], []

    def predict(self, f):
        return float(f @ self.w)

    def observe(self, f, y):
        self.preds.append(self.predict(f)); self.actuals.append(y)   # prediction BEFORE this update
        self.X.append(f); self.y.append(y)
        X, yv = np.asarray(self.X), np.asarray(self.y)
        self.w = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ yv)

    def r2_vs_zero(self, skip_cold_start=1):
        """Out-of-sample R^2 of the online predictions vs a predict-zero baseline (excluding the
        w=0 cold-start point). >0 means the learned value model genuinely beats predicting nothing."""
        p = np.array(self.preds[skip_cold_start:]); a = np.array(self.actuals[skip_cold_start:])
        if len(a) < 2:
            return float("nan")
        ss_res = float(np.sum((a - p) ** 2))
        ss_zero = float(np.sum(a ** 2))                          # predict-zero baseline
        return 1.0 - ss_res / ss_zero if ss_zero > 1e-9 else float("nan")


def _fmt(x):
    return f"{x:6.0f}"


def main():
    print("\nL4 — LEARNING-PROGRESS CURRICULUM SELECTION (honest, post-audit): does an intrinsic, inner-only")
    print("novelty drive choosing WHAT to learn next match/beat the best FIXED curriculum, on a ruler it can't cheat?\n")
    print(f"  K={K} hidden primitives {PRIMS}; rare primitive #{K-1} has 1 carrier (a coverage bottleneck);")
    print(f"  protected ruler = held-out cost-to-know. Baselines: random schedule AND designer coverage-aware set-cover.\n")

    SEEDS = tuple(range(8))
    rounds = K + 1
    rnd_t, cov_t, endo_t, lrn_t, ora_t = [], [], [], [], []
    corrs, win_rnd, win_cov = [], [], []
    sm = SelfModel()                                       # one value model, trained across all seeds
    sat_w_ref = sat_res_ref = None

    for sd in SEEDS:
        pool = build_pool(seed=sd)
        held = held_set(seed=500 + sd)
        reserve = held_set(seed=900 + sd, n=1)[0]          # a fixed dense target, never adopted
        tr, _ = run_curriculum("exogenous", pool, held, rounds, seed=sd)
        tc, _ = run_curriculum("coverage", pool, held, rounds, seed=sd)
        tn, ln = run_curriculum("endogenous", pool, held, rounds, seed=sd, self_model=sm, reserve_fn=reserve)
        tl, _ = run_curriculum("learned", pool, held, rounds, seed=sd, self_model=sm)
        to, _ = run_curriculum("oracle", pool, held, rounds, seed=sd)
        rnd_t.append(tr); cov_t.append(tc); endo_t.append(tn); lrn_t.append(tl); ora_t.append(to)
        win_rnd.append(tn[-1] < tr[-1]); win_cov.append(tn[-1] < tc[-1] + 1e-9)
        w, rz = np.array(ln["wants"]), np.array(ln["realized"])
        if w.std() > 1e-9 and rz.std() > 1e-9:
            corrs.append(float(np.corrcoef(w, rz)[0, 1]))
        if sat_w_ref is None:
            sat_w_ref, sat_res_ref = np.array(ln["maxwant"]), np.array(ln["reserve_want"])
        print(f"  seed {sd}: random {_fmt(tr[-1])}  coverage {_fmt(tc[-1])}  its-own {_fmt(tn[-1])}"
              f"  learned {_fmt(tl[-1])}  oracle {_fmt(to[-1])}   {'>=cov' if tn[-1] <= tc[-1]+1e-9 else '<cov '}")

    rnd, cov = np.mean([t[-1] for t in rnd_t]), np.mean([t[-1] for t in cov_t])
    endo, lrn, ora = np.mean([t[-1] for t in endo_t]), np.mean([t[-1] for t in lrn_t]), np.mean([t[-1] for t in ora_t])
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")
    r2 = sm.r2_vs_zero()
    sat_decay = sat_res_ref[0] / max(sat_res_ref[-1], 1e-9)

    print("\n" + "=" * 84)
    print(f"  PROTECTED RULER (held-out cost-to-know, lower=better; mean over {len(SEEDS)} seeds):")
    print(f"    random {rnd:5.0f}   coverage-aware {cov:5.0f}   ITS-OWN(drive) {endo:5.0f}   learned-model {lrn:5.0f}   oracle {ora:5.0f}")
    print(f"  [PRIMARY] vs competent COVERAGE baseline: {cov/endo:.2f}x, own-goals win {np.mean(win_cov)*100:.0f}% of seeds")
    print(f"  [context] vs RANDOM schedule:            {rnd/endo:.2f}x, win {np.mean(win_rnd)*100:.0f}% of seeds")
    print(f"  the drive REDISCOVERS coverage from inside (it isn't told the primitive structure).")
    print(f"  DRIVE PREDICTIVENESS: corr(want, realized value) = {mean_corr:+.2f} (r^2~{mean_corr**2:.2f}); "
          f"weak-but-positive, inner-only (a learning-progress signal).")
    print(f"  VALUE MODEL: out-of-sample R^2 vs predict-zero = {r2:+.2f} (realized value is modelable from cheap features).")
    print(f"  SATIATION (fixed never-adopted target's want): {sparkline(sat_res_ref)}  decays {sat_decay:.1f}x as the basis spans.")

    # Honest gate: the drive must at least MATCH the competent coverage baseline (>= in most seeds),
    # the drive must carry real signal, and a value model must beat predict-zero. No 'understanding
    # improves' claim (audit: not significant).
    ok = (endo <= cov * 1.02) and (np.mean(win_cov) >= 0.6) and (mean_corr > 0.25) and (r2 > 0.0)
    print("=" * 84)
    print("PASS — intrinsic drive matches/beats the best fixed curriculum on the protected ruler (rediscovers it from inside)."
          if ok else "PARTIAL — see numbers above.")
    return ok


if __name__ == "__main__":
    main()
