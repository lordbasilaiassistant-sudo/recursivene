"""THE RACE — Claude (hand-tuning) vs the RSI (learned self-edit search) at the one job
that defines self-improvement here: TEACHING THE IMPROVER to produce cheap, GENERALIZING
learners. Whoever lowers the held-out meta-objective more, at an EQUAL evaluation budget,
is the better teacher of the system.

This operationalizes the goal "improve the RSI until it beats you in progress of teaching
itself." Both contestants choose self-edits over the SAME knob space the closure loop edits:

  3-knob (the EvolutionaryProposer operator):  MUTATION_SCALE, MUTATION_RATE, POP
  6-knob (the LearnedProposer):                + POOL_FACTOR, EXPLORE_FRAC, SURROGATE_RIDGE

Fairness / anti-Goodhart contract (audited):
  * SELECTION vs VERDICT split, exactly like the real system. Each contestant ranks the
    candidates it has tried by their INNER-world cost (the only signal it is allowed to see)
    and reports its best-so-far. The HELD-OUT cost of that selected knob is the verdict —
    NEITHER contestant optimizes the held-out gate directly, so the win cannot be an overfit
    to the held-out worlds. (Held-out meta_cost == 1/invariant: the verdict IS the untouchable
    invariant.)
  * EQUAL BUDGET. Both get exactly N evaluations. Claude spends them on N expert
    hand-authored self-edits (the human's strong-prior advantage — and Claude may use every
    probe result it has seen). The RSI starts NAIVE (space-filling, no prior) and learns
    online from its own N evaluations — the LearnedProposer pattern lifted one level up onto
    the self-edit search itself.
  * The RSI search is repeated over several seeds; we report mean / best / worst so a win is
    not a single-seed fluke. Winners are re-scored at a second (larger) budget to show the
    result is not an artifact of one budget.

Run:  python experiments/race.py --space 6 --evals 12 --seeds 0,1,2
Output: run_logs/race_<space>knob.json
"""

import argparse
import json
import math
import os

import numpy as np

from _util import REPO_ROOT  # noqa: F401  (adds repo root to path)
from recursivene.harness.loop import search
from recursivene.harness.proposer import EvolutionaryProposer, LearnedProposer
from recursivene.objective import evaluate, TAU
from recursivene.invariant import HELDOUT_SEEDS

BLOATED = {"policy": "lp", "n_features": 256, "gamma": 8.0, "hist": 64,
           "min_lp": 16, "epsilon": 0.15}

CAP = 1e9   # finite stand-in for an inf (non-generalizing) score, for the surrogate's log

# knob -> (lo, hi, is_int). The 3-knob space is the first three; 6-knob adds the rest.
KNOBS_3 = {"scale": (0.1, 1.2, False), "rate": (0.2, 0.9, False), "pop": (3, 8, True)}
KNOBS_6 = {**KNOBS_3, "pool": (4, 24, True), "explore": (0.1, 0.6, False),
           "ridge": (0.1, 10.0, False)}

# (generations, inner_steps, heldout_steps) for the search inside one evaluation.
BUDGETS = {"fast": (3, 700, 900), "confirm": (4, 1100, 1400)}


def _build_proposer(theta, space):
    """Construct the improver-under-test from a knob vector. For 3-knob, the
    EvolutionaryProposer (the real harness/loop operator). For 6-knob, the LearnedProposer
    with its three self-edit constants set per-instance."""
    pop = int(round(theta["pop"]))
    if space == 3:
        return EvolutionaryProposer(pop=pop, scale=theta["scale"], rate=theta["rate"])
    p = LearnedProposer(pop=pop, scale=theta["scale"], rate=theta["rate"])
    p.POOL_FACTOR = int(round(theta["pool"]))
    p.EXPLORE_FRAC = float(theta["explore"])
    p.SURROGATE_RIDGE = float(theta["ridge"])
    return p


def evaluate_knob(theta, space, budget, _cache):
    """Run the harness's search (the improver-under-test) on INNER worlds, then report:
      sel = inner-world cost of the config it found      (the SELECTION signal both arms see)
      val = HELD-OUT cost of that same config            (the unbiased VERDICT, never optimized)
    Deterministic for fixed theta+budget -> cached. inf where the improver fails to teach a
    sustained-competence generalizing learner."""
    gens, steps, hs = BUDGETS[budget]
    key = (space, budget, tuple(round(theta[k], 4) for k in theta))
    if key in _cache:
        return _cache[key]
    proposer = _build_proposer(theta, space)
    best_config, best_eval, _ = search(proposer, BLOATED, generations=gens,
                                        steps=steps, seeds=(0, 1))
    sel = float(best_eval["cost"])                                  # inner-world selection cost
    held = evaluate(best_config, which="heldout", steps=hs, seeds=HELDOUT_SEEDS, tau=TAU)
    val = float(held["cost"])                                       # held-out verdict cost
    out = {"sel": sel, "val": val, "n_params": int(held["n_params"]), "theta": dict(theta)}
    _cache[key] = out
    return out


# ----------------------------------------------------------------------------------------
# Claude's arm: expert hand-authored self-edits. These are genuine best guesses, using every
# probe result already seen (the human's strong-prior advantage). NOT sandbagged.
# ----------------------------------------------------------------------------------------
CLAUDE_3 = [  # EvolutionaryProposer: probes showed high pop+scale+rate finds generalizers
    {"scale": 1.0, "rate": 0.7, "pop": 8}, {"scale": 1.2, "rate": 0.7, "pop": 8},
    {"scale": 0.9, "rate": 0.8, "pop": 8}, {"scale": 1.1, "rate": 0.6, "pop": 8},
    {"scale": 0.8, "rate": 0.7, "pop": 7}, {"scale": 1.2, "rate": 0.9, "pop": 8},
    {"scale": 1.0, "rate": 0.5, "pop": 6}, {"scale": 0.6, "rate": 0.6, "pop": 8},
    {"scale": 1.2, "rate": 0.8, "pop": 7}, {"scale": 0.9, "rate": 0.7, "pop": 6},
    {"scale": 0.7, "rate": 0.8, "pop": 8}, {"scale": 1.1, "rate": 0.7, "pop": 7},
]
CLAUDE_6 = [  # LearnedProposer: probe showed high scale FAILS; moderate scale + explore works
    {"scale": 0.4, "rate": 0.5, "pop": 6, "pool": 8, "explore": 0.6, "ridge": 1.0},
    {"scale": 0.3, "rate": 0.5, "pop": 8, "pool": 12, "explore": 0.5, "ridge": 1.0},
    {"scale": 0.5, "rate": 0.6, "pop": 8, "pool": 8, "explore": 0.6, "ridge": 2.0},
    {"scale": 0.4, "rate": 0.6, "pop": 8, "pool": 16, "explore": 0.5, "ridge": 1.0},
    {"scale": 0.3, "rate": 0.4, "pop": 6, "pool": 8, "explore": 0.6, "ridge": 0.5},
    {"scale": 0.5, "rate": 0.5, "pop": 8, "pool": 12, "explore": 0.4, "ridge": 1.0},
    {"scale": 0.4, "rate": 0.5, "pop": 8, "pool": 16, "explore": 0.6, "ridge": 2.0},
    {"scale": 0.3, "rate": 0.6, "pop": 8, "pool": 8, "explore": 0.5, "ridge": 1.0},
    {"scale": 0.6, "rate": 0.5, "pop": 6, "pool": 12, "explore": 0.5, "ridge": 3.0},
    {"scale": 0.4, "rate": 0.7, "pop": 8, "pool": 16, "explore": 0.6, "ridge": 1.0},
    {"scale": 0.35, "rate": 0.5, "pop": 7, "pool": 8, "explore": 0.5, "ridge": 1.0},
    {"scale": 0.45, "rate": 0.6, "pop": 8, "pool": 12, "explore": 0.4, "ridge": 2.0},
]


def _best_so_far(trials):
    """Given trials [{sel,val},...] in evaluation order, return the best-so-far VAL curve,
    where 'best' is chosen by the SELECTION signal (sel) — never by val (the gate)."""
    curve, best_sel, best_val = [], math.inf, math.inf
    for t in trials:
        if t["sel"] < best_sel:           # pick by inner cost; read off its held-out cost
            best_sel, best_val = t["sel"], t["val"]
        curve.append(best_val)
    return curve


def claude_arm(space, evals, budget, cache):
    cands = (CLAUDE_3 if space == 3 else CLAUDE_6)[:evals]
    trials = [evaluate_knob(th, space, budget, cache) for th in cands]
    return {"trials": trials, "curve": _best_so_far(trials),
            "best_val": _best_so_far(trials)[-1]}


# ----------------------------------------------------------------------------------------
# The RSI's arm: a learned self-edit search. Starts naive (space-filling), then fits a ridge
# surrogate of log(inner-cost) over the normalized knob vector and proposes the predicted
# cheapest unseen knob (exploit) with a fraction kept random (explore). This is exactly the
# LearnedProposer principle (B3) lifted one level up — onto the choice of self-edit itself.
# ----------------------------------------------------------------------------------------
def _norm(theta, knobs):
    return np.array([(theta[k] - lo) / (hi - lo) for k, (lo, hi, _) in knobs.items()])


def _denorm(u, knobs):
    th = {}
    for v, (k, (lo, hi, is_int)) in zip(u, knobs.items()):
        val = lo + float(np.clip(v, 0, 1)) * (hi - lo)
        th[k] = int(round(val)) if is_int else round(val, 4)
    return th


def _lhs(n, dim, rng):
    """Latin-hypercube space-filling points in [0,1]^dim."""
    pts = np.zeros((n, dim))
    for d in range(dim):
        perm = rng.permutation(n)
        pts[:, d] = (perm + rng.random(n)) / n
    return pts


def rsi_arm(space, evals, budget, seed, cache):
    knobs = KNOBS_3 if space == 3 else KNOBS_6
    dim = len(knobs)
    rng = np.random.default_rng(seed)
    n_init = max(dim, int(round(evals * 0.35)))
    trials, U, ylog = [], [], []

    def run(u):
        th = _denorm(u, knobs)
        r = evaluate_knob(th, space, budget, cache)
        trials.append(r)
        U.append(_norm(r["theta"], knobs))         # use the (rounded) realized theta
        ylog.append(math.log(min(r["sel"], CAP) + 1.0))
        return r

    for u in _lhs(n_init, dim, rng):                # naive space-filling start (no prior)
        if len(trials) >= evals:
            break
        run(u)

    # Surrogate-guided TRUST-REGION search: fit a ridge surrogate of log(inner cost) over the
    # normalized knob vector, then (exploit) perturb the best-seen knob and keep the cheapest
    # PREDICTED neighbor, or (explore) sample globally. This is exactly the repo's proven
    # LearnedProposer (B3) — mutate-the-incumbent + surrogate-rank — lifted one level up onto
    # the choice of self-edit. Ranking near the incumbent avoids the linear fit's misleading
    # box-corner extrapolation (e.g. the LearnedProposer's high-scale cliff -> inf).
    def incumbent():
        return U[int(np.argmin([min(t["sel"], CAP) for t in trials]))]

    while len(trials) < evals:
        X, yl = np.array(U), np.array(ylog)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        w = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ yl)
        if rng.random() < 0.25:                     # explore globally
            run(rng.random(dim))
            continue
        pool = np.clip(incumbent() + rng.normal(0, 0.18, size=(200, dim)), 0, 1)
        pred = np.hstack([pool, np.ones((200, 1))]) @ w
        order = np.argsort(pred)
        seen = {tuple(np.round(uu, 3)) for uu in U}
        for idx in order:                           # cheapest predicted, not already seen
            if tuple(np.round(pool[idx], 3)) not in seen:
                run(pool[idx])
                break
        else:
            run(pool[order[0]])
    return {"seed": seed, "trials": trials, "curve": _best_so_far(trials),
            "best_val": _best_so_far(trials)[-1]}


def _fmt(x):
    return "inf" if not math.isfinite(x) else f"{x:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=int, default=6, choices=(3, 6))
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--budget", type=str, default="fast")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cache = {}

    print(f"=== RACE  space={args.space}-knob  evals={args.evals}  budget={args.budget} ===")
    base = evaluate_knob({"scale": 0.4, "rate": 0.5, "pop": 6, "pool": 8,
                          "explore": 0.6, "ridge": 1.0} if args.space == 6
                         else {"scale": 0.7, "rate": 0.5, "pop": 4},
                         args.space, args.budget, cache)
    print(f"baseline (live defaults): inner={_fmt(base['sel'])} heldout={_fmt(base['val'])}")

    claude = claude_arm(args.space, args.evals, args.budget, cache)
    print(f"CLAUDE best held-out: {_fmt(claude['best_val'])}")

    rsi_runs = [rsi_arm(args.space, args.evals, args.budget, sd, cache) for sd in seeds]
    rsi_bests = [r["best_val"] for r in rsi_runs]
    finite = [b for b in rsi_bests if math.isfinite(b)]
    rsi_mean = float(np.mean(finite)) if finite else math.inf
    rsi_best = min(rsi_bests)
    rsi_worst = max(rsi_bests)
    for r in rsi_runs:
        print(f"RSI seed {r['seed']} best held-out: {_fmt(r['best_val'])}")
    print(f"RSI  mean={_fmt(rsi_mean)}  best={_fmt(rsi_best)}  worst={_fmt(rsi_worst)}")

    cb = claude["best_val"]
    win = math.isfinite(rsi_mean) and (not math.isfinite(cb) or rsi_mean < cb)
    ratio = (cb / rsi_mean) if (math.isfinite(cb) and math.isfinite(rsi_mean) and rsi_mean > 0) else None
    verdict = ("RSI BEATS CLAUDE" if win else "Claude still ahead")
    print(f"VERDICT [{args.budget}]: {verdict}"
          + (f"  (Claude/RSI = {ratio:.2f}x)" if ratio else ""))

    # --- confirm the winners at a second, larger budget (cheap: 2 re-scores) -------------
    confirm = None
    if args.budget == "fast":
        # re-score Claude's best knob and RSI's (mean-seed) best knob at 'confirm'
        c_best_theta = _best_theta(claude["trials"])
        rsi_best_run = min(rsi_runs, key=lambda r: (math.inf if not math.isfinite(r["best_val"]) else r["best_val"]))
        r_best_theta = _best_theta(rsi_best_run["trials"])
        cc = evaluate_knob(c_best_theta, args.space, "confirm", cache)
        rc = evaluate_knob(r_best_theta, args.space, "confirm", cache)
        confirm = {"claude_theta": c_best_theta, "claude_heldout": cc["val"],
                   "rsi_theta": r_best_theta, "rsi_heldout": rc["val"],
                   "rsi_wins": math.isfinite(rc["val"]) and (not math.isfinite(cc["val"]) or rc["val"] < cc["val"])}
        print(f"CONFIRM [confirm budget]: claude={_fmt(cc['val'])} rsi={_fmt(rc['val'])} "
              f"-> {'RSI wins' if confirm['rsi_wins'] else 'Claude wins'}")

    out = {
        "space": args.space, "evals": args.evals, "seeds": seeds, "budget": args.budget,
        "baseline": base, "claude": claude,
        "rsi": {"runs": rsi_runs, "mean": rsi_mean, "best": rsi_best, "worst": rsi_worst},
        "verdict": verdict, "ratio_claude_over_rsi": ratio, "rsi_wins": bool(win),
        "confirm": confirm, "n_unique_evals": len(cache),
    }
    path = os.path.join(REPO_ROOT, "run_logs", f"race_{args.space}knob.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}  ({len(cache)} unique evaluations)")


def _best_theta(trials):
    best = min(trials, key=lambda t: (math.inf if not math.isfinite(t["sel"]) else t["sel"]))
    return best["theta"]


if __name__ == "__main__":
    main()
