"""THE OBJECTIVE — protected. The proposer may not edit this file (selfmod refuses it).

This module defines what 'competence' and 'cost' MEAN. Everything else in the package is
machinery that computes toward this scalar and is therefore editable; this definition is
the anchor that never changes. Fixing the ruler while letting everything that moves the
needle be rewritten is what keeps the self-improvement loop from wireheading (you cannot
'improve' by redefining success).

  competence(world, learner)  : mean squared error on the learnable activities (lower better)
  cost-for-competence         : FLOPs spent until competence reaches tau (lower better) -> race to 0
  evaluate(config, which)     : trains a fresh learner and reports cost-for-competence,
                                on the INNER world set (selection) or HELD-OUT set (reward/invariant)
"""

import numpy as np

from .world import make_world

TAU = 0.05
EVAL_EVERY = 20


def competence(world, learner, grid=21):
    """WORST-CASE squared error over the world's LEARNABLE activities, on a fixed grid
    against ground truth (noise excluded). Worst-case, not mean: 'competent' means good
    at ALL of them, so reaching threshold REQUIRES mastering the hardest activity — which
    a uniform sampler under-feeds and a learning-progress sampler concentrates on. Mean
    error would let easy activities mask an unlearned hard one and reward spreading thin."""
    xs = np.linspace(-1.0, 1.0, grid)
    worst = 0.0
    for r in range(world.K):
        if not world.learnable[r]:
            continue
        m = learner.models[r]
        e = float(np.mean([(world.truth(r, x) - m.predict(x)) ** 2 for x in xs]))
        if not np.isfinite(e):     # guard RLS numerical blow-up from poisoning the metric
            e = 1e6
        worst = max(worst, e)
    return worst


def _first_at(values, comp, gate):
    hits = [i for i, c in enumerate(comp) if c <= gate]
    return values[hits[0]] if hits else None


def evaluate(config=None, which="inner", steps=3000, seeds=(0, 1, 2), tau=TAU):
    """Cost-for-competence (FLOPs to reach tau), averaged over worlds. The single scalar
    the whole system minimizes. `which`='inner' for selection, 'heldout' for the reward
    and the untouchable invariant (anti-gaming: the two sets are different worlds)."""
    from .seed import run, DEFAULT_CONFIG    # lazy import: run() is editable machinery
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    flops_to_tau, s2t, finals, noise_fracs, reached = [], [], [], [], []
    nparams = ram = None
    for sd in seeds:
        world = make_world(which, seed=sd)
        _, learner, log = run(cfg, world=world, steps=steps, seed=sd)
        nparams, ram = log["n_params"], log["ram_floats"]
        comp = log["competence"]
        # SUSTAINED competence: the config must still be competent at the END of the run,
        # not merely touch tau once. Without this, the search games the metric with tiny
        # unstable models that dip below tau then diverge (RLS blow-up at small D).
        sustained = len(comp) > 0 and comp[-1] <= tau
        f = _first_at(log["flops"], comp, tau) if sustained else None
        s = _first_at(log["steps"], comp, tau) if sustained else None
        flops_to_tau.append(f if f is not None else np.inf)
        s2t.append(s if s is not None else np.inf)
        reached.append(1.0 if f is not None else 0.0)
        finals.append(comp[-1])
        noise_fracs.append(float(np.isin(log["region_seq"], log["noise_indices"]).mean()))

    return {
        "reached": float(np.mean(reached)),
        "cost": float(np.mean(flops_to_tau)),     # inf if any seed failed
        "samples_to_tau": float(np.mean(s2t)),
        "final_competence": float(np.mean(finals)),
        "noise_fraction": float(np.mean(noise_fracs)),
        "n_params": int(nparams),
        "ram_floats": int(ram),
    }
