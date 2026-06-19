"""The config-level improvement ratchet (the META loop) and the META-OBJECTIVE used to
score edits to the harness itself.

The three module constants below — MUTATION_SCALE, MUTATION_RATE, POP — are the editable
knobs of the improvement operator. At the meta-meta level the proposer edits THIS FILE to
change them (or rewrites `search`), and the edit is kept only if `meta_evaluate` shows the
edited harness finds cheaper models on held-out worlds. That is the crucial move: a change
to the improver is judged by running the improver, not by scoring one config.
"""

import numpy as np

from ..objective import evaluate, TAU
from ..invariant import invariant_score, HELDOUT_SEEDS
from .space import mutate

# --- editable improvement-operator knobs (meta-meta self-edit targets) -------------
# Deliberately generic defaults: the meta level should DISCOVER better search
# hyperparameters here, exactly as stage 1 discovers better model hyperparameters. These
# three lines are what stage-2 self-edits rewrite in harness/loop.py.
MUTATION_SCALE = 0.7000
MUTATION_RATE = 0.4500
POP = 4


def default_proposer():
    """Build the evolutionary proposer FROM the editable module constants, so a self-edit
    to MUTATION_SCALE/MUTATION_RATE/POP actually changes how the search behaves (and thus
    the measured meta-objective). Without this the constants would be dead code."""
    from .proposer import EvolutionaryProposer
    return EvolutionaryProposer(pop=POP, scale=MUTATION_SCALE, rate=MUTATION_RATE)


def search(proposer, init_config, generations=6, steps=2500, seeds=(0, 1, 2),
           tau=TAU, verbose=False, on_gen=None):
    """Run the propose -> evaluate -> select ratchet on INNER worlds. Returns
    (best_config, best_eval, history). Only strict improvements are accepted, so the
    cost trajectory is monotone non-increasing by construction."""
    best_config = dict(init_config)
    best_eval = evaluate(best_config, which="inner", steps=steps, seeds=seeds, tau=tau)
    history = [{"gen": 0, "config": dict(best_config), **best_eval}]
    if on_gen:
        on_gen(0, best_config, best_eval)

    for gen in range(1, generations + 1):
        candidates = proposer.propose(best_config, history, np.random.default_rng(1000 + gen))
        scored = []
        for cfg in candidates:
            ev = evaluate(cfg, which="inner", steps=steps, seeds=seeds, tau=tau)
            scored.append((ev["cost"], ev["n_params"], cfg, ev))
        scored.sort(key=lambda t: (t[0], t[1]))    # lowest cost, then fewest params
        cost, _, cfg, ev = scored[0]
        if cost < best_eval["cost"]:
            best_config, best_eval = cfg, ev
        history.append({"gen": gen, "config": dict(best_config), **best_eval})
        if on_gen:
            on_gen(gen, best_config, best_eval)
        if verbose:
            print(f"  gen {gen:>2} cost={best_eval['cost']:.3e} D={best_config['n_features']}")
    return best_config, best_eval, history


def meta_evaluate(proposer_factory, init_config, generations=4, steps=2000,
                  inner_seeds=(0, 1), heldout_steps=2500):
    """THE META-OBJECTIVE. Run the harness's search, then measure the HELD-OUT cost of the
    config it found. Lower = the harness/proposer is better at producing cheap, GENERALIZING
    models. This is what gates self-edits to harness.py and proposer.py: an edit is kept only
    if it lowers this number without degrading the protected invariant.

    `proposer_factory` is a 0-arg callable returning a fresh proposer, or None/returns-None
    to use default_proposer() built from the editable module constants. Returns dict with
    meta_cost (held-out cost of found config) and invariant.
    """
    proposer = proposer_factory() if proposer_factory else None
    if proposer is None:
        proposer = default_proposer()
    best_config, _, _ = search(proposer, init_config, generations=generations,
                               steps=steps, seeds=inner_seeds)
    held = evaluate(best_config, which="heldout", steps=heldout_steps, seeds=HELDOUT_SEEDS)
    return {
        "meta_cost": held["cost"],                 # held-out cost of the harness's best find
        "invariant": invariant_score(best_config, steps=heldout_steps),
        "found_config": best_config,
        "found_params": held["n_params"],
    }
