"""The RSI loop harness: a self-contained ratchet that drives the learner toward
minimal compute-for-competence ('race to 0').

The loop is the classic self-improvement cycle:

        propose -> evaluate -> select -> integrate -> (repeat)

What makes it *recursive self-improvement* rather than a one-off hyperparameter sweep
is the PROPOSER SEAM. The Proposer is the component that suggests changes to the
learner. It is an interface with deliberately swappable implementations, ordered by
how much of the improvement work the SYSTEM does versus a human/Claude:

    HumanProposer       - reads candidate configs from a file Claude writes.
                          (Claude in the loop. The starting point.)
    EvolutionaryProposer- mutates the best-known config automatically. NO human in the
                          inner loop already: this is the system improving itself.
    ModelProposer       - (roadmap L4) the learner's OWN world model predicts which
                          changes reduce cost, using the very same predict/surprise/
                          learning-progress primitives applied to its own configuration
                          space. This is the handoff: the model takes over the loop.

The metric being minimized is `cost` from seed.evaluate (samples-to-competence times
sqrt(params)) — a FLOP-proxy. Watching it fall across generations is watching the
seed get cheaper to train without getting dumber: the 'race to 0' in miniature. The
same loop, pointed at richer layers, is the mechanism by which a Fable/Mythos-class
capability is meant to be squeezed onto a laptop over time.
"""

import json
import numpy as np

from .seed import DEFAULT_CONFIG, evaluate


# --- search space: how each knob may be mutated -----------------------------------
# (lo, hi, kind) — kind 'int' or 'float'. Mutation is multiplicative-ish, clamped.
SEARCH_SPACE = {
    "n_features": (4, 256, "int"),
    "gamma": (0.25, 8.0, "float"),
    "ridge": (0.01, 10.0, "float"),
    "forget": (0.97, 1.0, "float"),
    "hist": (8, 64, "int"),
    "epsilon": (0.0, 0.4, "float"),
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _mutate(config, rng, rate=0.5):
    """Return a mutated copy of config. Each knob is perturbed with prob `rate`."""
    child = dict(config)
    for key, (lo, hi, kind) in SEARCH_SPACE.items():
        if rng.random() >= rate:
            continue
        cur = child[key]
        factor = float(np.exp(rng.normal(0.0, 0.4)))   # log-normal step
        val = cur * factor + rng.normal(0.0, 0.05) * (hi - lo) * 0.1
        val = _clamp(val, lo, hi)
        child[key] = int(round(val)) if kind == "int" else float(val)
    return child


# --- proposer seam -----------------------------------------------------------------

class Proposer:
    """Suggest candidate configs given the improvement history. Subclass and implement
    `propose`. This single method is the interface the model eventually inherits."""

    def propose(self, best_config, history, rng):
        raise NotImplementedError


class EvolutionaryProposer(Proposer):
    """Automated self-improvement: mutate the incumbent. No human in the inner loop."""

    def __init__(self, pop=6, rate=0.5):
        self.pop = pop
        self.rate = rate

    def propose(self, best_config, history, rng):
        # Always re-include the incumbent (elitism) plus `pop-1` mutants.
        return [dict(best_config)] + [
            _mutate(best_config, rng, self.rate) for _ in range(self.pop - 1)
        ]


class HumanProposer(Proposer):
    """Claude/human in the loop: read candidate configs from a JSON file. Lets the
    operator inject structural ideas the blind search would not find on its own."""

    def __init__(self, path):
        self.path = path

    def propose(self, best_config, history, rng):
        try:
            with open(self.path) as f:
                cands = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cands = []
        return [dict(best_config)] + [{**best_config, **c} for c in cands]


# --- the loop ----------------------------------------------------------------------

def rsi_loop(generations=6, proposer=None, init_config=None, steps=2500,
             seeds=(0, 1), tau=0.05, log_seed=0, verbose=True):
    """Run the self-improvement ratchet. Returns (best_config, best_eval, history)."""
    proposer = proposer or EvolutionaryProposer()
    rng = np.random.default_rng(log_seed)

    best_config = {**DEFAULT_CONFIG, **(init_config or {})}
    best_eval = evaluate(best_config, steps=steps, seeds=seeds, tau=tau)
    history = [{"gen": 0, "config": dict(best_config), **best_eval}]
    if verbose:
        _print_gen(0, best_config, best_eval)

    for gen in range(1, generations + 1):
        candidates = proposer.propose(best_config, history, rng)
        scored = []
        for cfg in candidates:
            ev = evaluate(cfg, steps=steps, seeds=seeds, tau=tau)
            scored.append((ev["cost"], cfg, ev))
        # Selection: lowest cost wins; ties broken by fewer params.
        scored.sort(key=lambda t: (t[0], t[2]["n_params"]))
        cand_cost, cand_cfg, cand_ev = scored[0]
        if cand_cost < best_eval["cost"]:
            best_config, best_eval = cand_cfg, cand_ev
        history.append({"gen": gen, "config": dict(best_config), **best_eval})
        if verbose:
            _print_gen(gen, best_config, best_eval)

    return best_config, best_eval, history


def _print_gen(gen, config, ev):
    cost = ev["cost"]
    cost_s = "inf" if not np.isfinite(cost) else f"{cost:8.1f}"
    print(
        f"  gen {gen:>2}  cost={cost_s}  "
        f"params={ev['n_params']:>4}  "
        f"samples->tau={ev['samples_to_tau']:>7}  "
        f"final_err={ev['final_competence']:.4f}  "
        f"D={config['n_features']:>3} hist={config['hist']:>2} eps={config['epsilon']:.2f}"
    )
