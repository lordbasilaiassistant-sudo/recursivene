"""Proposers — the seam through which improvement is generated, and the one the model
eventually inherits.

A Proposer suggests candidates given the improvement history. The interface is tiny on
purpose: `propose` is the single method the model assumes when it takes over the loop.
Implementations are ordered by how much of the work the SYSTEM does vs a human/search:

  EvolutionaryProposer  - blind mutation of the incumbent. No human in the inner loop,
                          but no learning either (this is the 'weak RSI' baseline).
  LearnedProposer       - fits a surrogate model of cost over config-space from the run's
                          own history (using RecursiveNe's OWN world-model primitive,
                          RFF+RLS) and proposes configs predicted to be cheap. The system
                          modelling its own improvement landscape — the meta level getting
                          better at improving, which is what must beat blind mutation (B3).
  SeamProposer          - reads candidates from a file a human/Claude/the model writes.
                          The literal hand-off seam: drop richer proposals here and they
                          enter the loop under the same protected gate as everything else.
"""

import json
import numpy as np

from ..model import RFFOnlineRegressor
from .space import SEARCH_SPACE, mutate, vec


class Proposer:
    def propose(self, best_config, history, rng):
        raise NotImplementedError


class EvolutionaryProposer(Proposer):
    """Blind mutation around the incumbent (elitism + mutants)."""

    def __init__(self, pop=6, scale=0.4, rate=0.5):
        self.pop = pop
        self.scale = scale
        self.rate = rate

    def propose(self, best_config, history, rng):
        return [dict(best_config)] + [
            mutate(best_config, rng, self.scale, self.rate) for _ in range(self.pop - 1)
        ]


class LearnedProposer(Proposer):
    """Surrogate-guided proposal — the system modelling its OWN improvement landscape.

    Fits a ridge regression of log-cost over the FULL normalized config vector from the run's
    own history (not a scalar collapse — that is what made the first version useless), then
    oversamples mutants and keeps the cheapest-PREDICTED ones, with EXPLORE_FRAC kept random so
    the surrogate keeps seeing new regions. A learned proposal distribution should beat blind
    mutation once it has a few data points (B3) — and its three constants below are the
    meta-meta self-edit targets for stage 3 (the proposer improving the proposer)."""

    POOL_FACTOR = 16
    EXPLORE_FRAC = 0.6000
    SURROGATE_RIDGE = 1.0        # ridge on the surrogate fit

    def __init__(self, pop=6, scale=0.4, rate=0.5, seed=0):
        self.pop = pop
        self.scale = scale
        self.rate = rate
        self.X, self.y = [], []
        self._fitted = 0
        self.w = None

    def _refit(self, history):
        for h in history[self._fitted:]:
            c = h.get("cost", np.inf)
            if np.isfinite(c) and "config" in h:
                self.X.append(np.concatenate([vec(h["config"]), [1.0]]))   # +bias
                self.y.append(np.log(c + 1.0))
        self._fitted = len(history)
        if len(self.X) >= 3:
            X, y = np.asarray(self.X), np.asarray(self.y)
            d = X.shape[1]
            self.w = np.linalg.solve(X.T @ X + self.SURROGATE_RIDGE * np.eye(d), X.T @ y)

    def _predict(self, cfg):
        if self.w is None:
            return 0.0
        return float(np.concatenate([vec(cfg), [1.0]]) @ self.w)

    def propose(self, best_config, history, rng):
        self._refit(history)
        n_explore = max(1, int(round(self.pop * self.EXPLORE_FRAC)))
        cands = [dict(best_config)]
        pool = [mutate(best_config, rng, self.scale, self.rate)
                for _ in range(self.POOL_FACTOR * self.pop)]
        pool.sort(key=self._predict)                       # cheapest predicted first
        cands += pool[: self.pop - 1 - n_explore]
        cands += [mutate(best_config, rng, self.scale * 1.5, 0.7) for _ in range(n_explore)]
        return cands


class SeamProposer(Proposer):
    """Read candidate configs from a JSON file (a list of partial-config dicts). The
    hand-off point: a human, Claude, or the model writes proposals here."""

    def __init__(self, path):
        self.path = path

    def propose(self, best_config, history, rng):
        try:
            with open(self.path) as f:
                cands = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cands = []
        return [dict(best_config)] + [{**best_config, **c} for c in cands]
