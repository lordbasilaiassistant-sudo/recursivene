"""Open-endedness module: invent new problems at the learnable frontier.

A fixed test saturates and then lies (Goodhart): once the learner aces the current
activities, "samples-to-competence" stops measuring anything — every policy looks equally
good against a ceiling it already hit. The cure is to keep MOVING the test: when the
repertoire of mastered activities is full, the generator proposes a NEW activity that is

  (a) NOVEL          — far in frequency from everything already present,
  (b) UNSOLVED       — the learner cannot do it yet at its current budget,
  (c) NOW-SOLVABLE   — but its error genuinely falls with more samples (not flat noise),
  (d) NON-FORGETTING — adding it leaves the already-mastered activities mastered.

Gate (c) is the open-ended analogue of the agent's own noisy-TV rejection: a frequency
too high for the model's capacity flatlines at the E[sin^2]=0.5 floor regardless of
samples (verified empirically: w>=~24 at gamma=8,D=96), exactly like a noise arm sitting
at its variance. Proposing it would be proposing noise. So the generator REFUSES it, the
same way the LP policy refuses the noisy TV — Goodhart-proofing the frontier itself.

References:
  Schmidhuber 2011, PowerPlay (arXiv:1112.5309): only add a task once you can solve it
    without forgetting the rest.
  Wang et al. 2019, POET (arXiv:1901.01753): co-evolve environments and solvers at the
    frontier; gate new environments by "minimal criterion" + novelty.
  Lehman & Stanley 2011, Novelty Search: reward behavioral novelty, not a fixed objective.
"""

from .generator import propose_problems, Problem, frontier_summary
from .abstraction import mint_abstraction

__all__ = ["propose_problems", "Problem", "frontier_summary", "mint_abstraction"]
