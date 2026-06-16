"""recursivene.harness — the META level: the machinery that improves the object-level
learner by searching its configuration for lower cost-for-competence.

This package is EDITABLE by the proposer (that is the whole point of the meta-meta
level): the search constants in loop.py and the proposer definitions in proposer.py are
legal targets of self-modification, gated by the protected objective and invariant.
"""

from .space import SEARCH_SPACE, CONFIG_KEYS, mutate, clamp, vec
from .proposer import Proposer, EvolutionaryProposer, LearnedProposer, SeamProposer
from .loop import search, meta_evaluate, MUTATION_SCALE, MUTATION_RATE, POP

__all__ = [
    "SEARCH_SPACE", "CONFIG_KEYS", "mutate", "clamp", "vec",
    "Proposer", "EvolutionaryProposer", "LearnedProposer", "SeamProposer",
    "search", "meta_evaluate", "MUTATION_SCALE", "MUTATION_RATE", "POP",
]
