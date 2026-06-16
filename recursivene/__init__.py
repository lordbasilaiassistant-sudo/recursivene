"""RecursiveNe — a world-model-first, least-training-intensive, self-teaching learner
whose self-improvement loop is closed over itself.

Three levels, one objective:
  * OBJECT  — a predictive world model learned online from surprise        (model, seed)
  * META    — a harness/proposer that makes the object cheaper to learn    (harness, proposer)
  * META-META — the proposer edits the proposer / the harness itself       (selfmod)

The objective (cost-for-competence) and the held-out invariant are PROTECTED and never
editable (objective.py, invariant.py). Everything that computes toward the objective is
editable by the proposer, gated by sandboxed evaluation against the untouchable invariant.

Nothing here uses backprop, replay buffers, epochs, or a labeled corpus.
"""

from .world import World, Activity, make_world
from .model import RFFOnlineRegressor
from .agent import RegionLearner, choose, POLICIES
from .objective import competence, evaluate, TAU
from .seed import run, DEFAULT_CONFIG

__all__ = [
    "World", "Activity", "make_world",
    "RFFOnlineRegressor",
    "RegionLearner", "choose", "POLICIES",
    "competence", "evaluate", "TAU",
    "run", "DEFAULT_CONFIG",
]
