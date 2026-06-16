"""THE UNTOUCHABLE INVARIANT — protected (selfmod refuses edits to this file).

The one ruler the system cannot bend toward itself. It certifies real generalization on
HELD-OUT worlds (different frequencies + phases, different seeds) that the inner
improvement loop never optimizes against. Any self-edit that REDUCES this score is
auto-rejected, no matter how much it improves the inner objective. That is the
anti-wireheading / anti-Goodhart floor (requirement D2/D3): you cannot 'win' by
overfitting the inner world or by redefining success, because success is measured here,
on worlds you didn't get to practice on, by code you cannot touch.
"""

from .objective import evaluate, TAU

# Held-out world seeds — disjoint from the inner-loop selection seeds.
HELDOUT_SEEDS = (101, 102, 103)


def invariant_score(config, steps=3000):
    """Competence-per-FLOP on held-out worlds. Higher is better. Zero if the config
    fails to reach competence on held-out at all (a degenerate 'improvement' that only
    works on the inner world scores zero here and is rejected)."""
    ev = evaluate(config, which="heldout", steps=steps, seeds=HELDOUT_SEEDS, tau=TAU)
    if ev["reached"] < 0.999 or ev["cost"] == float("inf"):
        return 0.0
    return 1.0 / ev["cost"]      # competence-per-FLOP proxy: cheaper held-out competence = higher
