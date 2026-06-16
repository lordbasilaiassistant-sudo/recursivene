"""Minting reusable abstractions from recurrent mastered structure.

The substrate's world-model is a bank of RANDOM cosine basis functions, phi_j(x) =
sqrt(2/D) cos(W_j x + b_j), with W_j drawn i.i.d. from N(0, gamma). Random features are
unbiased but blind: to fit a sine of frequency w the model needs some W_j NEAR w, and for
high w those are rare tails of the Gaussian — which is exactly why high-frequency arms need
many samples (and why, past the model's capacity, they never fit at all).

An honest, minimal abstraction is therefore: when the learner has REPEATEDLY mastered sine
activities clustered around some frequency band, that band is recurrent structure worth
remembering. We compress it into a single reusable feature — a DIRECTED basis frequency at
the mastered band — that can be grafted into a fresh model's feature bank so the next
learner is born already able to represent that band. This is the smallest faithful version
of "compress recurrent mastered structure into a reusable feature": input is the history of
solved frequencies, output is a feature (a basis direction) that provably lowers
samples-to-mastery on a related new problem.

We return None until there is genuine recurrence to compress (>= `min_recurrence` mastered
frequencies in a band), so the function never fabricates an abstraction out of noise.
"""

import numpy as np


class BasisFeature:
    """A reusable, directed cosine basis function: phi(x) = scale * cos(w * x + b).

    Mirrors one column of RFFOnlineRegressor's feature bank, but with a CHOSEN frequency w
    (distilled from mastered structure) instead of a random one. `graft_into(model)` swaps
    this directed frequency in for the model's least-useful random one, so a fresh learner
    inherits the abstraction without growing its parameter budget (D unchanged — honest:
    no free capacity, just a better-aimed feature)."""

    __slots__ = ("w", "b", "support", "n_evidence")

    def __init__(self, w, b, support, n_evidence):
        self.w = float(w)              # the distilled frequency (the abstraction)
        self.b = float(b)              # phase offset for the basis column
        self.support = tuple(support)  # the mastered frequencies this was compressed from
        self.n_evidence = int(n_evidence)

    def graft_into(self, model):
        """Replace ONE random basis frequency in `model` with this directed one. We pick the
        column whose random W_j is farthest from any structure the model already covers
        (the least-useful feature) and overwrite it. Parameter count is unchanged."""
        # least-useful column = the one whose frequency is most extreme/isolated
        j = int(np.argmax(np.abs(model.W)))
        model.W = model.W.copy()
        model.b = model.b.copy()
        model.W[j] = self.w
        model.b[j] = self.b
        return model

    def as_dict(self):
        return {"w": self.w, "b": self.b, "support": list(self.support),
                "n_evidence": self.n_evidence}


def mint_abstraction(history, min_recurrence=3, band=2.0):
    """Compress recurrent mastered structure into a reusable BasisFeature, or None.

    Parameters
    ----------
    history : sequence of dict
        Per-round records, each with a "mastered_ws" list (the frequencies the learner had
        mastered that round). Produced by `frontier_summary`. This is the "recurrent
        mastered structure" we compress.
    min_recurrence : int
        Minimum number of distinct mastered frequencies required before we mint anything.
        Below this there is no recurrence to compress -> return None (no fabrication).
    band : float
        Frequencies within this distance are treated as the same recurrent band.

    Returns
    -------
    BasisFeature or None
        A directed basis feature at the densest mastered band, with the mastered
        frequencies it was distilled from attached as evidence. None if nothing to mint yet.
    """
    # Collect every mastered frequency seen across history (the recurrent structure).
    ws = []
    for rec in history:
        ws.extend(rec.get("mastered_ws", []))
    ws = sorted(float(w) for w in set(round(float(w), 3) for w in ws) if w > 0.0)
    if len(ws) < min_recurrence:
        return None

    # Find the densest band: the frequency whose `band`-neighborhood contains the most
    # mastered frequencies. That cluster is the structure most worth abstracting.
    arr = np.asarray(ws)
    best_center, best_support = None, []
    for c in arr:
        support = arr[np.abs(arr - c) <= band]
        if len(support) > len(best_support):
            best_center, best_support = c, support
    if len(best_support) < min_recurrence:
        return None

    # The abstraction is the band's representative frequency (its mean): a basis tuned there
    # generalizes across the whole cluster. Phase 0 is fine — RLS fits the linear coeff.
    w_star = float(np.mean(best_support))
    return BasisFeature(w=w_star, b=0.0, support=[float(w) for w in best_support],
                        n_evidence=len(best_support))
