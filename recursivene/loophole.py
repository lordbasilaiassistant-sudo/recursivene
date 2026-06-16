"""WALL DIAGNOSTIC — a stall-RESPONSE POLICY named after a taxonomy of why apparent limits break.

HONEST FRAMING (read this before trusting any output):
    A WALL is an APPARENT limit the entity hit this season: either `cost_to_know` spiked, or the
    frontier was UNREACHABLE at current capacity (`cost == inf`). This module classifies the stall
    into one of four LOOPHOLE CATEGORIES — historically the four ways an apparent hard limit turns
    out to be breakable — and emits the matching MOVE the entity already has machinery for.

    It is NOT a physics solver and it does NOT claim any given limit is fake. Every non-None
    classification is explicitly a HYPOTHESIS about which loophole (if any) applies (so every output
    carries is_hypothesis=True). Some walls are GENUINE: the honest output for those is category #2
    ("this limit is real — grow AROUND it, not through it"; real_wall_acknowledged=True), or
    category None ("this was not a wall").

The four loophole categories (why apparent limits break) and the MOVE each drives:
    #1 hidden_ledger_term  — a resource you already paid for but aren't counting (the BANK).
                             MOVE use_bank: re-use banked discovered freqs before growing.
    #2 missing_structure   — a symmetry/structure the system GENUINELY lacks; the limit is REAL.
                             MOVE grow_capacity: widen the representation, route AROUND the wall.
    #3 wrong_domain        — a fixed/coarse representation applied past where it holds (the RFF tail
                             is carrying structure that should be discovered/learned).
                             MOVE richer_representation: discover() the freq so it leaves the RFF tail.
    #4 proxy_confusion     — optimizing a PROXY (cost_to_know = samples-to-tau), not the protected
                             invariant (held-out competence under objective.TAU).
                             MOVE re_examine_metric: defer to the protected ruler; do NOT spend
                             capacity reacting to a proxy artifact.

The MOVES are ADVISORY — they only re-order EXISTING primitives (discover/grow/re-evaluate). This
module never edits the objective and never redefines success (no wireheading).
"""

import numpy as np

from .objective import TAU


def _median_cost(rings):
    """Median finite cost_to_know over the supplied window of rings (the SAME notion status() uses)."""
    known = [r["cost_to_know"] for r in rings
             if r.get("cost_to_know") is not None and np.isfinite(r["cost_to_know"])]
    if not known:
        return None
    return float(np.median(known))


def _dominant_freq(encoder):
    """Cheap estimate of the dominant frequency currently sitting in the encoder's rolling buffer.

    This is the structure the LAST observations carried — i.e. what the spike was paying to learn.
    We reuse the encoder's own candidate grid + buffer so this never re-implements the discoverer.
    Returns None if the buffer is too thin to be meaningful.
    """
    bx, by = encoder.bx, encoder.by
    if len(bx) < 60:
        return None
    X = np.asarray(bx)
    Y = np.asarray(by)
    Y = Y - Y.mean()
    cands = encoder.candidates
    S = np.sin(np.outer(cands, X)) @ Y
    C = np.cos(np.outer(cands, X)) @ Y
    energy = S * S + C * C
    return float(cands[int(np.argmax(energy))])


class WallDiagnostic:
    """The 4-category classifier + matching-move emitter (spec PIECE 1).

    Stateless: call `diagnose(ring, prior_rings, encoder, heldout_probe=None)` once per season.
    `prior_rings` is the last K rings BEFORE `ring` (used for the median-cost baseline). `encoder`
    is the live SpectralEncoder (its banked `freqs`, `bx/by` buffer, `fmax`, `min_sep`).
    `heldout_probe` (optional) is a callable -> held-out MSE under the protected ruler; if provided
    it lets category #4 fire honestly (proxy spiked but the protected quantity is intact).
    """

    SPIKE_MULT = 3.0   # the SAME spike rule status() uses: cost > 3 * median

    # Events the entity stamps when a season went UNREACHABLE at current capacity and had to GROW
    # (encoder.grow) before it could make the unknown known. Post-hoc, this is the ONLY trace of the
    # cost==inf wall that _season already resolved — so we read it as the category-#2 signal.
    GREW_EVENTS = ("grew+knew",)

    def diagnose(self, ring, prior_rings, encoder, heldout_probe=None):
        cost = ring.get("cost_to_know")
        med = _median_cost(prior_rings)
        event = ring.get("event", "")

        # Did total_known still advance this season? (proxy-vs-protected guard for #4)
        prev_known = prior_rings[-1]["total_known"] if prior_rings else 0
        advanced = ring.get("total_known", prev_known) > prev_known

        # ---- decision order is deterministic (spec section "Classifier") ----

        # 0) UNREACHABLE-THEN-GREW (#2), read from the event. The live entity's _season resolves a
        #    cost==inf frontier IN-PLACE by growing, so after the season the only honest trace of the
        #    genuine missing-structure wall is event=='grew+knew'. The pool just deepened / a novel
        #    primitive appeared and capacity was exhausted: this is the REAL-wall branch. (For a
        #    hand-crafted ring that still carries cost==inf, branch (3) below catches it identically.)
        if event in self.GREW_EVENTS:
            return self._mk(2, "missing_structure", "grow_capacity",
                            f"frontier was UNREACHABLE at prior capacity (event={event}); capacity "
                            f"exhausted -> encoder grew to rep={ring.get('rep_size')} -> genuine "
                            f"floor, routed AROUND it by adding form",
                            real_wall=True)

        # 1) PROXY GUARD FIRST (#4). A spike in cost_to_know (a proxy) while the PROTECTED quantity
        #    (held-out competence under objective.TAU) is intact and total_known advanced. Run first
        #    so the entity never burns capacity reacting to a proxy artifact.
        spiked = (cost is not None and np.isfinite(cost) and med is not None
                  and cost > self.SPIKE_MULT * med)
        if spiked and advanced:
            heldout = None
            if heldout_probe is not None:
                try:
                    heldout = float(heldout_probe())
                except Exception:
                    heldout = None
            protected_intact = (heldout is not None and heldout <= TAU)
            if protected_intact:
                return self._mk(4, "proxy_confusion", "re_examine_metric",
                                f"cost {cost} > {self.SPIKE_MULT:.0f}*median {med:.0f} but total_known "
                                f"advanced and held-out MSE {heldout:.4f} <= TAU {TAU}: proxy spiked, "
                                f"protected quantity intact",
                                real_wall=False)

        # 2) REACHABLE SPIKE (#1 vs #3). Finite cost but spiking. Inspect the bank.
        if spiked:
            dom = _dominant_freq(encoder)
            if dom is not None:
                near_bank = bool(np.any(np.abs(np.asarray(encoder.freqs) - dom) < encoder.min_sep))
                if near_bank:
                    # #1: you already PAID for this structure — it lives in the bank.
                    return self._mk(1, "hidden_ledger_term", "use_bank",
                                    f"cost {cost} > {self.SPIKE_MULT:.0f}*median {med:.0f}; dominant "
                                    f"freq {dom:.2f} within min_sep {encoder.min_sep} of a banked "
                                    f"freq -> resource already counted",
                                    real_wall=False)
                if dom <= encoder.fmax:
                    # #3: reachable (within fmax) but NOT yet discovered — the RFF tail is paying.
                    return self._mk(3, "wrong_domain", "richer_representation",
                                    f"cost {cost} > {self.SPIKE_MULT:.0f}*median {med:.0f}; dominant "
                                    f"freq {dom:.2f} within fmax {encoder.fmax:.1f} but not banked -> "
                                    f"fixed RFF tail carrying it, discover() to migrate it",
                                    real_wall=False)
            # spike but no usable buffer fingerprint: still a hidden-ledger candidate (cheapest move
            # first). Re-use the bank before paying to grow.
            return self._mk(1, "hidden_ledger_term", "use_bank",
                            f"cost {cost} > {self.SPIKE_MULT:.0f}*median {med:.0f}; buffer too thin to "
                            f"fingerprint -> try banked structure before growing",
                            real_wall=False)

        # 3) UNREACHABLE (#2). cost == inf / None at current capacity: the structure GENUINELY isn't
        #    here. This is the honest "the wall is REAL — grow AROUND it" branch.
        if cost is None:
            return self._mk(2, "missing_structure", "grow_capacity",
                            "frontier UNREACHABLE at current capacity (cost==inf); missing structure "
                            "is novel (not banked) -> genuine floor, route around it by adding form",
                            real_wall=True)

        # 4) NO WALL — explicit. Finite cost, no spike.
        return self._mk(None, None, "none",
                        f"cost {cost} <= {self.SPIKE_MULT:.0f}*median "
                        f"{('%.0f' % med) if med is not None else 'n/a'}: not a wall",
                        real_wall=False)

    @staticmethod
    def _mk(category, name, move, signature, real_wall):
        return {
            "category": category,
            "name": name,
            "signature": signature,
            "move": move,
            "is_hypothesis": True,            # every classification is a HYPOTHESIS, by construction
            "real_wall_acknowledged": bool(real_wall),
        }
