"""LearnedSelfEditProposer — the upgrade that lets the loop teach ITSELF better than blind heuristics.

The shipped closure proposed self-edits by blind bracketed multipliers (catalog.constant_edits): a fixed
spread that does not learn. This proposer instead MODELS its own improvement landscape — it fits a ridge
surrogate of log(meta_cost) over the normalized editable-constant vector from the loop's OWN history of
tried edits, and proposes the constant values predicted to most lower the meta-objective (trust-region
exploit + a little exploration). It is the LearnedProposer principle (KNOWN #3 / B3) lifted onto the
self-edit search itself — the lever the conundrum points at (notes/10): improve the improver's SEARCH,
not the answers.

Scoring is supplied by the caller, so this module is agnostic to HOW meta_cost is measured:
  * the live closure loop scores each candidate in a sandbox subprocess via SelfModifier.try_edit
    (the protected gate is unchanged — safety does not depend on this proposer);
  * the in-process demonstration (experiments/selfteach.py) scores via harness.loop.meta_evaluate.

Cold start (history < 3): falls back to catalog's bracketed multipliers, so a fresh loop still proposes
sensible edits and the gate's first-accept is reliable. As history accrues, proposals get smarter — the
loop learns to teach itself.
"""

import numpy as np

from .catalog import EDITABLE_CONSTANTS, _fmt, constant_edits

CAP = 1e9  # finite stand-in for an inf (non-generalizing) meta_cost, for the log surrogate


class LearnedSelfEditProposer:
    def __init__(self, ids, ridge=1.0, seed=0):
        self.ids = list(ids)
        self.lo = {c: EDITABLE_CONSTANTS[c][2][0] for c in self.ids}
        self.hi = {c: EDITABLE_CONSTANTS[c][2][1] for c in self.ids}
        self.kind = {c: EDITABLE_CONSTANTS[c][3] for c in self.ids}
        self.X, self.y = [], []
        self.rng = np.random.default_rng(seed)
        self.ridge = ridge

    # --- history -----------------------------------------------------------------------
    def _vec(self, vals):
        return np.array([(float(vals.get(c, 0.5 * (self.lo[c] + self.hi[c]))) - self.lo[c])
                         / (self.hi[c] - self.lo[c]) for c in self.ids])

    def observe(self, vals, meta_cost):
        mc = CAP if (meta_cost is None or not np.isfinite(meta_cost)) else float(meta_cost)
        self.X.append(np.append(self._vec(vals), 1.0))
        self.y.append(np.log(mc + 1.0))

    def _fit(self):
        X, y = np.asarray(self.X), np.asarray(self.y)
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + self.ridge * np.eye(d), X.T @ y)

    def _clampfmt(self, c, raw):
        v = float(np.clip(raw, self.lo[c], self.hi[c]))
        return int(round(v)) if self.kind[c] == "int" else round(v, 4)

    # --- proposals ---------------------------------------------------------------------
    def propose_vectors(self, cur_vals, n=4, tr=0.18):
        """Return up to n FULL constant-value dicts predicted cheapest (for in-process scoring)."""
        if len(self.X) < 3:
            return self._bracket_vectors(cur_vals, n)
        w = self._fit(); cur = self._vec(cur_vals); dim = len(self.ids)
        pool = np.clip(cur + self.rng.normal(0, tr, size=(200, dim)), 0, 1)
        pred = np.hstack([pool, np.ones((200, 1))]) @ w
        out = []
        for idx in np.argsort(pred):
            d = dict(cur_vals)
            for j, c in enumerate(self.ids):
                d[c] = self._clampfmt(c, self.lo[c] + pool[idx][j] * (self.hi[c] - self.lo[c]))
            if d != cur_vals and d not in out:
                out.append(d)
            if len(out) >= n:
                break
        # keep a little exploration
        if self.rng.random() < 0.25:
            d = {**cur_vals, **{c: self._clampfmt(c, self.rng.uniform(self.lo[c], self.hi[c])) for c in self.ids}}
            out.append(d)
        return out

    def propose_edits(self, cur_vals, n=3, tr=0.18):
        """Return up to n SINGLE-constant (id, value) edits predicted cheapest — fits the live loop's
        one-edit-at-a-time gate (coordinate descent guided by the surrogate)."""
        if len(self.X) < 3:
            return self._bracket_edits(cur_vals, n)
        w = self._fit(); cur = self._vec(cur_vals)
        cands = []
        for j, c in enumerate(self.ids):
            for v in np.clip(cur[j] + self.rng.normal(0, tr, size=6), 0, 1):
                u = cur.copy(); u[j] = v
                pred = float(np.append(u, 1.0) @ w)
                val = self._clampfmt(c, self.lo[c] + v * (self.hi[c] - self.lo[c]))
                if val != cur_vals.get(c):
                    cands.append((pred, c, val))
        cands.sort(key=lambda t: t[0])
        seen, out = set(), []
        for _, c, val in cands:
            if (c, val) in seen:
                continue
            seen.add((c, val)); out.append((c, val))
            if len(out) >= n:
                break
        return out

    # --- cold-start fallbacks (blind bracketed multipliers, via the catalog) -----------
    def _bracket_edits(self, cur_vals, n):
        out = []
        for c in self.ids:
            stage = EDITABLE_CONSTANTS[c][4]
            for e in constant_edits(cur_vals, self.rng, stage=stage):
                if e["id"] == c:
                    out.append((c, _val_of(e)))
        self.rng.shuffle(out)
        return out[:n] if out else []

    def _bracket_vectors(self, cur_vals, n):
        out = []
        for c, val in self._bracket_edits(cur_vals, n):
            out.append({**cur_vals, c: val})
        return out


def _val_of(edit):
    """Parse the numeric value out of a catalog edit's replace string ('NAME = 0.7000' -> 0.7)."""
    return float(edit["replace"].split("=")[-1].strip())
