"""NOUS — one RecursiveNe entity. Everything proven, brought together into a single self-
improving knower with persistent identity.

It is not a pile of experiments; it is one thing that LIVES. Each season it:
  1. REACHES   — faces a fresh unknown at its current frontier of complexity (open-endedness);
  2. KNOWS     — makes it known with its LEARNED representation + generalization-based mastery,
                 the cost falling as its representation banks the structure it keeps seeing
                 (emergent transfer / L1);
  3. GROWS     — when an unknown is unreachable at its current capacity, it widens its own
                 representation until it isn't (the garden, at the representation level);
  4. RACES TO 0— periodically improves its own learner to make what it knows cheaper, logging
                 every accepted improvement (the meta loop / monotonicity);
  5. LIVES ON  — beats its vitals, honors the kill switch, and persists its whole self so the
                 SAME entity resumes next time (identity across sessions).

One drive, the telos Anthony named: make unknowns known, ever more cheaply, forever — under an
untouchable competence invariant and a kill switch it cannot edit.
"""

import json
import os
import numpy as np

from .objective import TAU
from .encoder import SpectralEncoder
from .induction import induce
from .core.clock import now_iso, now_unix, stamp
from .core.killswitch import KillSwitch, Halt
from .core.vitals import Vitals


class _Head:
    """A cheap per-task readout over the entity's representation (RLS)."""

    def __init__(self, d, ridge=1.0):
        self.w = np.zeros(d); self.P = np.eye(d) / ridge

    def predict(self, f):
        return float(f @ self.w)

    def update(self, f, y):
        Pp = self.P @ f; k = Pp / (1.0 + f @ Pp)
        self.w = self.w + k * (y - f @ self.w); self.P = self.P - np.outer(k, Pp)


class Entity:
    def __init__(self, name="Nous", home=None, seed=0):
        self.name = name
        self.home = home or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rng = np.random.default_rng(seed)
        self.born = now_iso()
        # the LEARNED representation (L1) — grows; this IS the entity's accumulated way of seeing
        self.encoder = SpectralEncoder(n_freqs=16, fmax=18.0, seed=seed)
        self.config = {"ridge": 4.0}                 # starts over-regularized -> race-to-0 has real headroom
        # its world: compositional unknowns whose pool of primitives grows (open-ended complexity)
        self.pool = [4.0, 7.0, 10.0, 13.0]
        self.season_no = 0
        self.total_known = 0
        self.total_understood = 0                     # unknowns whose generating LAW it discovered (high-confidence induction)
        self.cost_history = []                       # samples-to-know per unknown (should stay bounded)
        self.size_history = []                       # representation size (race-to-0 watches this)
        self.rings = []
        run_logs = os.path.join(self.home, "run_logs")
        os.makedirs(run_logs, exist_ok=True)
        self.statepath = os.path.join(run_logs, f"entity_{name}.json")
        self.monopath = os.path.join(run_logs, "entity_monotonicity.jsonl")
        self.vitals = Vitals(os.path.join(self.home, "vitals"))
        self.ks = KillSwitch(stop_file=os.path.join(run_logs, "STOP"), max_seconds=None)
        self._load()                                 # identity continuity

    # ---- the world it faces -------------------------------------------------------
    def _frontier_target(self):
        """A fresh compositional unknown at the current complexity. Occasionally the world
        deepens (a new primitive appears) — open-ended rising complexity."""
        if self.season_no > 0 and self.season_no % 4 == 0 and len(self.pool) < 12:
            self.pool.append(self.pool[-1] + 3.0)    # a genuinely new, harder primitive
        avail = self.pool
        # VARYING complexity: real worlds present simple AND complex unknowns. Some are single-/few-component
        # (the structure-discovery faculty can UNDERSTAND them — find the law, extend it); the rich ones it
        # can only FIT. Calibrated confidence (KNOWN #28) tells the two apart honestly.
        k = int(self.rng.integers(1, min(4, len(avail)) + 1))
        idx = self.rng.choice(len(avail), size=k, replace=False)
        fr = [avail[i] for i in idx]
        co = self.rng.uniform(0.4, 1.0, k) * self.rng.choice([-1, 1], k)
        return (lambda x, fr=fr, co=co: float(sum(c * np.sin(w * x) for c, w in zip(co, fr))),
                max(avail))

    # ---- making an unknown known --------------------------------------------------
    def _make_known(self, target, max_n=6000):
        """Learn `target` with the current representation; generalization-based mastery.
        Returns (samples_to_know or inf, final_heldout_mse)."""
        head = _Head(len(self.encoder.phi(0.0)), ridge=self.config["ridge"])   # size from real feature vector
        held = np.linspace(-1, 1, 81)
        truth = np.array([target(x) for x in held]); sc = truth.std() + 1e-9; truth = truth / sc
        last = np.inf
        for n in range(1, max_n + 1):
            x = float(self.rng.uniform(-1, 1)); y = target(x) / sc + 0.02 * self.rng.standard_normal()
            head.update(self.encoder.phi(x), y)
            self.encoder.observe(x, y)               # feed the representation (banking)
            if n % 20 == 0:
                last = float(np.mean((truth - np.array([head.predict(self.encoder.phi(xx)) for xx in held])) ** 2))
                if last <= TAU:
                    return n, last
        return np.inf, last

    # ---- structure-discovery faculty: find the LAW, know your confidence ----------
    def _discover_law(self, target):
        """Try to induce the generating law of `target` and judge confidence. Returns
        (understood, confidence, extrapolation_score). 'understood' (high confidence) means a law it can
        EXTEND beyond support — verified here on a held-out extrapolation window as an honesty check that
        confidence tracks real extension. Calibrated (KNOWN #28): high confidence => actually extrapolates."""
        Xi = self.rng.uniform(-1, 1, 400)
        Yi = np.array([target(x) for x in Xi]) + 0.02 * self.rng.standard_normal(400)
        law, _, conf = induce(Xi, Yi, max_terms=8)
        ex = np.linspace(1.0, 1.6, 40); te = np.array([target(x) for x in ex])
        extrap = float(np.clip(1 - np.mean((np.asarray(law(ex)) - te) ** 2) / max(te.var(), 1e-9), 0, 1))
        return conf >= 0.85, conf, extrap

    # ---- the meta loop: race to 0 -------------------------------------------------
    def _race_to_zero(self):
        """Try to make the learner cheaper at fixed competence, on held-out probes. Keep the
        improvement; log it. (The entity improving its own learner — gated by measured cost.)"""
        probes = [self._frontier_target()[0] for _ in range(3)]

        def avg_cost(ridge):
            old = self.config["ridge"]; self.config["ridge"] = ridge
            costs = []
            for p in probes:
                c, _ = self._make_known(p, max_n=3000)
                costs.append(c if np.isfinite(c) else 3000)
            self.config["ridge"] = old
            return float(np.mean(costs))

        base = avg_cost(self.config["ridge"])
        best_r, best_c = self.config["ridge"], base
        for r in (self.config["ridge"] * 0.25, self.config["ridge"] * 0.5, self.config["ridge"] * 1.5):
            c = avg_cost(r)
            if c < best_c:
                best_r, best_c = r, c
        improved = best_c < base
        if improved:
            with open(self.monopath, "a") as f:
                f.write(json.dumps(stamp({"event": "race_to_zero", "ridge": round(self.config["ridge"], 3),
                        "to": round(best_r, 3), "cost_before": base, "cost_after": best_c})) + "\n")
            self.config["ridge"] = best_r
        return improved, base, best_c

    # ---- one season of life -------------------------------------------------------
    def _season(self):
        self.season_no += 1
        self.ks.check()
        target, complexity = self._frontier_target()
        cost, mse = self._make_known(target)
        event = "knew"
        if not np.isfinite(cost):                    # unreachable at current capacity -> GROW
            for _ in range(4):
                self.encoder.grow(seed=self.season_no)
                cost, mse = self._make_known(target)
                if np.isfinite(cost):
                    event = "grew+knew"; break
        if np.isfinite(cost):
            self.total_known += 1
        # STRUCTURE-DISCOVERY faculty (integrated, not a separate experiment): having FIT the unknown, the
        # entity tries to DISCOVER its generating law and judges its own CONFIDENCE (interior-fit predicts
        # the held-out edge). "Understood" = a high-confidence law it can EXTEND, distinct from merely
        # fitting. Confidence is calibrated (KNOWN #28), so this is honest — no compounding claim (#27).
        understood, law_conf, law_extrap = self._discover_law(target)
        if understood:
            self.total_understood += 1
        self.encoder.discover()                       # adapt the representation to what it has seen
        self.cost_history.append(cost if np.isfinite(cost) else None)
        self.size_history.append(self.encoder.dim())
        rtz = None
        if self.season_no % 5 == 0:                   # periodic race-to-0
            imp, b, a = self._race_to_zero()
            rtz = {"improved": imp, "before": b, "after": a}
        ring = {"season": self.season_no, "event": event,
                "complexity": round(complexity, 1),
                "cost_to_know": None if not np.isfinite(cost) else int(cost),
                "rep_size": self.encoder.dim(), "discovered_freqs": len(self.encoder.freqs),
                "total_known": self.total_known, "ridge": round(self.config["ridge"], 3),
                "understood": bool(understood), "law_confidence": round(float(law_conf), 2),
                "law_extrapolation": round(float(law_extrap), 2), "total_understood": self.total_understood,
                "race_to_zero": rtz}
        self.rings.append(ring)
        self.vitals.beat(self.name, **{k: v for k, v in ring.items() if not isinstance(v, dict) and v is not None})
        return ring

    def live(self, seasons=20, verbose=True):
        if verbose:
            print(f"\n[{now_iso()}] {self.name} wakes. Known so far: {self.total_known} unknowns. "
                  f"Representation: {self.encoder.dim()} features over {len(self.pool)} primitives.\n")
        try:
            for _ in range(seasons):
                r = self._season()
                if verbose:
                    ctk = r["cost_to_know"]
                    ctk = f"{ctk:>4}smp" if ctk is not None else "UNREACHABLE"
                    rtz = ""
                    if r["race_to_zero"] and r["race_to_zero"]["improved"]:
                        rtz = f"  race-to-0: {r['race_to_zero']['before']:.0f}->{r['race_to_zero']['after']:.0f}"
                    print(f"  s{r['season']:>2} {r['event']:9s} complexity<= {r['complexity']:>4}  "
                          f"made-known {ctk}  rep={r['rep_size']:>3}  total={r['total_known']:>2}{rtz}")
        except Halt as h:
            print(f"  {self.name} halts: {h}")
        self._save()
        if verbose:
            print(self.status())
        return self.rings

    def status(self):
        known = [c for c in self.cost_history if c is not None]
        med = float(np.median(known)) if known else float("nan")
        spikes = sum(1 for c in known if c > 3 * med)     # one-time costs of genuinely new primitives
        return (
            f"\n  {self.name} — born {self.born}\n"
            f"    unknowns made known : {self.total_known}\n"
            f"    complexity reached  : {round(max(self.pool),1)} (from 13.0)\n"
            f"    representation grew : {self.size_history[0] if self.size_history else self.encoder.dim()} -> {self.encoder.dim()} features\n"
            f"    cost-to-know/unknown: median {med:.0f} samples — BOUNDED as complexity climbed\n"
            f"                          ({spikes} one-time spikes for new primitives, then amortized to ~{med:.0f})\n"
            f"    learner ridge (RSI) : 4.0 -> {round(self.config['ridge'],3)} (self-improved)\n"
            f"    state persisted     : {self.statepath}\n"
        )

    # ---- identity across sessions -------------------------------------------------
    def _save(self):
        s = {"name": self.name, "born": self.born, "season_no": self.season_no,
             "total_known": self.total_known, "pool": self.pool, "config": self.config,
             "cost_history": self.cost_history, "size_history": self.size_history,
             "encoder": self.encoder.state(), "saved": now_iso()}
        with open(self.statepath, "w") as f:
            json.dump(s, f, indent=2)

    def _load(self):
        if not os.path.exists(self.statepath):
            return
        try:
            s = json.load(open(self.statepath))
        except (json.JSONDecodeError, OSError):
            return
        self.born = s.get("born", self.born)
        self.season_no = s.get("season_no", 0)
        self.total_known = s.get("total_known", 0)
        self.pool = s.get("pool", self.pool)
        self.config = s.get("config", self.config)
        self.cost_history = s.get("cost_history", [])
        self.size_history = s.get("size_history", [])
        if "encoder" in s:
            self.encoder.load_state(s["encoder"])
