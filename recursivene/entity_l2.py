"""NOUS-L2 — the entity, now living in MULTI-DIMENSIONAL worlds.

Same drive as Nous (make unknowns known, ever more cheaply) and the same scaffolding (persistent
identity, vitals, kill switch, capacity growth, race-to-0), but its world-model backend is the
LEARNED representation (SharedDeepBackend) that crosses the dimension wall. Its world is d-dimensional
compositional structure, not 1-D frequencies. Because the learned body PERSISTS and accumulates the
world's shared structure, the entity gets cheaper at making d-D unknowns known over its life — the L1
flattening effect, now past the curse of dimensionality.

This is L2 wired into a living entity, not a demo. Honest scope: toy d-D worlds, mild continual-
learning forgetting; it illustrates the mechanism with the same held-out, cost-to-know discipline.
"""

import json
import os
import numpy as np

from .deep_encoder import SharedDeepBackend
from .core.clock import now_iso, now_unix, stamp
from .core.killswitch import KillSwitch, Halt
from .core.vitals import Vitals

TAU = 0.02            # held-out MSE threshold for "known" in the d-D world


class L2Entity:
    def __init__(self, name="NousL2", home=None, dim=3, seed=0):
        self.name = name
        self.dim = int(dim)
        self.home = home or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rng = np.random.default_rng(seed)
        self.born = now_iso()
        self.backend = SharedDeepBackend(self.dim, hidden=160, seed=seed)
        self.config = {"lr": 2e-3}
        self.pool = [self.rng.uniform(1.5, 3.0, self.dim) for _ in range(5)]   # shared hidden structure
        self.season_no = 0
        self.total_known = 0
        self.cost_history = []        # held-out MSE per unknown (lower over time = cheaper to know)
        self.rings = []
        run_logs = os.path.join(self.home, "run_logs")
        os.makedirs(run_logs, exist_ok=True)
        self.statepath = os.path.join(run_logs, f"entity_{name}.json")
        self.vitals = Vitals(os.path.join(self.home, "vitals"))
        self.ks = KillSwitch(stop_file=os.path.join(run_logs, "STOP"))
        self._load()

    def _frontier_target(self):
        # occasionally deepen the world (a new structural primitive appears)
        if self.season_no > 0 and self.season_no % 4 == 0 and len(self.pool) < 12:
            self.pool.append(self.rng.uniform(1.5, 4.0, self.dim))
        idx = self.rng.choice(len(self.pool), size=min(3, len(self.pool)), replace=False)
        ws = [self.pool[i] for i in idx]
        co = self.rng.uniform(-1, 1, len(ws))
        return lambda x, ws=ws, co=co: float(sum(c * np.sin(w @ x) for c, w in zip(co, ws)))

    def _make_known(self, target):
        """Make `target` known with the persistent learned body; grow the body if it can't reach."""
        self.backend.lr = self.config["lr"]
        mse = self.backend.fit_target(target, N=600, iters=1600, seed=self.season_no)
        event = "knew"
        grows = 0
        while mse > TAU and grows < 2:                 # frontier hard at current capacity -> GROW
            self.backend.grow(add=48, seed=self.season_no + grows)
            mse = self.backend.fit_target(target, N=800, iters=1600, seed=self.season_no)
            grows += 1
            event = "grew+knew"
        return mse, event, grows

    def _race_to_zero(self):
        """Make the learner cheaper: try a different lr on probes, keep it if it lowers held-out MSE."""
        probes = [self._frontier_target() for _ in range(2)]

        def avg(lr):
            old = self.config["lr"]; self.config["lr"] = lr; self.backend.lr = lr
            m = np.mean([self.backend.fit_target(p, N=500, iters=900, seed=99) for p in probes])
            self.config["lr"] = old
            return float(m)
        base = avg(self.config["lr"])
        best_lr, best = self.config["lr"], base
        for lr in (self.config["lr"] * 0.5, self.config["lr"] * 2.0):
            m = avg(lr)
            if m < best:
                best_lr, best = lr, m
        if best < base:
            self.config["lr"] = best_lr
            return True, base, best
        return False, base, best

    def _season(self):
        self.season_no += 1
        self.ks.check()
        target = self._frontier_target()
        mse, event, grows = self._make_known(target)
        known = mse <= TAU
        if known:
            self.total_known += 1
            self.cost_history.append(mse)
        rtz = None
        if self.season_no % 5 == 0:
            imp, b, a = self._race_to_zero()
            rtz = {"improved": imp, "before": b, "after": a}
        ring = {"season": self.season_no, "event": event, "dim": self.dim,
                "pool": len(self.pool), "heldout_mse": round(mse, 5), "known": known,
                "hidden": self.backend.H, "total_known": self.total_known,
                "lr": self.config["lr"], "grows": grows, "race_to_zero": rtz}
        self.rings.append(ring)
        self.vitals.beat(self.name, **{k: v for k, v in ring.items() if not isinstance(v, dict) and v is not None})
        return ring

    def live(self, seasons=18, verbose=True):
        if verbose:
            print(f"\n[{now_iso()}] {self.name} wakes (d={self.dim}). Known so far: {self.total_known}. "
                  f"Learned body: {self.backend.H} hidden units over {len(self.pool)} primitives.\n")
        try:
            for _ in range(seasons):
                r = self._season()
                if verbose:
                    rtz = ""
                    if r["race_to_zero"] and r["race_to_zero"]["improved"]:
                        rtz = f"  race-to-0 lr->{r['lr']:.1e}"
                    k = "known" if r["known"] else "UNREACHED"
                    print(f"  s{r['season']:>2} {r['event']:9s} d={r['dim']} pool={r['pool']:>2} "
                          f"held-out MSE {r['heldout_mse']:.4f} [{k}]  hidden={r['hidden']:>3}  "
                          f"total={r['total_known']:>2}{rtz}")
        except Halt as h:
            print(f"  {self.name} halts: {h}")
        self._save()
        if verbose:
            print(self.status())
        return self.rings

    def status(self):
        c = self.cost_history
        first = float(np.mean(c[:4])) if len(c) >= 4 else (c[0] if c else float("nan"))
        last = float(np.mean(c[-4:])) if len(c) >= 4 else (c[-1] if c else float("nan"))
        drop = first / last if last and last > 0 else float("nan")
        return (
            f"\n  {self.name} — born {self.born}  (d={self.dim} world)\n"
            f"    unknowns made known : {self.total_known}\n"
            f"    learned body grew   : 160 -> {self.backend.H} hidden units\n"
            f"    cost-to-know (held-out MSE): {first:.4f} -> {last:.4f}  "
            f"({'CHEAPER %.1fx as the body learned the world' % drop if last <= first else 'flat/rising'})\n"
            f"    learner lr (RSI)    : {self.config['lr']:.2e}\n"
            f"    state persisted     : {self.statepath}\n"
        )

    def _save(self):
        s = {"name": self.name, "born": self.born, "dim": self.dim, "season_no": self.season_no,
             "total_known": self.total_known, "pool": [w.tolist() for w in self.pool],
             "config": self.config, "cost_history": self.cost_history,
             "backend": self.backend.state(), "saved": now_iso()}
        with open(self.statepath, "w") as f:
            json.dump(s, f)

    def _load(self):
        if not os.path.exists(self.statepath):
            return
        try:
            s = json.load(open(self.statepath))
        except (json.JSONDecodeError, OSError):
            return
        self.born = s.get("born", self.born)
        self.dim = s.get("dim", self.dim)
        self.season_no = s.get("season_no", 0)
        self.total_known = s.get("total_known", 0)
        self.pool = [np.array(w) for w in s.get("pool", [])] or self.pool
        self.config = s.get("config", self.config)
        self.cost_history = s.get("cost_history", [])
        if "backend" in s:
            self.backend.load_state(s["backend"])
