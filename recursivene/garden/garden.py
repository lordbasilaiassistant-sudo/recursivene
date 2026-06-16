"""The gardener loop, measured by the only quantity that matters: making unknowns known.

cost_to_know(w) = how much experience (samples, FLOPs) it takes to turn the frequency-w
function from UNKNOWN (the model can't predict it) into KNOWN (grid error <= tau), at the
current capacity. That single measurement is the atom of the whole project: intelligence as
the efficient conversion of reducible uncertainty into mastery.

A season:
  1. REACH  — probe the next frontier unknown (cur_max + gap). If the model can make it known
              within the sample budget at current capacity, master it; the repertoire grows.
  2. GROW   — if the frontier is UNKNOWABLE at current capacity (cost_to_know = inf), widen the
              model's perceptual bandwidth (gamma) and capacity (D) until that unknown becomes
              knowable. Growing to meet what you couldn't reach is the developmental event.
  3. RING   — log the growth ring: which unknown was made known, its cost_to_know, the capacity
              it took, and the cumulative repertoire / hardest-known complexity.

Run many seasons and the rings answer the open question by EXPERIMENT, not speculation:
does the loop COMPOUND (repertoire + complexity keep climbing), and what happens to the cost
of each new unknown as the system grows?
"""

import numpy as np

from ..model import RFFOnlineRegressor
from ..objective import TAU

# starting repertoire: the base inner ladder (already-known frequencies)
BASE = (0.0, 1.5, 3.0, 6.0, 9.0, 12.0)


def cost_to_know(w, gamma, D, max_samples=5000, obs_noise=0.02, eval_every=20, seed=0):
    """Turn frequency-w from unknown -> known. Returns {known, samples, flops, final_mse}."""
    rng = np.random.default_rng(seed)
    m = RFFOnlineRegressor(n_features=int(D), gamma=float(gamma), seed=seed)
    xs = np.linspace(-1.0, 1.0, 41)
    truth = np.sin(w * xs)
    mse = np.inf
    for n in range(1, max_samples + 1):
        x = rng.uniform(-1.0, 1.0)
        y = np.sin(w * x) + obs_noise * rng.standard_normal()
        m.update(x, y)
        if n % eval_every == 0:
            mse = float(np.mean((truth - np.array([m.predict(xx) for xx in xs])) ** 2))
            if mse <= TAU:
                return {"known": True, "samples": n, "flops": m.flops, "final_mse": mse}
    return {"known": False, "samples": np.inf, "flops": np.inf, "final_mse": mse}


def grow_capacity(cap):
    """Widen perceptual bandwidth (gamma) and capacity (D) — the developmental growth step."""
    return {"gamma": min(28.0, cap["gamma"] * 1.3),
            "n_features": min(256, int(round(cap["n_features"] * 1.25)))}


def propose_frontier(present, cap, gap=1.5, max_samples=5000):
    """The next unknown just past the known frontier. Returns (w, cost) if it can be made
    known at this capacity, else (None, cost-of-the-attempt)."""
    w = max(present) + gap
    c = cost_to_know(w, cap["gamma"], cap["n_features"], max_samples=max_samples)
    return (w, c) if c["known"] else (None, c)


def season(state, idx, max_grow=4, gap=1.5):
    """One developmental season. Returns (new_state, ring)."""
    present = list(BASE) + list(state["extra"])
    cap = dict(state["cap"])
    grows = 0
    w, cost = propose_frontier(present, cap, gap=gap)
    event = "reach"
    while w is None and grows < max_grow:          # frontier unknowable here -> grow to reach it
        cap = grow_capacity(cap)
        grows += 1
        w, cost = propose_frontier(present, cap, gap=gap)
        event = "grow+reach"
    if w is None:
        return state, {"season": idx, "event": "stalled", "added_w": None,
                       "repertoire": len(present), "hardest_known": max(present),
                       "gamma": cap["gamma"], "n_features": cap["n_features"],
                       "cost_to_know_samples": None, "cost_to_know_flops": None, "grows": grows}
    new_state = {"extra": list(state["extra"]) + [w], "cap": cap}
    ring = {"season": idx, "event": event, "added_w": round(w, 3),
            "repertoire": len(present) + 1, "hardest_known": round(w, 3),
            "gamma": round(cap["gamma"], 3), "n_features": cap["n_features"],
            "cost_to_know_samples": cost["samples"], "cost_to_know_flops": cost["flops"],
            "grows": grows}
    return new_state, ring


def master_cost(extra_w, cap, seed=0, steps=None):
    """Honest full-repertoire check: cost-for-competence over the WHOLE known ladder at cap
    (FLOPs to sustained worst-case competence). Grows with the repertoire — the metabolic
    cost of everything the system now knows. Periodic because it is the expensive measurement."""
    from ..world import make_world
    from ..seed import run, DEFAULT_CONFIG
    from ..objective import competence
    world = make_world("inner", seed=seed, extra_w=tuple(extra_w))
    n_learn = int(world.learnable.sum())
    steps = steps or max(3000, 700 * n_learn)
    cfg = {**DEFAULT_CONFIG, "n_features": cap["n_features"], "gamma": cap["gamma"]}
    _, learner, log = run(cfg, world=world, steps=steps, seed=seed)
    comp, flops = log["competence"], log["flops"]
    if not comp or comp[-1] > TAU:
        return np.inf
    for i, c in enumerate(comp):
        if c <= TAU:
            return flops[i]
    return np.inf


def grow_garden(seasons=18, repo_root=None, init_cap=None, verbose=True,
                full_check_every=4):
    """Tend the seed for `seasons`. Persist growth rings + vitals. Returns the ring list."""
    import json
    import os
    cap = init_cap or {"gamma": 8.0, "n_features": 96}
    state = {"extra": [], "cap": cap}
    rings = []
    vit = None
    if repo_root:
        from ..core.vitals import Vitals
        vit = Vitals(os.path.join(repo_root, "vitals"))

    for idx in range(1, seasons + 1):
        state, ring = season(state, idx)
        if ring["event"] != "stalled" and idx % full_check_every == 0:
            ring["full_repertoire_flops"] = master_cost(state["extra"], state["cap"])
        rings.append(ring)
        if vit:
            vit.beat("garden", **{k: v for k, v in ring.items() if v is not None})
        if verbose:
            cs = ring.get("cost_to_know_samples")
            cs = f"{cs}smp" if cs not in (None, np.inf) else "UNREACHABLE"
            print(f"  season {idx:>2} {ring['event']:11s} "
                  f"known up to w={ring['hardest_known']:>5}  "
                  f"repertoire={ring['repertoire']:>2}  "
                  f"D={ring['n_features']:>3} gamma={ring['gamma']:>5}  "
                  f"cost_to_know={cs}"
                  + (f"  grows={ring['grows']}" if ring["grows"] else ""))
        if ring["event"] == "stalled":
            if verbose:
                print(f"  season {idx}: STALLED — frontier unknowable even after growth. Honest stop.")
            break

    if repo_root:
        out = os.path.join(repo_root, "run_logs", "garden_rings.jsonl")
        with open(out, "w") as f:
            for r in rings:
                f.write(json.dumps({k: (None if v == np.inf else v) for k, v in r.items()}) + "\n")
    return rings
