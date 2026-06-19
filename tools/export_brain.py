"""export_brain.py — compile the RSI's LAST-RUN artifacts into a single viz/brain_data.json that the
3D brain visualizer replays. Pure stdlib. Every field traces to a real logged artifact (no fabrication);
the visualizer is a REPLAY of these outputs, not a live system.

Sources (whatever exists):
  run_logs/entity_Nous.json   -> encoder.freqs (the discovered "concepts"/neurons), pool (goals)
  vitals/Nous.jsonl           -> per-season life: complexity, cost_to_know, rep_size, discovered_freqs,
                                 total_known, ridge  (the entity making unknowns known)
  run_logs/garden_rings.jsonl -> open-ended growth: repertoire, hardest_known, n_features, gamma, cost
  run_logs/monotonicity.jsonl -> self-edits (the system rewiring itself, held-out-gated)
  run_logs/race_{3,6}knob.json-> learned self-edit search vs human hand-tuning (HUD stat)

Run:  python tools/export_brain.py        (writes viz/brain_data.json)
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(rel):
    p = os.path.join(ROOT, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _load_jsonl(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _seasons():
    """Per-season life of Nous: last heartbeat per season number."""
    by_season = {}
    for r in _load_jsonl("vitals/Nous.jsonl"):
        s = r.get("season")
        if s is not None:
            by_season[s] = r          # keep last seen for that season
    seasons = []
    for s in sorted(by_season):
        r = by_season[s]
        seasons.append({
            "season": s,
            "complexity": r.get("complexity"),
            "cost_to_know": r.get("cost_to_know"),
            "rep_size": r.get("rep_size"),
            "discovered_freqs": r.get("discovered_freqs"),
            "total_known": r.get("total_known"),
            "ridge": r.get("ridge"),
        })
    return seasons


def _neurons(entity, seasons):
    """The encoder's DISCOVERED frequencies = the concepts the brain formed from raw data. Assign each
    a discovery season from the discovered_freqs progression (how many it had discovered by season s)."""
    freqs = ((entity or {}).get("encoder") or {}).get("freqs") or []
    # discovery schedule: neuron i becomes 'discovered' when discovered_freqs >= i+1
    sched = [(s["season"], s["discovered_freqs"]) for s in seasons if s.get("discovered_freqs") is not None]
    first_season = seasons[0]["season"] if seasons else 1

    def discovered_season(i):
        for season, df in sched:
            if df is not None and df >= i + 1:
                return season
        return first_season

    return [{"id": i, "freq": round(float(f), 4), "discovered_season": discovered_season(i)}
            for i, f in enumerate(freqs)]


def _race_summary(rel):
    d = _load_json(rel)
    if not d:
        return None
    return {
        "space": d.get("space"), "verdict": d.get("verdict"),
        "claude_best": d.get("claude", {}).get("best_val"),
        "rsi_mean": d.get("rsi", {}).get("mean"), "rsi_best": d.get("rsi", {}).get("best"),
        "ratio_claude_over_rsi": d.get("ratio_claude_over_rsi"),
    }


def main():
    entity = _load_json("run_logs/entity_Nous.json") or {}
    seasons = _seasons()
    garden = [{"season": r.get("season"), "added_w": r.get("added_w"),
               "repertoire": r.get("repertoire"), "hardest_known": r.get("hardest_known"),
               "gamma": r.get("gamma"), "n_features": r.get("n_features"),
               "cost_to_know_samples": r.get("cost_to_know_samples"), "grows": r.get("grows")}
              for r in _load_jsonl("run_logs/garden_rings.jsonl")]
    self_edits = [{"event": r.get("event"), "stage": r.get("stage"), "target": r.get("target"),
                   "descr": r.get("descr"), "meta_cost_before": r.get("meta_cost_before"),
                   "meta_cost_after": r.get("meta_cost_after"), "accepted": r.get("accepted", False)}
                  for r in _load_jsonl("run_logs/monotonicity.jsonl")]

    # the headline meta-cost RATCHET (stage-1 config search) — kept SEPARATE from self_edits[]
    closure = _load_json("run_logs/closure_summary.json") or {}
    model_stage = (closure.get("stages") or {}).get("model") or {}
    ratchet = [{"gen": h.get("gen"), "cost": h.get("cost"), "n_params": h.get("n_params")}
               for h in (model_stage.get("history") or [])]

    data = {
        "meta": {
            "schema_version": 2,
            "title": "RecursiveNe — Nous: a brain learning to know, ever more cheaply",
            "live": False,
            "replay_note": "Replay of the repo's LAST-RUN outputs — not a live system.",
            "sources": ["run_logs/entity_Nous.json", "vitals/Nous.jsonl",
                        "run_logs/garden_rings.jsonl", "run_logs/monotonicity.jsonl",
                        "run_logs/race_6knob.json", "run_logs/race_3knob.json"],
            "telos": "make unknowns known, ever more cheaply (cost-for-competence; race to 0)",
        },
        "neurons": _neurons(entity, seasons),
        "pool": entity.get("pool", []),
        "encoder_fmax": ((entity or {}).get("encoder") or {}).get("fmax"),
        "seasons": seasons,
        "garden": garden,
        "self_edits": self_edits,
        "race": {"6knob": _race_summary("run_logs/race_6knob.json"),
                 "3knob": _race_summary("run_logs/race_3knob.json")},
        "ratchet": {"history": ratchet,
                    "start_cost": model_stage.get("start_cost"),
                    "best_cost": model_stage.get("best_cost")},
        "totals": {
            "n_neurons": len(_neurons(entity, seasons)),
            "n_seasons": len(seasons),
            "total_known": (seasons[-1]["total_known"] if seasons else None),
            "repertoire_end": (garden[-1]["repertoire"] if garden else None),
            "hardest_end": (garden[-1]["hardest_known"] if garden else None),
            "n_features_end": (garden[-1]["n_features"] if garden else None),
            "accepted_self_edits": sum(1 for e in self_edits if e["accepted"]),
        },
    }

    out_dir = os.path.join(ROOT, "viz")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "brain_data.json")
    blob = json.dumps(data, separators=(",", ":"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(blob)
    # also a .js global so the page loads via <script src> on file:// (no fetch/CORS) AND on GH Pages
    with open(os.path.join(out_dir, "brain_data.js"), "w", encoding="utf-8") as f:
        f.write("window.BRAIN_DATA=" + blob + ";")
    t = data["totals"]
    print(f"wrote {out} (+ brain_data.js)")
    print(f"  neurons={t['n_neurons']} seasons={t['n_seasons']} total_known={t['total_known']} "
          f"repertoire_end={t['repertoire_end']} hardest_end={t['hardest_end']} "
          f"n_features_end={t['n_features_end']} accepted_self_edits={t['accepted_self_edits']}")


if __name__ == "__main__":
    main()
