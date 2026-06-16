# RecursiveNe — Frozen Module Contracts

These interfaces are STABLE. Every specialist module depends only on what is written here,
never on another specialist's internals. If you own a module folder, you may change
anything *inside* your folder; you may NOT change these signatures or any file outside your
folder. This is the "modulation" that lets specialists run in parallel without collision.

## Protected (never edit — not even the proposer)
- `recursivene/objective.py`  — `competence(world, learner)`, `evaluate(config, which, steps, seeds, tau) -> dict`, `TAU`
- `recursivene/invariant.py`  — `invariant_score(config, steps) -> float` (held-out, higher=better)
- `recursivene/world.py`      — `make_world(which, seed, n_noise, obs_noise, extra_w) -> World`
- `recursivene/core/`         — `now_iso()`, `stamp(d)`, `KillSwitch`, `Halt`, `Vitals`
- `recursivene/closure/selfmod.py` — the trust kernel

## Object level (substrate) — owner: spine (Claude)
- `recursivene/model.py`  — `RFFOnlineRegressor(n_features,gamma,ridge,forget,seed)` with
  `.predict(x)`, `.update(x,y)->err`, `.n_params()`, `.flops`, `.ram_floats()`
- `recursivene/agent.py`  — `RegionLearner(...)`, `choose(policy, learner, world, rng, epsilon)`, `POLICIES`
- `recursivene/seed.py`   — `run(config, world, steps, seed) -> (world, learner, log)`, `DEFAULT_CONFIG`
  - `log` keys: `steps[], competence[], flops[], region_seq[], visits, noise_indices, names, n_params, ram_floats`

## Meta level — owner: spine (Claude)
- `recursivene/harness/loop.py` — `search(proposer, init_config, generations, steps, seeds) -> (best_config, best_eval, history)`,
  `meta_evaluate(proposer_factory, init_config, ...) -> {meta_cost, invariant, found_config, found_params}`
- `recursivene/harness/proposer.py` — `Proposer.propose(best_config, history, rng) -> [config,...]`

## Meta-meta — owner: spine (Claude)
- `recursivene/closure/driver.py` — `run_closure(repo_root, init_config, ...) -> summary dict`
- artifacts: `run_logs/closure_summary.json`, `run_logs/monotonicity.jsonl`, `vitals/*.jsonl`

---

## Specialist module folders (each owned exclusively, non-overlapping)

### `recursivene/generator/` — owner: OPEN-ENDEDNESS specialist
Open-ended problem generation + abstraction: when the learner aces the current world, INVENT
new activities at the learnable frontier (gated novel + previously-unsolvable + now-solvable +
non-forgetting), and mint reusable abstractions to break plateaus.
- MUST expose: `propose_problems(learner_state, world) -> [activity_params,...]`,
  `mint_abstraction(history) -> feature` (may be a stub returning None initially).
- Reads: `make_world(extra_w=...)`, `competence`, the learner's per-activity error/LP.
- Writes ONLY under `recursivene/generator/`. Its own tests in `recursivene/generator/tests_*.py`.

### `recursivene/bench/` — owner: BENCHMARKS specialist
Non-saturating benchmarks proving "smarter over time": race-to-0 curve fit (slope) from
`monotonicity.jsonl` + stage-1 history; the plateau-break demo (weak-RSI plateaus vs closed
loop keeps descending); repertoire-growth / hardest-solved-complexity / transfer metrics that
do NOT saturate; a saturation detector showing a FIXED test flatlines while the open-ended
metric climbs.
- MUST expose: `race_to_zero_curve(log_path) -> {slope, points}`, `open_ended_report() -> dict`.
- Reads: `run_logs/*.jsonl`, `evaluate`, `search`. Writes ONLY under `recursivene/bench/` and
  `run_logs/bench_*.json`.

### `knowledge/` — owner: RESEARCH workflows
Timestamped research deposits (markdown) feeding the project over time. One file per topic,
frontmatter with `ts`, `topic`, `sources`. Never edits code.

### `tools/` — owner: OBSERVABILITY specialist
`tools/dashboard.py` — read `vitals/*.jsonl` + `run_logs/*` and print a live status of parent
+ RSI child (timestamps, current generation, cost-for-competence, invariant, trend). Read-only
over everything else.
