# RecursiveNe — Observability

How to watch a long autonomous closure run without attaching a debugger.

## What writes the data
The closure driver (`recursivene/closure/driver.py`) keeps **vitals on both itself and its
RSI child** every step of the way, so the run is a legible, time-ordered record instead of a
black box. Three artifact streams feed the dashboard, all append-only and timestamped:

- `vitals/parent.jsonl` — the **parent** (orchestrator / three-stage improvement loop). One
  beat when each stage finishes and one each time a self-edit lands. Fields depend on stage:
  `model` beats carry `start_cost / best_cost / passed`; `harness` and `proposer` beats carry
  `meta_cost` and `accepted_edit`.
- `vitals/child.jsonl` — the **child** (the RSI learner being improved). One beat per
  generation of the stage-1 config ratchet: `cost` (cost-for-competence), `n_params`,
  `competence`, `noise_fraction`. This is the **race to 0** series.
- `run_logs/monotonicity.jsonl` — one record per self-edit *attempt* (accept or reject), with
  before/after `meta_cost` and `invariant`, and a `reason` on rejects. This is the audit trail
  the trust kernel (`closure/selfmod.py`) writes.
- `run_logs/closure_summary.json` — written once the full run completes (or halts); per-stage
  pass/fail and elapsed time.

## The tools
Both are pure stdlib, **read-only over the whole repo**, and import only `core.clock` for
their own timestamp — they never load the model/harness/closure code, so running the dashboard
can never perturb the run it is watching.

```bash
python tools/dashboard.py        # one snapshot, then exit
python tools/watch.py            # re-render every 5s until Ctrl-C
python tools/watch.py 2          # re-render every 2s
python tools/watch.py --once     # single frame (same as dashboard.py)
```

`watch.py` delegates all reading and formatting to `dashboard.render()`, so there is exactly
one source of truth for the panel.

## Reading the panel
```
PARENT  stage / last-beat age / meta_cost / stage-1 cost transition + pass
CHILD   stage / generation / cost-for-competence / n_params / competence / noise
        cost trend  : <sparkline>   v down (good)
        start <c0> -> now <cN>   (<cN as % of c0>)
SELF-EDITS  attempts / accepted / rejected / crash-garbage, last few accepts + rejects
KILL SWITCH manual STOP file + inferred tripwires
```

- **Last-beat age** is the liveness check. A child age that keeps growing while the parent is
  still in `stage=model` means stage 1 stalled; an age larger than a generation's wall-clock
  (tens of seconds here) is the first thing to look at.
- **cost trend / sparkline** is the headline: cost-for-competence should march down toward 0
  (`v down (good)`). A flat or rising trend with competence not climbing is a stalled ratchet.
  Newest sample is on the **right**.
- **`% of start`** quantifies the descent so far (e.g. `0.73% of start` = cost fell ~137x).
- **SELF-EDITS** shows strong RSI actually happening: every accept *lowered* the held-out
  `meta_cost` without degrading the invariant; rejects show the gate doing its job (a
  `invariant degraded` or `no meta_cost improvement` reject is the system *refusing* a bad
  self-edit, which is healthy, not a failure).

## Kill-switch readout (inferred, not imported)
The dashboard can't hold the live `KillSwitch` object, so it reports the two observable things
that trip it (`core/killswitch.py`):

- **manual STOP** — presence of `run_logs/STOP`, the operator's big red button. If present, the
  panel shows `*** STOPPED ***`.
- **tripwires** — read from the monotonicity log: a tail run of `crash / no result` rejects is
  thrashing pressure (the `max_consecutive_failures` tripwire); `invariant`-reason rejects mean
  the invariant floor caught a safety regression. "none detected — run looks healthy" means
  neither is present.

## Failure modes it tolerates
Missing files degrade to "no heartbeats yet" per panel; a half-written final JSONL line (a
writer mid-append) is skipped rather than fatal; and the output is encoding-safe — it prefers
UTF-8 (sparkline + box rules) and falls back to ASCII glyphs on a console that refuses it, so
`python tools/dashboard.py` always runs cleanly even when vitals are sparse.
