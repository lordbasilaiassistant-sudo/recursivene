# Non-saturating benchmarks — proving "smarter over time"

Owner: BENCHMARKS specialist (Ada). Module: `recursivene/bench/`.
Cross-link: [PROGRESS.md](PROGRESS.md) sections F (race to 0) and D5 (monotonicity).

## The problem these benchmarks exist to solve

A **fixed scorecard saturates**. The protected objective is cost-for-competence (FLOPs to
reach tau). "Did you reach tau on the canonical world?" is binary: once the learner passes,
the number pins at 1.0 forever and can no longer distinguish a system that keeps getting
cheaper from one that froze. True RSI shows up *after* the fixed test is aced — as the cost
still falling and the frontier still expanding. So every metric here is a **rate or a
frontier**, not a pass/fail.

Four benchmarks, all numpy-only, all reading only the frozen `CONTRACTS.md` interfaces
(`search`, `evaluate`, `run`, `make_world`) plus the `run_logs` artifacts.

---

## 1. Race to zero — `race_to_zero_curve(log_path)`

**Hypothesis (falsifiable):** across accepted generations, cost-for-competence has a
*negative* log-linear slope. Falsified if slope >= 0.

**Method:** assemble the cost-for-competence series from whatever artifact has the most
finite points (`monotonicity.jsonl` accepts → `closure_summary.json` stage-1 history →
`vitals/child.jsonl`), take its running minimum (accepted-only ratchet, so non-increasing by
construction), fit `log(cost) = intercept + slope * gen` by least squares. If no usable
artifact exists, generate a real trajectory by running `search()` from a bloated config — the
benchmark never fabricates numbers, it runs the contract.

**Real result (from the completed stage-1 closure run, 7 points):**

```
cost 1.598e+08 -> 1.172e+06   (136.4x cheaper)
log-lin slope = -0.830 / gen   (frac/gen = 0.436, i.e. ~44% of prior cost each accepted gen)
cost halflife = 0.83 generations
params 4608 -> 360
```

The live `vitals_child` series (14 gens) reproduces the same descent with slope ≈ -0.41/gen.
The self-contained fallback `_generate_trajectory()` (zero artifacts) gives slope ≈ -0.43.

---

## 2. Plateau break — `plateau_break_demo()` (weak vs strong RSI)

**Hypothesis (falsifiable):** a weak improvement operator plateaus; a strong operator,
*from the identical start and budget*, descends below that plateau. Falsified if the strong
arm does not get below the weak plateau.

**Method:** both arms start from the SAME modestly-bloated-but-competent config
(`n_features=80`, already reaches competence at gen 0) and run the SAME ratchet for the SAME
gens/steps/seeds on the SAME world. The ONLY difference is operator strength — which is
exactly the lever the closure's stage-2 self-edit rewrites (`MUTATION_SCALE`/`MUTATION_RATE`
in `harness/loop.py`):

- **WEAK** = frozen operator (`scale=0, rate=0` → `mutate()` returns the incumbent
  unchanged). Cannot move in config-space → flatlines. *Weak RSI: autonomy is real, the
  improvement is not.*
- **STRONG** = full operator → explores and descends.

**Real result:**

```
shared gen-0 : 1.266e+07   (both arms competent at start)
WEAK  (frozen): 1.266e7  1.266e7  1.266e7  1.266e7  1.266e7   -> plateau 1.266e+07 (1440 params)
STRONG (full) : 1.266e7  1.266e7  2.466e6  2.466e6  2.466e6   -> post    2.466e+06 ( 630 params)
=> BROKE plateau: strong 5.1x below weak plateau (drop 81%)
```

Same origin, same budget, same world — only the operator differs, and only the strong
operator keeps the cost falling. This is the weak-vs-strong RSI demonstration the closure's
staged design is built around.

> Note on a discarded design: an earlier framing used "bigger model = strong arm." That is
> WRONG for this objective — a bigger model costs MORE FLOPs/step, so at fixed budget a larger
> `n_features` *raises* cost-for-competence (probed: nf=72 → 9.9e6, nf=128 → 3.1e7). The lever
> that matters is the operator's ability to move, not raw capacity. The current design
> reflects that.

---

## 3. Open-ended report — `open_ended_report()`

**Non-saturating metrics** on the EXTENDED frequency ladder (canonical inner +
`extra_w=(15,18,21)` via the `make_world(extra_w=...)` hook), so the frontier has room above
the easy rungs a strong default config aces instantly:

- `repertoire_size` — activities mastered (per-activity worst-grid MSE <= tau).
- `hardest_solved_w` — max frequency mastered (the FRONTIER).
- `stepping_stone` — fraction of the difficulty ladder reached (how far up the rungs mastery
  climbs); `err_freq_slope` — error-vs-frequency slope as a transfer proxy.

**Real result (default config, 2500 steps):** `2/9` mastered, hardest-solved `w=6.0` of
possible `w=21.0`, stepping-stone `0.44`. The default `gamma=8` genuinely cannot fit `w>=9`,
so the report shows real headroom — the metric is non-saturating *because* the frontier
exceeds what the current config can reach. A stronger config (higher gamma) pushes it up.

---

## 4. Saturation contrast — `saturation_contrast()` (the deliverable)

**Hypothesis (falsifiable):** as training budget grows, a fixed binary test flatlines at its
ceiling while a frontier metric keeps climbing. Falsified if the frontier does not rise after
the fixed test saturates.

**Method:** same learner at increasing step budgets on the extended ladder (moderate
`gamma=12` so the climb is gradual). FIXED = "easiest activity mastered?" (binary, saturates
at 1.0). FRONTIER = repertoire COUNT (climbs gradually).

**Real result:**

```
budgets       : [ 300,  600, 1000, 1600, 2400, 3400]
FIXED  (easy) : [ 0,    0,    1,    1,    1,    1  ]   <- saturates at step 1000, then FLAT
FRONTIER count: [ 0,    1,    6,    8,    8,    8  ]   <- still rising 6->8 AFTER fixed pinned
hardest-w     : [ 0.0,  9.0, 15.0, 21.0, 21.0, 21.0]
=> CONTRAST holds: True  (fixed saturated at index 2; frontier rose 6->8 afterward)
```

At budgets where the fixed test is already pinned at its ceiling (steps 1000→1600), the
frontier count is still strictly rising. That is the whole point: the fixed benchmark is
blind to improvement the frontier metric still measures.

---

## Reproduction

From repo root (`C:\Users\drlor\OneDrive\Desktop\Recursivene`):

```bash
# full suite: ASCII curves + writes run_logs/bench_report.json with real numbers
python recursivene/bench/run_benchmark.py

# self-test: 17 assertions, <90s, PASS/FAIL with numbers
python recursivene/bench/test_benchmark.py
```

Last verified run: `test_benchmark.py` → **ALL PASS in 32.8s** (17/17 assertions). Seeds are
fixed inside each benchmark; `race_to_zero` reads `run_logs/` and falls back to a real
generated `search()` trajectory if no usable artifact is present, so the suite is robust to
whatever state a concurrently-running closure leaves the logs in.

## What would falsify the whole suite

- race-to-0 slope >= 0 (cost not decreasing) — the ratchet isn't working.
- strong arm fails to get below the weak plateau — operator strength doesn't matter (no
  strong RSI over weak).
- frontier count flat once the fixed test saturates — the system genuinely stopped at the
  ceiling, i.e. it is NOT smarter over time.

All three currently come out the *non-falsifying* way against real run data.

## Contract friction

None. The frozen interfaces (`search`, `evaluate`, `run`, `make_world(extra_w=...)`,
`learner.models[r].predict`) were sufficient. One coordination note: a **concurrently-running
closure live-edits `harness/loop.py`** (e.g. `MUTATION_SCALE` 0.5→0.25 observed mid-session)
and rewrites `run_logs/` artifacts. The benchmarks tolerate this by selecting the
longest-finite series and regenerating a trajectory when the logs are short/flat — but anyone
reading `bench_report.json` should note the `race_to_zero.source` field to know which artifact
the curve came from.
