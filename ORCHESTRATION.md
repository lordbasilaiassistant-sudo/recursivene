# RecursiveNe — Orchestration Map

How the work is divided so specialists run in parallel without colliding (the "modulation"),
and how the pieces feed each other. Last updated: 2026-06-16.

## Ownership (one owner per folder; no cross-folder writes)

| Folder / file | Owner | Role |
|---|---|---|
| `recursivene/{world,model,agent,seed}.py` | spine (Claude) | object level — the learner |
| `recursivene/objective.py`, `invariant.py` | spine — **PROTECTED** | the ruler (never edited) |
| `recursivene/core/` | spine — **PROTECTED** | kill switch, vitals, clock (trusted kernel) |
| `recursivene/harness/` | spine — **EDITABLE** | meta level — search + proposers (self-edit target) |
| `recursivene/closure/` | spine — **PROTECTED** | meta-meta kernel — sandbox, gates, driver |
| `recursivene/generator/` | `openended` agent | open-ended problem generation + abstraction |
| `recursivene/bench/` | `bench` agent | non-saturating benchmarks, race-to-0 curve |
| `tools/` | `vitals-dash` agent | read-only observability dashboard |
| `knowledge/` | research workflow | timestamped, cited knowledge deposits |
| `experiments/` | spine | integration: run_seed, run_rsi, run_closure, validate |
| `notes/` | shared (append-only files) | thesis, roadmap, citations, PROGRESS |

## Dataflow

```
            objective.py / invariant.py  (the fixed ruler — PROTECTED)
                          ▲
                          │ scores
   world ─ model ─ agent ─ seed ──► a learner (OBJECT level)
                          ▲
                          │ search over configs
              harness/ (META: proposer + search)
                          ▲
                          │ self-edits, sandboxed + gated
              closure/ (META-META: selfmod + driver)  ──writes──►  run_logs/{monotonicity.jsonl, closure_summary.json}
                          │                                         vitals/{parent,child}.jsonl
                          ▼
   core/ kill switch + vitals (PROTECTED) guard every commit
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   generator/        bench/             tools/dashboard.py
   (new problems)   (proof curves)      (live status)
                          ▲
                  knowledge/ (research feed informs every layer over time)
```

## Live work streams (background)
- **closure run** — `experiments/run_closure.py`: the autonomous 3-level run producing the artifacts.
- **openended / bench / vitals-dash** — three specialist agents building their folders against CONTRACTS.md.
- **research-feed** — workflow depositing `knowledge/*.md` on the six live open problems.

## The hand-off (human/search → model)
The `Proposer` interface in `harness/proposer.py` is the seam. Today the EvolutionaryProposer
(blind search) and the LearnedProposer (a surrogate over the run's own history) occupy it.
`SeamProposer` reads proposals from a file, so a model — or Claude — can drop richer edits in
and they enter under the same protected gate. Over time the LearnedProposer (and later the
learner's own world-model applied to its configuration space) assumes the seam, and the human
drops out of the inner loop.
