"""DEMO: run Nous for ~16 seasons and, for every STALL (a cost spike or an UNREACHABLE frontier),
classify it with the WALL DIAGNOSTIC and print which loophole-category it was and which MOVE was
taken — on REAL entity output.

This does NOT edit entity.py. It drives `Entity._season()` season-by-season (the public life loop
calls the same method) and runs `WallDiagnostic.diagnose(...)` against the live encoder + the prior
rings AFTER each season, exactly where the spec says the hook belongs.

Honest framing carried through to the print: every non-None classification is a HYPOTHESIS; the
dimension/complexity walls come out as category #3 (wrong-domain: RFF tail carrying reachable
structure -> richer representation) and capacity-exhaustion / unreachable frontiers come out as
category #2 (missing structure: a GENUINE floor, grow AROUND it).

Reproduction:
    python experiments/loophole_demo.py            (cwd = repo root)
    seed = 0 (Entity default); state at run_logs/entity_<name>.json
"""

import os
import shutil

from _util import REPO_ROOT  # noqa: F401  (path setup + repo import side effect)

from recursivene.entity import Entity
from recursivene.loophole import WallDiagnostic, _median_cost

SEED = 0
SEASONS = 18  # ~16 seasons of life; a touch longer so rising complexity surfaces a reachable spike
K = 6  # window of prior rings the diagnostic uses for its median-cost baseline


def _heldout_probe(entity):
    """A cheap held-out probe under the PROTECTED ruler (objective.TAU): make a fresh frontier
    target known and report its held-out MSE. Lets category #4 (proxy_confusion) fire honestly.
    Cheap cap so the probe itself is not the expensive thing."""
    def probe():
        target, _ = entity._frontier_target()
        _, mse = entity._make_known(target, max_n=1500)
        return mse
    return probe


def main():
    # Fresh identity so the demo is reproducible from a clean slate (don't resume a prior life).
    name = "NousLoophole"
    statepath = os.path.join(REPO_ROOT, "run_logs", f"entity_{name}.json")
    if os.path.exists(statepath):
        os.remove(statepath)

    ent = Entity(name=name, seed=SEED)
    diag = WallDiagnostic()

    print("=" * 100)
    print("WALL DIAGNOSTIC over a real Nous life — classify every stall into its loophole category")
    print("=" * 100)
    print(f"seed={SEED}  seasons={SEASONS}  spike rule: cost > 3*median(prior {K})  state={statepath}")
    print("Categories: #1 hidden_ledger_term(use_bank)  #2 missing_structure(grow_capacity)  "
          "#3 wrong_domain(richer_representation)  #4 proxy_confusion(re_examine_metric)")
    print("-" * 100)

    stalls = 0
    cat_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for _ in range(SEASONS):
        ring = ent._season()
        prior = ent.rings[-(K + 1):-1]  # the K rings BEFORE this one
        wall = diag.diagnose(ring, prior, ent.encoder, heldout_probe=_heldout_probe(ent))
        ring["wall"] = wall  # attach (advisory), exactly as the spec's _season hook would

        ctk = ring["cost_to_know"]
        ctk_s = f"{ctk:>4}smp" if ctk is not None else "UNREACHABLE"
        med = _median_cost(prior)
        med_s = f"{med:>5.0f}" if med is not None else "  n/a"

        if wall["category"] is None:
            print(f"  s{ring['season']:>2} {ring['event']:9s} cmplx<= {ring['complexity']:>4}  "
                  f"made-known {ctk_s}  rep={ring['rep_size']:>3}  med={med_s}  -> no wall")
            continue

        stalls += 1
        cat_counts[wall["category"]] += 1
        real = " [REAL WALL — grow AROUND it]" if wall["real_wall_acknowledged"] else ""
        print(f"  s{ring['season']:>2} {ring['event']:9s} cmplx<= {ring['complexity']:>4}  "
              f"made-known {ctk_s}  rep={ring['rep_size']:>3}  med={med_s}")
        print(f"       WALL  #{wall['category']} {wall['name']:<17s} MOVE: {wall['move']:<20s}"
              f"(HYPOTHESIS){real}")
        print(f"             signature: {wall['signature']}")

    print("-" * 100)
    print(f"stalls classified: {stalls} / {SEASONS} seasons")
    print(f"  #1 use_bank (hidden ledger term)     : {cat_counts[1]}")
    print(f"  #2 grow_capacity (missing structure) : {cat_counts[2]}   <- genuine floors, grown AROUND")
    print(f"  #3 richer_representation (wrong dom) : {cat_counts[3]}   <- dimension/complexity walls")
    print(f"  #4 re_examine_metric (proxy confusion): {cat_counts[4]}")
    print("Every classification above is a HYPOTHESIS about which loophole applies; #2 is the honest")
    print("'this limit is genuine' verdict (route around it by adding form, not through it).")
    print("=" * 100)


if __name__ == "__main__":
    main()
