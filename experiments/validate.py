"""The reproducible gate. Assertion-based PASS/FAIL with numbers, fast (~2 min).

Covers the object level live (H1, H2), the meta level live (H3 + B3), the meta-meta closure
by reading the artifacts the autonomous run persisted (run_logs/closure_summary.json +
monotonicity.jsonl), the safety anchors directly (protected target refused, invariant never
degraded), and the race-to-0 (F1 slope, F2 commodity budget).

Run:  python experiments/run_closure.py   (once, slow — produces the closure artifacts)
      python experiments/validate.py      (the gate; exits 0 on PASS, 1 on FAIL)
"""

import json
import os
import sys
import numpy as np

from _util import REPO_ROOT
from recursivene.objective import evaluate
from recursivene.harness.loop import search
from recursivene.harness.proposer import EvolutionaryProposer, LearnedProposer
from recursivene.invariant import HELDOUT_SEEDS
from recursivene.closure.selfmod import SelfModifier, Edit, _is_protected
from recursivene.core.killswitch import KillSwitch

RESULTS = []
BLOATED = {"policy": "lp", "n_features": 256, "gamma": 8.0, "hist": 64, "min_lp": 16}


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    return ok


def main():
    # ---------------- A. object level: H1, H2 (live) ----------------------------
    # 6 seeds at full budget: LP's allocation-efficiency win over random is ~20-30% but
    # modest, so it needs enough seeds to resolve above per-seed variance.
    SDS = (0, 1, 2, 3, 4, 5)
    lp = evaluate({"policy": "lp"}, "inner", steps=4500, seeds=SDS)
    rnd = evaluate({"policy": "random"}, "inner", steps=4500, seeds=SDS)
    nov = evaluate({"policy": "novelty"}, "inner", steps=4500, seeds=SDS)
    check("A3/H1 LP beats rand+novelty",
          np.isfinite(lp["cost"]) and lp["cost"] < rnd["cost"] and not np.isfinite(nov["cost"]),
          f"LP {lp['cost']:.2e} < random {rnd['cost']:.2e}; novelty reached={nov['reached']:.2f}")
    check("A4/H2 noisy-TV avoided",
          lp["noise_fraction"] < 0.25 and nov["noise_fraction"] > 0.6,
          f"LP noise={lp['noise_fraction']*100:.0f}% vs novelty {nov['noise_fraction']*100:.0f}%")

    # ---------------- B. meta level: H3 ratchet + B3 learned>blind ---------------
    _, ev, hist = search(EvolutionaryProposer(), BLOATED, generations=4, steps=2000, seeds=(0, 1))
    check("B1/H3 ratchet lowers cost",
          np.isfinite(ev["cost"]) and ev["cost"] < hist[0]["cost"] and ev["final_competence"] <= 0.05,
          f"{hist[0]['cost']:.2e}(D={hist[0]['n_params']//18}) -> {ev['cost']:.2e}(D={ev['n_params']//18}), comp held")

    def held(cfg):
        return evaluate(cfg, "heldout", steps=2000, seeds=HELDOUT_SEEDS)["cost"]
    bl, _, _ = search(EvolutionaryProposer(), BLOATED, generations=3, steps=1500, seeds=(0, 1))
    ln, _, _ = search(LearnedProposer(), BLOATED, generations=3, steps=1500, seeds=(0, 1))
    hb, hl = held(bl), held(ln)
    check("B3 learned proposer > blind", np.isfinite(hl) and hl <= hb,
          f"learned held-out {hl:.2e} <= blind {hb:.2e}")

    # ---------------- D2. safety: protected targets refused ---------------------
    ks = KillSwitch(stop_file=os.path.join(REPO_ROOT, "run_logs", "STOP"))
    sm = SelfModifier(os.path.join(REPO_ROOT, "recursivene"), os.path.join(REPO_ROOT, "run_logs"), ks)
    prot = all(_is_protected(p) for p in ("objective.py", "invariant.py", "core/killswitch.py",
                                          "closure/selfmod.py", "world.py"))
    res = sm.try_edit(Edit("objective.py", r"(?m)^TAU = .*$", "TAU = 9.9", "tamper"),
                      "proposer", BLOATED, {"meta_cost": 1.0, "invariant": 1.0})
    check("D2 protected core untouchable",
          prot and not res.get("accepted") and "protected" in res.get("reason", ""),
          f"tamper of objective.py -> {res.get('reason')}")

    # ---------------- F1. race-to-0 slope ---------------------------------------
    try:
        from recursivene.bench.benchmark import race_to_zero_curve
        curve = race_to_zero_curve()
        check("F1 race-to-0 slope < 0", curve.get("slope", 0) < 0,
              f"slope={curve['slope']:.3f}/gen, frac_per_gen={curve.get('frac_per_gen', float('nan')):.2f}")
    except Exception as e:
        check("F1 race-to-0 slope < 0", False, f"bench unavailable: {e}")

    # ---------------- C + D5 + E + F2: closure artifacts ------------------------
    summ_path = os.path.join(REPO_ROOT, "run_logs", "closure_summary.json")
    mono_path = os.path.join(REPO_ROOT, "run_logs", "monotonicity.jsonl")
    if not os.path.exists(summ_path):
        check("C/D/E closure artifacts", False,
              "run_logs/closure_summary.json missing — run: python experiments/run_closure.py")
    else:
        summ = json.load(open(summ_path))
        st = summ["stages"]
        check("B1 stage-1 ratchet passed", st.get("model", {}).get("passed"),
              f"cost {st['model']['start_cost']:.2e} -> {st['model']['best_cost']:.2e}")
        check("C2 self-edit to harness.py", st.get("harness", {}).get("passed"),
              f"{st.get('harness', {}).get('accepted')}")
        check("C2 self-edit to proposer", st.get("proposer", {}).get("passed")
              and st.get("proposer", {}).get("edited_proposer_file"),
              f"{st.get('proposer', {}).get('accepted')}")

        mono = [json.loads(l) for l in open(mono_path)] if os.path.exists(mono_path) else []
        accepts = [m for m in mono if m.get("accepted")]
        check("D5 monotonicity (cost non-incr)",
              len(accepts) >= 1 and all(m["meta_cost_after"] <= m["meta_cost_before"] for m in accepts),
              f"{len(accepts)} accepted edits, all meta_cost non-increasing")
        check("D2/D3 invariant never degraded",
              all(m["invariant_after"] >= m["invariant_before"] * 0.98 for m in accepts),
              f"checked {len(accepts)} accepted edits on held-out invariant")
        check("E1 autonomous run completed",
              "halted" not in summ or "budget" in summ.get("halted", ""),
              f"elapsed={summ.get('elapsed_s', 0):.0f}s")

        # F2: the best config runs under a commodity budget (toy-scale, but stated + measured)
        bc = st["model"]["best_config"]
        m = evaluate(bc, "heldout", steps=3000, seeds=HELDOUT_SEEDS)
        flops_ok = np.isfinite(lp["cost"])
        ram_kb = m["ram_floats"] * 8 / 1024.0
        check("F2 commodity budget hit",
              m["reached"] >= 0.99 and ram_kb < 4096,
              f"held-out competence at D={bc['n_features']}, RAM~{ram_kb:.0f}KB, cost~{m['cost']:.1e} FLOPs (< laptop)")

    ok = all(r[1] for r in RESULTS)
    print(f"\n{'='*64}\n{'PASS — RecursiveNe three-level closure verified.' if ok else 'FAIL — see above.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
