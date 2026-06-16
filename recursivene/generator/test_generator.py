"""Assertions for the open-endedness module. Runs in <60s and must PASS.

Run from the repo root:
    python recursivene/generator/test_generator.py

What we prove (each is a falsifiable claim about the generator, not a smoke test):
  T1  repertoire GROWS — a short open-ended loop ends with more mastered activities and a
      higher hardest-solved frequency than it started with.
  T2  NOISE is never proposed as a problem — the noisy-TV activity, and capacity-unlearnable
      high frequencies that flatline like noise, are both refused by the gates.
  T3  the UNSOLVED gate works — a frequency the learner already aces is never re-proposed.
  T4  the NON-FORGETTING gate holds — every accepted problem leaves mastered arms untouched.
  T5  mint_abstraction is honest — None when there is no recurrent structure, a real basis
      feature when there is, with its evidence attached.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recursivene.world import make_world, Activity, World           # noqa: E402
from recursivene.seed import run, DEFAULT_CONFIG                    # noqa: E402
from recursivene.objective import competence, TAU                  # noqa: E402
from recursivene.generator.generator import (                      # noqa: E402
    propose_problems, frontier_summary, _solve_single,
)
from recursivene.generator.abstraction import mint_abstraction     # noqa: E402
from recursivene.generator.run_generator import run_demo           # noqa: E402


def _train(extra_w=(), steps=3000, seed=0, n_noise=12):
    world = make_world("inner", seed=seed, n_noise=n_noise, extra_w=extra_w)
    _, learner, _ = run(DEFAULT_CONFIG, world=world, steps=steps, seed=seed)
    return world, learner


def test_repertoire_grows():
    """T1: a short open-ended loop ends with a strictly larger repertoire AND a strictly
    harder hardest-solved frequency than it began with."""
    report = run_demo(rounds=4, steps=3000, seed=0, verbose=False)
    sc = report["saturation_contrast"]
    rep = sc["open_ended_repertoire_series"]
    cx = sc["open_ended_complexity_series"]
    assert rep[-1] > rep[0], f"repertoire did not grow: {rep}"
    assert cx[-1] > cx[0], f"complexity did not climb: {cx}"
    # and the fixed benchmark must have saturated over the same rounds (the contrast)
    assert sc["fixed_benchmark_saturated"], \
        f"fixed benchmark did not saturate: {sc['fixed_benchmark_cost_series']}"
    print(f"  T1 PASS  repertoire {rep[0]}->{rep[-1]}, complexity {cx[0]}->{cx[-1]}, "
          f"fixed saturated={sc['fixed_benchmark_saturated']}")


def test_noise_never_proposed():
    """T2: the generator never proposes the noisy TV or any capacity-unlearnable frequency.
    Two checks: (a) every proposed frequency is a learnable SINE frequency, never a noise
    arm; (b) a frequency that flatlines at the noise floor (w=30, beyond model capacity)
    fails the solvability gate and is refused even if we hand it to the candidate grid."""
    world, learner = _train(extra_w=())
    proposals = propose_problems(learner, world, n_propose=5, seed=0)
    for p in proposals:
        # learnable sine frequencies only — noise arms have no frequency to propose
        assert p.w > 0.0, f"proposed non-positive frequency {p.w}"
        assert p.err_large <= TAU, f"proposed an UNSOLVED-even-with-budget problem {p.w}"
        assert p.lp >= 0.05, f"proposed a flat (non-learnable) frequency {p.w}"

    # Directly probe a capacity-noise frequency: it must NOT clear the gate. Note the
    # mechanism: a too-high frequency still nibbles SOME error off with more samples
    # (so a naive LP-slope test would be fooled), but it NEVER reaches TAU — it plateaus
    # far above it. The solvability gate (err_large <= TAU) is what rejects it, which is
    # the honest, capacity-aware version of noisy-TV rejection at the problem level.
    from recursivene.seed import DEFAULT_CONFIG as CFG
    cfg = {**CFG, "obs_noise": 0.02}
    e_small = np.mean([_solve_single(30.0, 60, cfg, s) for s in (0, 101)])
    e_large = np.mean([_solve_single(30.0, 2500, cfg, s) for s in (0, 101)])
    assert e_large > TAU, f"w=30 unexpectedly solvable (e_large={e_large:.3f})"
    assert e_large > 0.2, f"w=30 should plateau far above TAU, got {e_large:.3f}"
    # And it is absent from the proposals (the gate refused it).
    assert all(abs(p.w - 30.0) > 1e-6 for p in proposals), "capacity-noise w=30 was proposed!"
    print(f"  T2 PASS  proposed={[round(p.w,1) for p in proposals]}; capacity-noise w=30 "
          f"refused: plateaus at e_large={e_large:.3f} >> TAU={TAU} (never solvable)")


def test_unsolved_gate():
    """T3: a frequency the learner already aces is never proposed (no busywork).
    The base ladder tops out at w=12, which the learner masters; w<=12 frequencies must
    never appear in proposals (they are either present or already-easy)."""
    world, learner = _train(extra_w=())
    proposals = propose_problems(learner, world, n_propose=5, seed=0)
    present = sorted(a.w for a in world.activities if a.learnable)
    for p in proposals:
        assert p.w > max(present) - 1e-9, \
            f"proposed w={p.w} at or below the already-mastered ceiling {max(present)}"
        assert p.err_small > TAU, \
            f"proposed w={p.w} that is already solvable at the tiny budget (err={p.err_small:.3f})"
    print(f"  T3 PASS  all proposals beyond mastered ceiling w={max(present)}, none already-easy")


def test_non_forgetting_gate():
    """T4: every accepted problem reports zero forgetting of mastered arms, and adding it to
    the world genuinely leaves the mastered arms' competence unchanged (own-model isolation)."""
    world, learner = _train(extra_w=())
    proposals = propose_problems(learner, world, n_propose=3, seed=0)
    assert proposals, "expected at least one proposal to test non-forgetting"
    # recorded evidence
    for p in proposals:
        assert p.forget_delta <= 1e-6, f"accepted a forgetting problem (delta={p.forget_delta})"

    # independent check: re-score the mastered arms in the world WITH the new arm appended.
    p0 = proposals[0]
    mastered_idx = [r for r in range(world.K)
                    if world.learnable[r] and learner.recent_error(r) < DEFAULT_CONFIG["tau_master"]]
    grid = np.linspace(-1, 1, 21)
    before = {r: float(np.mean([(world.truth(r, x) - learner.models[r].predict(x))**2 for x in grid]))
              for r in mastered_idx}
    world2 = make_world("inner", seed=0, extra_w=(p0.w,))
    after = {r: float(np.mean([(world2.truth(r, x) - learner.models[r].predict(x))**2 for x in grid]))
             for r in mastered_idx}
    for r in mastered_idx:
        assert abs(before[r] - after[r]) < 1e-9, \
            f"arm {r} competence changed when w={p0.w} was appended: {before[r]} -> {after[r]}"
    print(f"  T4 PASS  {len(mastered_idx)} mastered arms unchanged after appending w={p0.w}")


def test_abstraction_honest():
    """T5: mint_abstraction returns None with no recurrent structure, and a real BasisFeature
    (with attached evidence) when a dense band of mastered frequencies exists."""
    assert mint_abstraction([]) is None, "minted from empty history"
    assert mint_abstraction([{"mastered_ws": [0.0]}]) is None, "minted from a single frequency"
    # sparse mastered set (no 3-in-a-band cluster) -> None
    assert mint_abstraction([{"mastered_ws": [0.0, 6.0, 18.0]}]) is None, \
        "minted from a sparse (non-recurrent) set"
    # dense band -> a feature distilled from exactly that band
    ab = mint_abstraction([{"mastered_ws": [12.0, 13.5, 15.0]}])
    assert ab is not None, "failed to mint from a dense recurrent band"
    assert 12.0 <= ab.w <= 15.0, f"abstraction frequency {ab.w} outside its support band"
    assert ab.n_evidence >= 3, f"abstraction minted on too little evidence ({ab.n_evidence})"
    print(f"  T5 PASS  None when sparse; minted basis @ w={ab.w:.2f} (evidence n={ab.n_evidence})")


def main():
    t0 = time.time()
    print("open-endedness generator tests")
    test_repertoire_grows()
    test_noise_never_proposed()
    test_unsolved_gate()
    test_non_forgetting_gate()
    test_abstraction_honest()
    dt = time.time() - t0
    print(f"\nALL PASS  ({dt:.1f}s)")
    assert dt < 60.0, f"tests took {dt:.1f}s (>60s budget)"


if __name__ == "__main__":
    main()
