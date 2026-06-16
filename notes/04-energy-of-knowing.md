# The energy of knowing — the Landauer rung, settled

`experiments/landauer_test.py` · verified + adversarially audited (workflow: physics-verify → build → honesty-audit).

The one rung of the deep "why does life/compute need energy?" question that is actually testable in
our substrate — and it is settled, honestly. This ILLUSTRATES established physics (Landauer 1961,
Bennett 1973/1982, Dyson 1979); it is **not** a new result.

## Setup
One learning computation makes a stream of unknowns known (RFF + running normal-equations fit), read
two ways — as an **irreversible** account (fold each observation in, then erase it) and a
**reversible** account (keep the whole trace). Both reach **identical competence on the same data**
(worst held-out MSE 0.0001 ≪ τ=0.02, asserted). So "knowing" is held exactly equal; the *only*
measured difference is energy vs memory.

## Result (real numbers, T=300 K)
| account | bits erased | energy | memory held |
|---|---|---|---|
| **irreversible** (forgets data) | 6,774 | **1.9e-17 J** (lower bound on the floor) | 12,800 bits |
| **reversible** (keeps trace) | 0 | **~0 J** (quasi-static only) | 19,574 bits (the whole stream) |

- **Forgetting is what costs.** The energy of an irreversible learner is set, by the generalized
  Landauer law `Q = k_B·T·ln2·ΔH_logical`, by the bits it *erases* — not by the bits it computes.
- **Reversible → ~0 J, but the cost is relocated, not abolished:** it's paid in **memory** (store
  everything) and only in the quasi-static limit (finite speed costs ~1/τ more). No free lunch.
- **Colder is cheaper:** energy ∝ T exactly. 300 K → 1.9e-17 J; 77 K → 5.0e-18 J; 2.725 K (CMB) →
  1.8e-19 J.
- **The floor is hard:** lossless compression cannot push erased bits below the data's own Shannon
  entropy `H_data` — the floor is set by information content, not register width.

## What it settles (and what it doesn't)
**Settles:** the *energy of knowing* has a real physical floor, and that floor lives entirely in
**forgetting**. The "race to 0" — driving the cost of competence toward zero — bottoms out exactly
two ways: **never forget** (reversibility, paid in ever-growing memory) or **get colder** (energy ∝ T).
That is precisely Dyson's "Time Without End" bargain — the computational core of the intuition that
*removing dissipative energy needs lets mind persist as free energy thins.*

**Does NOT settle:** the cosmological "why" — why a low-entropy start, why potential rather than
nothing. No experiment in this substrate touches those; they remain the genuine open floor.

The honesty caveats (theoretical floor not hardware; modelling choices; quasi-static idealization;
not a new result) are printed in the output and were enforced by the adversarial audit.
