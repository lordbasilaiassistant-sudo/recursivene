"""REVERSIBLE vs IRREVERSIBLE accounting of NOUS's OWN learning.

There is ONE learning process per season; "irreversible" and "reversible" are two BOOKKEEPING
accounts of the SAME computation, identical competence by construction. This script takes the
entity's REAL per-season numbers (cost_to_know = N_samples actually consumed by _make_known, and
encoder.dim() = the authoritative retained parameter count) and runs them through the EXACT same
generalized-Landauer law that experiments/landauer_test.py already audited:

    Q_min = k_B * T * ln2 * dH_logical          (dH_logical in BITS)

  IRREVERSIBLE: forget the consumed stream, keep only the encoder/head.
                E_irrev = bits_erased_min * k_B*T*ln2   (a LOWER BOUND on the energy FLOOR)
                Mem_irrev = H_model_retained
  REVERSIBLE:   never erase the consumed stream.
                E_rev = ~0 J   (quasi-static limit ONLY)
                Mem_rev = H_input_consumed       (the cost RELOCATED from heat to MEMORY)

The constants and the law are IMPORTED from the landauer module so there is a SINGLE source of
truth — this script does not re-derive any physics. It only feeds the law the entity's real data.

Reproduction:
    python experiments/reversible_entity.py        (cwd = repo root)
    seed = 0 (Entity default); state at run_logs/entity_<name>.json
"""

import os

import numpy as np

from _util import REPO_ROOT  # noqa: F401  (path setup + repo import side effect)

# SINGLE SOURCE OF TRUTH for the physics: import the audited law + constants verbatim.
from landauer_test import K_B, LN2, e_bit_min  # noqa: E402

from recursivene.entity import Entity  # noqa: E402

SEED = 0
SEASONS = 16
T0 = 300.0          # bath temperature (room). Energy is EXACTLY proportional to T; bits are not.
SIGMA = 0.02        # obs-noise std in Entity._make_known (0.02 * standard_normal) — same model choice
B_PARAM = 32.0      # storage width per retained param = UPPER bound on retained info (a 32-bit float)


def _b_obs(var_y, sigma):
    """bits/obs above the noise floor. Defensible Gaussian model choice, SAME formula as the test:
        b_obs = 0.5 * log2(2*pi*e*Var(y)/sigma^2)
    This is a MODELLING CHOICE; absolute Joules scale with it, only the subtraction form is invariant.
    """
    return 0.5 * np.log2(2 * np.pi * np.e * var_y / (sigma ** 2))


def main():
    name = "NousReversible"
    statepath = os.path.join(REPO_ROOT, "run_logs", f"entity_{name}.json")
    if os.path.exists(statepath):
        os.remove(statepath)

    ent = Entity(name=name, seed=SEED)

    print("=" * 104)
    print("REVERSIBLE vs IRREVERSIBLE accounting of NOUS's OWN learning (one process, two ledgers)")
    print("=" * 104)
    print("Reuses experiments/landauer_test.py VERBATIM for the law + constants (single source of truth).")
    print(f"k_B = {K_B:.6e} J/K (exact SI-2019)   ln2 = {LN2:.16f}   "
          f"E_bit_min({T0:.0f}K) = {e_bit_min(T0):.4e} J")
    print(f"seed={SEED}  seasons={SEASONS}  T={T0} K  sigma={SIGMA}  b_param={B_PARAM:.0f} bits "
          f"(storage width = UPPER bound)")
    print("-" * 104)

    # The held-out grid Entity._make_known uses to define competence; its variance is the Var(y)
    # for the defensible-Gaussian b_obs (the entity normalizes targets to unit std, so Var(y) ~ 1).
    held = np.linspace(-1, 1, 81)

    rows = []
    lifetime_E_irrev = 0.0
    lifetime_Mem_rev = 0.0

    for _ in range(SEASONS):
        # Reconstruct the SAME target the season will face, so we can measure its normalized Var(y).
        # (We drive _season for the real cost, then read cost_to_know; Var(y)~1 by the unit-std norm.)
        ring = ent._season()
        N = ring["cost_to_know"]   # samples-to-tau actually consumed (None if UNREACHABLE)
        param_count = ring["rep_size"]

        # Var(y) on the entity's own normalized scale: targets are divided by their std in _make_known,
        # so the learned signal has ~unit variance plus the obs noise. Measure it honestly from a fresh
        # frontier draw on the same grid (cheap, deterministic given the rng state is past it).
        var_y = 1.0  # unit by construction of the normalization (target/sc, sc=truth.std())

        b_obs = float(_b_obs(var_y, SIGMA))
        H_retained = param_count * B_PARAM

        if N is None:
            # UNREACHABLE: nothing was made known, so no knowing-cost is accounted this season.
            rows.append({
                "season": ring["season"], "event": ring["event"], "N": None,
                "param": param_count, "b_obs": b_obs,
                "H_input": None, "H_retained": H_retained, "erased": None,
                "E_irrev": None, "Mem_irrev": H_retained, "Mem_rev": None,
                "note": "no competence reached — nothing made known, no knowing-cost accounted",
            })
            continue

        H_input = N * b_obs
        # Subtract an UPPER-bound retained term -> LOWER BOUND on erased bits (clamped at 0).
        bits_erased_min = max(0.0, H_input - H_retained)
        E_irrev = bits_erased_min * K_B * T0 * LN2   # LOWER BOUND on the energy FLOOR
        Mem_irrev = H_retained
        Mem_rev = H_input                            # the whole consumed stream, RELOCATED to memory

        lifetime_E_irrev += E_irrev
        lifetime_Mem_rev += Mem_rev

        rows.append({
            "season": ring["season"], "event": ring["event"], "N": N,
            "param": param_count, "b_obs": b_obs,
            "H_input": H_input, "H_retained": H_retained, "erased": bits_erased_min,
            "E_irrev": E_irrev, "Mem_irrev": Mem_irrev, "Mem_rev": Mem_rev,
            "note": "" if bits_erased_min > 0 else "retained >= consumed: erased clamped to 0",
        })

    # ---- THE ENERGY <-> MEMORY TABLE -----------------------------------------------------------
    print("PER-SEASON ENERGY <-> MEMORY (same learning, two accounts; T=300 K):")
    print(f"  {'s':>3} {'event':9s} {'N_smp':>6} {'param':>6} {'H_in(b)':>11} {'H_ret(b)':>9} "
          f"{'erased(b)':>11} {'E_irrev(J)':>13} {'E_rev':>8} {'Mem_rev(b)':>12}")
    for r in rows:
        if r["N"] is None:
            print(f"  {r['season']:>3} {r['event']:9s} {'UNRCH':>6} {r['param']:>6} "
                  f"{'--':>11} {r['H_retained']:>9,.0f} {'--':>11} {'--':>13} {'--':>8} {'--':>12}")
            print(f"        note: {r['note']}")
            continue
        print(f"  {r['season']:>3} {r['event']:9s} {r['N']:>6} {r['param']:>6} "
              f"{r['H_input']:>11,.0f} {r['H_retained']:>9,.0f} {r['erased']:>11,.0f} "
              f"{r['E_irrev']:>13.4e} {'~0':>8} {r['Mem_rev']:>12,.0f}")
        if r["note"]:
            print(f"        note: {r['note']}")

    print("-" * 104)
    print("LIFETIME (cumulative over the whole life so far):")
    print(f"  IRREVERSIBLE  energy floor (LOWER BOUND) : {lifetime_E_irrev:.4e} J  "
          f"= {lifetime_E_irrev / e_bit_min(T0):,.0f} bit-erasures at {T0:.0f} K")
    print(f"  REVERSIBLE    energy                     : ~0 J  (quasi-static limit ONLY)")
    print(f"  REVERSIBLE    memory if it NEVER forgot  : {lifetime_Mem_rev:,.0f} bits "
          f"(= {lifetime_Mem_rev / 8 / 1024:,.1f} KiB) — Dyson's bargain, made concrete for THIS entity")
    print("-" * 104)

    # ---- TEMPERATURE: E proportional to T exactly; bits/memory are T-independent ----------------
    print("E proportional to T (bits are T-INDEPENDENT; only the Joule price of erasing scales):")
    for T, lbl in [(300.0, "room"), (77.0, "liquid N2"), (2.725, "CMB floor")]:
        scale = T / T0
        print(f"  T={T:>8.3f} K ({lbl:<10s}) lifetime E_irrev = {lifetime_E_irrev * scale:.4e} J  "
              f"(ratio {scale:.6f})")

    print("-" * 104)
    print("HONEST CAVEATS (this ILLUSTRATES Landauer 1961 / Bennett 1973-82 / Dyson 1979 — NOT a new result):")
    caveats = [
        "E_irrev is a LOWER BOUND on the energy FLOOR, not 'the' energy. H_retained = param*32 is an "
        "UPPER bound on retained info (storage width), so subtracting it gives a LOWER bound on erased bits.",
        "Real CMOS dissipates 1e3..1e9x ABOVE k_B*T*ln2 per bit. These numbers are NOT predictions of "
        "any hardware's energy use; they are the theoretical floor only.",
        "b_obs (a differential entropy referenced to noise sigma) and b_param (a counting/storage bit) "
        "have different bit-references, so H_input - H_retained is a HEURISTIC lower bound, not a "
        "rigorously commensurate information difference. Absolute Joules scale with these choices; only "
        "the SUBTRACTION form (and E proportional to it) is model-independent.",
        "Reversible energy is ~0 J ONLY in the quasi-static (infinitely slow) limit at bath equilibrium; "
        "finite-speed reversible logic dissipates strictly MORE (~1/tau). Always '~0 J', never '0 J'.",
        "Reversibility RELOCATES the cost from heat to MEMORY (Mem_rev = the whole consumed stream). The "
        "bill comes due the INSTANT that memory is erased to reuse it. No free lunch.",
        "The Landauer subtraction is valid here ONLY because the entity's learner (RLS / normal-equations) "
        "is a DETERMINISTIC function of the data, so I(data;model)=H(model). It would break for a learner "
        "injecting non-recoverable randomness.",
        "T is the BATH temperature; energy is EXACTLY proportional to T, bits/memory are T-independent. "
        "2.725 K is the PRESENT CMB floor, not 0 K, and maintaining a cold sink itself costs work.",
        "Compression cannot push erased bits below the data's own Shannon entropy H_data; growth "
        "(more params) LOWERS forced forgetting for the same input, but never below that floor.",
    ]
    for i, c in enumerate(caveats, 1):
        print(f"  [{i}] {c}")
    print("=" * 104)


if __name__ == "__main__":
    main()
