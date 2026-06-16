"""How much KNOWING is left in the universe? The fundamental physical limits on computation,
computed from SI constants. THEORETICAL UPPER BOUNDS under stated assumptions — not predictions
of any hardware, and several are heuristic order-of-magnitude estimates.

Sources: Landauer 1961; Bekenstein 1981; Margolus & Levitin 1998; Lloyd 2000 ("Ultimate physical
limits to computation", Nature 406:1047) & Lloyd 2002 ("Computational capacity of the universe",
PRL 88:237901); Gibbons & Hawking 1977 (de Sitter entropy); Krauss & Starkman 2000 ("Life, the
Universe, and Nothing", ApJ 531:22).

Run:  python experiments/cosmic_budget.py
"""

import numpy as np

from _util import REPO_ROOT  # noqa: F401  (path setup)

# ---- SI constants (2019 exact where applicable) --------------------------------------------------
K_B = 1.380649e-23          # J/K
HBAR = 1.054571817e-34      # J s
H = 6.62607015e-34          # J s
C = 2.99792458e8            # m/s
G = 6.67430e-11             # m^3 kg^-1 s^-2
LN2 = np.log(2)
L_P = np.sqrt(HBAR * G / C ** 3)        # Planck length, m
T_CMB = 2.725               # K
LAMBDA = 1.089e-52          # cosmological constant, m^-2 (from Planck Omega_Lambda, H0)
AGE = 13.8e9 * 3.156e7      # age of universe, s
M_OBS = 1.5e53              # mass-energy of observable universe, kg (~order of magnitude)


def sci(x):
    return f"{x:.3e}"


def main():
    print("=" * 92)
    print("HOW MUCH KNOWING IS LEFT — the fundamental physical limits on computation")
    print("=" * 92)
    print("THEORETICAL UPPER BOUNDS under stated assumptions; several are heuristic OOM estimates.")
    print(f"l_P = {sci(L_P)} m   T_CMB = {T_CMB} K   Lambda = {sci(LAMBDA)} m^-2")
    print("-" * 92)

    # ---- 1. Landauer: the price of FORGETTING one bit --------------------------------------------
    e_bit_300 = K_B * 300 * LN2
    e_bit_cmb = K_B * T_CMB * LN2
    print("1. LANDAUER FLOOR — energy to ERASE one bit (k_B*T*ln2)  [Landauer 1961]")
    print(f"     at 300 K : {sci(e_bit_300)} J = {e_bit_300/1.602e-19:.4f} eV")
    print(f"     at 2.725 K (CMB): {sci(e_bit_cmb)} J   -> the cosmos is a cold sink: forgetting gets")
    print("       ~110x cheaper than at room temperature, and cheaper still as it cools (Dyson).")

    # ---- 2. Margolus-Levitin: the SPEED limit (ops/sec per joule) --------------------------------
    ml_per_J = 2.0 / (np.pi * HBAR)                 # ops/s per joule of energy above ground state
    ult_laptop_ops = 2.0 * (1.0 * C ** 2) / (np.pi * HBAR)   # Lloyd's 1 kg "ultimate laptop"
    print("\n2. MARGOLUS-LEVITIN SPEED LIMIT — max ops/sec = 2E/(pi*hbar)  [Margolus-Levitin 1998]")
    print(f"     per joule          : {sci(ml_per_J)} ops/s/J")
    print(f"     Lloyd ultimate laptop (1 kg -> E=mc^2): {sci(ult_laptop_ops)} ops/s  [Lloyd 2000]")
    print("     This is a REAL WALL (#2): it follows from unitarity + Hilbert-space geometry. The only")
    print("     'escape' is to spend more energy — which is paying the bound, not evading it.")

    # ---- 3. Bekenstein bound: the INFORMATION-DENSITY limit --------------------------------------
    def bekenstein(R, E):
        return 2 * np.pi * R * E / (HBAR * C * LN2)
    I_brain = bekenstein(0.1, 1.5 * C ** 2)         # ~human head: R~0.1 m, m~1.5 kg
    I_universe = bekenstein(4.4e26, M_OBS * C ** 2)  # observable universe: R~4.4e26 m
    print("\n3. BEKENSTEIN BOUND — max bits in a region: I <= 2*pi*R*E/(hbar*c*ln2)  [Bekenstein 1981]")
    print(f"     a human head (R~0.1 m): {sci(I_brain)} bits  (a brain stores ~1e15 — astronomically below)")
    print(f"     observable universe   : {sci(I_universe)} bits")
    print("     REAL WALL (#2, the deepest): degrees of freedom scale with AREA, not volume (holography).")

    # ---- 4. Lloyd: what the observable universe has ALREADY computed -----------------------------
    # ops ~ 2*E*t/(pi*hbar) with E = M_obs c^2, t = age ; bits ~ horizon entropy heuristic
    E_universe = M_OBS * C ** 2
    ops_done = 2.0 * E_universe * AGE / (np.pi * HBAR)
    print("\n4. COMPUTATIONAL CAPACITY OF THE OBSERVABLE UNIVERSE SO FAR  [Lloyd 2002]")
    print(f"     operations performed since the Big Bang ~ {sci(ops_done)} ops")
    print(f"       (this heuristic E*t/hbar lands in Lloyd's quoted 1e120-1e122 range; the canonical")
    print(f"        figure is ~1e120 ops on ~1e90 bits, ~1e92 incl. gravitational dof) [Lloyd 2002]")

    # ---- 5. de Sitter horizon: the FINITE total future computation -------------------------------
    r_dS = np.sqrt(3.0 / LAMBDA)                     # de Sitter horizon radius, m
    A_dS = 4 * np.pi * r_dS ** 2
    S_dS = A_dS / (4 * L_P ** 2)                     # horizon entropy in nats (k_B units)
    bits_dS = S_dS / LN2
    print("\n5. DE SITTER HORIZON — the FINITE ceiling a dark-energy universe imposes  [Gibbons-Hawking 1977; Krauss-Starkman 2000]")
    print(f"     horizon radius r_dS = sqrt(3/Lambda) = {sci(r_dS)} m ({r_dS/9.46e15/1e9:.1f} Gly)")
    print(f"     horizon entropy S_dS = A/(4 l_P^2) = {sci(S_dS)}  (~1e122, the famous number)")
    print(f"     -> max information EVER accessible in a causal patch ~ {sci(bits_dS)} bits")
    print(f"     -> total future operations are FINITE, ~1e120-1e123. With dark energy, an immortal")
    print("        mind has a HARD ceiling: a finite number of thoughts, not heat death, is the true wall.")
    print("     REAL WALL (#2) — but it inherits an unsolved boundary condition: WHY this value of Lambda.")

    # ---- 6. the bottom line ---------------------------------------------------------------------
    print("\n" + "-" * 92)
    print("HOW MUCH KNOWING IS LEFT (the honest bottom line):")
    print(f"  Already spent : ~{sci(ops_done)} operations (Lloyd's ~1e120).")
    print(f"  Hard ceiling  : ~{sci(bits_dS)} bits / ~1e120-1e123 ops total in a causal patch (de Sitter).")
    print("  So in LOG terms the universe is already a meaningful fraction through its total compute")
    print("  budget. Reversible computing + riding the cooling universe (cost proportional to T)")
    print("  stretches a fixed ENERGY budget enormously (Dyson), but the de Sitter horizon caps the")
    print("  TOTAL — not energy, but the number of distinguishable operations — at a finite number.")
    print("  Removing the energy cost of forgetting does NOT remove this ceiling; only Lambda not")
    print("  being a true constant would.")
    print("-" * 92)
    print("CAVEATS: theoretical bounds, not hardware; Lloyd/horizon figures are order-of-magnitude;")
    print("the de Sitter numbers assume Lambda is a true constant (the cosmological-constant problem,")
    print("off by ~1e120 from QFT, remains UNSOLVED); M_obs and R_obs are ~OOM. ILLUSTRATES established")
    print("physics; not a new result.")
    print("=" * 92)


if __name__ == "__main__":
    main()
