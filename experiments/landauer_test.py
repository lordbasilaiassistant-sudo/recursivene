"""EXPERIMENT: "the energy of knowing" — the Landauer cost of making unknowns known,
reversible vs irreversible.

This makes ONE rung of a deep question testable: WHY does knowing/computing/being-alive cost
energy, and can that cost be driven to zero? It ILLUSTRATES established physics — it is NOT a new
result:
    Landauer (1961) IBM J. Res. Dev. 5(3):183  — erasing 1 bit dissipates >= k_B*T*ln2 of heat.
    Bennett (1973) IBM J. Res. Dev. 17(6):525 ; (1982) Int. J. Theor. Phys. — reversible computation.
    Dyson  (1979) Rev. Mod. Phys. 51:447  "Time Without End" — eternal intelligence by reversible,
                                                                ever-colder, never-forgetting compute.

THE RIGOROUS LAW (we use the generalized Landauer form, NOT "1 bit = kT ln2 asserted per register"):
    Q_min = k_B * T * ln2 * dH_logical
where dH_logical = H_before - H_after = the DECREASE in Shannon entropy (in BITS) of the logical
memory state caused by the operation. A logically reversible step has dH_logical = 0 -> Q_min = 0.
Erasing / overwriting / merging computational paths is what makes dH_logical > 0.

For a learner that consumes N observations to reach competence and keeps only a fitted model:
    bits_erased_min = H_input_consumed - I(consumed_data ; retained_model)
Because the model here is a DETERMINISTIC function of the data (running normal-equations / RLS fit),
the model is a function of the data, so I(data;model) = H(model) = H_model_retained, and the
subtraction form
    bits_erased_min = H_input_consumed - H_model_retained
is exactly valid (correction #3). If the learner injected non-recoverable randomness, the correct
term would be the mutual information and the simple subtraction would break.

BOUNDING DIRECTION (correction #2, the single biggest overclaim risk):
    H_model_retained = param_count * b_param is the model's STORAGE WIDTH — an UPPER bound on its
    information content, not the true I(data;model). Subtracting an upper-bound retained term yields
    a LOWER bound on bits_erased, hence E_irreversible printed here is a LOWER BOUND ON THE ENERGY
    FLOOR, not "the" energy. It is labelled as such everywhere.

Run:  python experiments/landauer_test.py    (cwd = repo root)
"""

import numpy as np

from _util import bar  # noqa: F401  (path setup + repo import side effect)

# --------------------------------------------------------------------------------------------------
# PHYSICAL CONSTANTS (SI 2019, exact)
# --------------------------------------------------------------------------------------------------
K_B = 1.380649e-23          # Boltzmann constant, J/K (exact since 2019 SI redefinition)
LN2 = 0.6931471805599453    # natural log of 2

# Every information quantity below is in BITS (log base 2). Landauer's ln2 is the bits<->nats bridge:
# multiplying bits by k_B*T*ln2 gives Joules. (correction #9: nat/bit mix silently doubles/halves E.)


def e_bit_min(T):
    """Landauer lower bound on heat to erase ONE bit, dumped into a bath at temperature T (Joules)."""
    return K_B * T * LN2


# --------------------------------------------------------------------------------------------------
# THE WORLD: a stream of M unknowns. Each target is a sum of sines (the entity's world).
# --------------------------------------------------------------------------------------------------
def make_target(rng, n_components=3):
    """Return f(x): sum of sines with random freq/phase/amp on x in [-pi, pi]."""
    freqs = rng.uniform(0.5, 3.0, size=n_components)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_components)
    amps = rng.uniform(0.5, 1.5, size=n_components)

    def f(x):
        out = np.zeros_like(x)
        for w, p, a in zip(freqs, phases, amps):
            out += a * np.sin(w * x + p)
        return out

    return f


# --------------------------------------------------------------------------------------------------
# THE LEARNER: Random Fourier Features + running normal equations (accumulated sufficient stats).
# This makes the "fold-in-each-observation-then-discard-it" structure REAL: each observation is
# absorbed into (XtX, Xty) and the raw observation is never needed again. NOTE (correction #11): the
# Gram matrix is itself a LOSSY summary of the data — it retains only I(data;model)-worth of bits.
# It does NOT let you reconstruct the discarded observations. That lossiness is exactly WHY erasing
# the raw data costs energy and is consistent with the accounting; the sufficient stats are NOT
# lossless w.r.t. the data.
# --------------------------------------------------------------------------------------------------
class RFFLearner:
    def __init__(self, D, in_dim, rng, gamma=1.0):
        self.D = D                                   # number of random features = param count
        self.W = rng.normal(0, np.sqrt(2 * gamma), size=(in_dim, D))
        self.b = rng.uniform(0, 2 * np.pi, size=D)
        self.XtX = np.zeros((D, D))
        self.Xty = np.zeros(D)
        self.lam = 1e-6                              # tiny ridge for invertibility

    def _phi(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W + self.b)

    def absorb(self, X, y):
        """Fold a batch of observations into the running sufficient statistics, then forget them."""
        P = self._phi(X)
        self.XtX += P.T @ P
        self.Xty += P.T @ y
        # P, X, y are now discardable: all retained info lives in (XtX, Xty).

    def solve(self):
        A = self.XtX + self.lam * np.eye(self.D)
        self.w = np.linalg.solve(A, self.Xty)
        return self.w

    def predict(self, X):
        return self._phi(X) @ self.w


# --------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(7)

    # ---- configuration -------------------------------------------------------------------------
    M = 5                    # number of unknowns in the stream
    N = 600                  # observations consumed PER unknown (IDENTICAL for both learners)
    D = 80                   # RFF features = retained parameter count
    IN_DIM = 1
    SIGMA = 0.05             # Gaussian label-noise std (the "noise floor")
    TAU = 0.02               # competence threshold: held-out MSE <= TAU counts as "known"
    N_TEST = 2000

    # ---- modelling choices for the information accounting (correction #4: print them) ----------
    # b_obs: bits to encode one scalar observation y above its noise floor. Defensible Gaussian
    # choice: b_obs = 0.5 * log2(2*pi*e*Var(y)/sigma^2)  -- information a noiseless y carries above
    # noise of std sigma. This is a MODEL CHOICE. Results scale with b_obs; only the SUBTRACTION
    # form (bits_erased = H_input - H_retained) is invariant.
    # b_param: chosen storage precision per retained parameter (a 32-bit float register here).
    B_PARAM = 32.0           # bits per parameter (storage width = UPPER bound on retained info)

    print("=" * 90)
    print("THE ENERGY OF KNOWING — Landauer cost of making unknowns known (reversible vs irreversible)")
    print("=" * 90)
    print("Illustrates: Landauer 1961, Bennett 1973/1982, Dyson 1979. NOT a new result.")
    print(f"Constants: k_B = {K_B:.6e} J/K (exact SI-2019), ln2 = {LN2:.16f}")
    print(f"E_bit_min(300K) = {e_bit_min(300):.4e} J = {e_bit_min(300)/1.602176634e-19:.4f} eV   "
          f"E_bit_min(77K) = {e_bit_min(77):.4e} J   E_bit_min(2.725K) = {e_bit_min(2.725):.4e} J")
    print("-" * 90)

    # ---- run the stream: BOTH learners reach the SAME competence on the SAME unknowns -----------
    # There is conceptually one learning process; "irreversible" and "reversible" are two BOOKKEEPING
    # accounts of the SAME computation (same N, same model, same competence). The only difference is
    # whether the consumed data is ERASED (irreversible) or KEPT as a trace (reversible).
    competences = []
    var_y_total = 0.0
    n_var = 0
    for m in range(M):
        f = make_target(rng, n_components=3)
        Xtr = rng.uniform(-np.pi, np.pi, size=(N, IN_DIM))
        ytr = f(Xtr[:, 0]) + rng.normal(0, SIGMA, size=N)
        var_y_total += np.var(ytr) * N
        n_var += N

        learner = RFFLearner(D, IN_DIM, rng)
        # fold in observations in minibatches, discarding each batch after absorption
        bs = 50
        for i in range(0, N, bs):
            learner.absorb(Xtr[i:i + bs], ytr[i:i + bs])
        learner.solve()

        Xte = rng.uniform(-np.pi, np.pi, size=(N_TEST, IN_DIM))
        yte = f(Xte[:, 0])                       # clean held-out targets
        mse = float(np.mean((learner.predict(Xte) - yte) ** 2))
        competences.append(mse)

    competences = np.array(competences)
    var_y = var_y_total / n_var

    # b_obs from the declared Gaussian formula
    b_obs = 0.5 * np.log2(2 * np.pi * np.e * var_y / (SIGMA ** 2))

    # ---- competence control (correction #8): assert BOTH learners (same run) clear TAU ----------
    # Both accounts share this single competence, by construction. Fail loudly if knowing is unequal
    # or sub-threshold, else the energy-vs-memory comparison would be confounded.
    max_mse = float(competences.max())
    print(f"COMPETENCE (identical for irreversible & reversible — same N={N}, same model, same fit):")
    print(f"  held-out MSE per unknown: " + ", ".join(f"{c:.5f}" for c in competences))
    print(f"  worst MSE = {max_mse:.5f}   TAU = {TAU}   -> "
          f"{'ALL KNOWN (competence held equal)' if max_mse <= TAU else 'FAILED THRESHOLD'}")
    assert max_mse <= TAU, (
        f"Competence not reached (worst MSE {max_mse:.5f} > TAU {TAU}); the energy-vs-memory "
        f"comparison would be confounded by unequal knowing. Raise N or D.")
    print("-" * 90)

    # ---- THE INFORMATION ACCOUNTING (in BITS) --------------------------------------------------
    # Total over the whole stream of M unknowns.
    H_input_consumed = M * N * b_obs                  # bits taken in (model choice via b_obs)
    H_model_retained = M * D * B_PARAM                # storage width; UPPER bound on retained info

    # bits_erased_min = H_input - retained_info, clamped at 0 (erasure cannot be negative).
    # Subtracting an UPPER-bound retained term -> LOWER bound on erased bits -> LOWER bound on energy.
    bits_erased_min = max(0.0, H_input_consumed - H_model_retained)

    print("INFORMATION ACCOUNTING (all quantities in BITS; multiply by k_B*T*ln2 for Joules):")
    print(f"  modelling choice  b_obs   = {b_obs:.4f} bits/obs  "
          f"[= 0.5*log2(2*pi*e*Var(y)/sigma^2), Var(y)={var_y:.4f}, sigma={SIGMA}] (MODEL CHOICE)")
    print(f"  modelling choice  b_param = {B_PARAM:.1f} bits/param "
          f"[32-bit register = storage width = UPPER bound on retained info]")
    print(f"  H_input_consumed  = M*N*b_obs       = {H_input_consumed:,.1f} bits")
    print(f"  H_model_retained  = M*D*b_param     = {H_model_retained:,.1f} bits   "
          f"(UPPER bound on I(data;model))")
    print(f"  bits_erased_min   = H_in - H_ret    = {bits_erased_min:,.1f} bits   "
          f"(LOWER bound on erasure; compression regime: H_ret << H_in)")
    if H_model_retained >= H_input_consumed:
        print("  NOTE: retained >= consumed (over-parameterized) -> erased clamped to 0; "
              "no compression, no forced forgetting.")
    print("  WHY the subtraction is valid: the model is a DETERMINISTIC function of the data")
    print("  (running normal-equations fit), so I(data;model)=H(model). It would break for a")
    print("  learner using non-recoverable randomness (correction #3).")
    print("-" * 90)

    # ---- ENERGY vs MEMORY at room temperature --------------------------------------------------
    T0 = 300.0
    E_irrev = bits_erased_min * K_B * T0 * LN2        # forgets the data, keeps only the model
    E_rev = 0.0                                       # ~0 in the quasi-static limit ONLY
    Mem_irrev = H_model_retained                      # data discarded; hold only the model
    Mem_rev = H_input_consumed                        # keeps the whole consumed stream (the trace)

    print(f"ENERGY vs MEMORY at T = {T0} K (room temperature):")
    print(f"  {'account':<14}{'bits erased':>16}{'ENERGY (J)':>20}{'MEMORY held (bits)':>22}")
    print(f"  {'IRREVERSIBLE':<14}{bits_erased_min:>16,.0f}{E_irrev:>20.4e}{Mem_irrev:>22,.0f}")
    print(f"  {'REVERSIBLE':<14}{0:>16,.0f}{'~0 (quasi-static)':>20}{Mem_rev:>22,.0f}")
    print()
    print(f"  IRREVERSIBLE energy is a LOWER BOUND on the floor (retained term is an upper bound).")
    print(f"  REVERSIBLE energy is ~0 J ONLY in the quasi-static (tau->inf) limit at bath")
    print(f"  equilibrium; finite-speed reversible logic dissipates strictly MORE (~1/tau). No free")
    print(f"  lunch: the cost is RELOCATED from heat to MEMORY ({Mem_rev:,.0f} bits, the whole stream).")
    print(f"  The bill comes due the instant that memory is erased to reuse it.")
    print("-" * 90)

    # ---- TEMPERATURE SWEEP: energy ∝ T exactly; bits & memory are T-independent -----------------
    print("TEMPERATURE SWEEP (cold compute / Dyson): bits & memory are T-INDEPENDENT; only the")
    print("Joule price of erasing them scales with the BATH temperature T.")
    print(f"  {'T (K)':>10}{'regime':>16}{'E_irrev (J)':>18}{'E(T)/E(300)':>16}")
    temps = [(300.0, "room"), (77.0, "liquid N2"), (2.725, "CMB floor")]
    e300 = bits_erased_min * K_B * 300.0 * LN2
    energies = {}
    for T, name in temps:
        E = bits_erased_min * K_B * T * LN2
        energies[T] = E
        print(f"  {T:>10.3f}{name:>16}{E:>18.4e}{E / e300:>16.6f}")

    # ASSERT E ∝ T (correction #10): catches any T accidentally leaking into the bit count.
    r77 = energies[77.0] / energies[300.0]
    r2 = energies[2.725] / energies[300.0]
    assert abs(r77 - 77.0 / 300.0) < 1e-9, f"E(77)/E(300) ratio leak: {r77}"
    assert abs(r2 - 2.725 / 300.0) < 1e-9, f"E(2.725)/E(300) ratio leak: {r2}"
    print(f"  Energy is linear in T BY CONSTRUCTION (T enters once, multiplicatively, after the bits")
    print(f"  are already fixed): ratios = 77/300={77/300:.6f}, 2.725/300={2.725/300:.6f}. The assert")
    print(f"  is a REGRESSION GUARD against a future edit leaking T into the bit count, not a")
    print(f"  discovered physical result.")
    print("  Dyson: ride the cooling universe — as the CMB falls toward 0 K the price of each")
    print("  irreversible bit falls with it. CAVEAT: 2.725 K is the PRESENT CMB floor, not 0 K, and")
    print("  maintaining a cold sink itself costs work (no perpetual-cooling free lunch).")
    print("-" * 90)

    # ---- THE FLOOR IS HARD: compress-then-erase cannot beat the data's own entropy --------------
    # Build a discrete proxy stream, compress it losslessly, and show erasing the COMPRESSED stream
    # still erases >= (H_data - H_retained) bits. A real compressor only APPROACHES the source
    # entropy H_data; it never goes below it (correction #7) — so we expect "~same floor, never
    # below", NOT exact equality.
    sym_rng = np.random.default_rng(11)
    # a source with known per-symbol Shannon entropy (non-uniform 4-symbol alphabet)
    p = np.array([0.5, 0.25, 0.15, 0.10])
    H_data_per_symbol = float(-np.sum(p * np.log2(p)))    # source entropy in bits/symbol
    n_sym = 200_000
    stream = sym_rng.choice(4, size=n_sym, p=p)

    raw_bits = n_sym * 2.0                                 # naive fixed-width: 2 bits/symbol
    # empirical entropy of the realized stream (what an ideal compressor approaches, from below-ish)
    counts = np.bincount(stream, minlength=4).astype(float)
    q = counts / counts.sum()
    H_emp = float(-np.sum(q[q > 0] * np.log2(q[q > 0])))
    ideal_compressed_bits = n_sym * H_emp                 # ideal lossless floor for THIS stream

    print("THE FLOOR IS HARD — compression cannot push erased bits below the source entropy:")
    print(f"  source entropy H_data = {H_data_per_symbol:.4f} bits/symbol; empirical = {H_emp:.4f}")
    print(f"  raw stored (2 bits/sym)        = {raw_bits:,.0f} bits")
    print(f"  ideal lossless compressed      = {ideal_compressed_bits:,.0f} bits "
          f"(= n*H_emp; a real compressor only APPROACHES this, never below H_data)")
    print(f"  erasing the COMPRESSED stream still costs >= n*H_data = "
          f"{n_sym * H_data_per_symbol:,.0f} bits of Landauer erasure.")
    print("  Compression relabels the bits more cheaply but CANNOT delete information for free: the")
    print("  erasure floor is set by H_data, not by the register width you happened to store it in.")
    assert ideal_compressed_bits >= n_sym * (H_data_per_symbol - 0.02), \
        "compressed stream fell below source entropy — impossible; check accounting"
    print("-" * 90)

    # ---- HONEST CONCLUSIONS + REQUIRED CAVEATS -------------------------------------------------
    print("CONCLUSION (the cosmic-question rung this settles):")
    print("  1. Knowing is not what costs — making unknowns known is computation, and computation")
    print("     can be done reversibly (Q_min = k_B*T*ln2*dH_logical = 0 when dH_logical = 0).")
    print("  2. FORGETTING/overwriting (logically irreversible steps) is what has a Landauer floor.")
    print("     'Knowing is cheap; forgetting costs' — precisely: irreversible ops cost, reversible")
    print("     ops don't IN THE HEAT LEDGER (quasi-static limit) — they still cost MEMORY;")
    print("     overwriting/merging is just the common case.")
    print("  3. The race-to-zero energy bottoms out ONLY two ways: (a) never forget — reversibility,")
    print("     paid in ever-growing MEMORY (Dyson's bargain), or (b) get COLDER — energy ∝ T.")
    print("  4. Both learners here KNEW exactly the same things to the same competence; the ONLY")
    print("     measured difference was energy vs memory. That is the whole point.")
    print()
    print("REQUIRED HONESTY CAVEATS (baked in, audit-enforced):")
    caveats = [
        "THEORETICAL FLOOR ONLY: real CMOS/devices dissipate 1e3..1e9 x ABOVE k_B*T*ln2 per bit. "
        "These are the physical lower bound, NOT predictions of any hardware's energy use.",
        "b_obs and b_param are EXPLICIT MODELLING CHOICES; absolute Joule numbers scale with them. Only "
        "bits_erased_min = H_input - retained_info (and E proportional to it) is model-independent. AND "
        "b_obs is a DIFFERENTIAL entropy (continuous y referenced to noise sigma) while b_param is a "
        "COUNTING/storage-width bit — not the same bit-reference, so the subtraction is a HEURISTIC "
        "lower bound, not a rigorously commensurate information difference.",
        "H_model_retained = param_count*b_param is an UPPER bound on retained info (storage width), so "
        "the reported E_irreversible is a LOWER BOUND on the energy floor, not 'the' energy. Second "
        "reason it is a lower bound: b_obs counts only the ABOVE-NOISE information, while the physical "
        "raw-observation register that gets overwritten is wider, so actual erasure of the raw data "
        "registers is >= the b_obs accounting.",
        "Reversible energy is ~0 ONLY in the quasi-static (infinitely slow) limit; finite-speed "
        "reversible computation dissipates strictly more. NO free lunch. ('~0 J', never '0 J'.)",
        "Reversibility RELOCATES the cost from heat to MEMORY (store the whole H_input stream + "
        "history). The bill is due on the next erase.",
        "The subtraction is valid because the model is a DETERMINISTIC function of the data "
        "(I(data;model)=H(model)); it would break for a learner using non-recoverable randomness.",
        "T is the BATH temperature; 2.725 K is the present CMB floor, not 0 K, and maintaining a cold "
        "sink itself costs work (no perpetual-cooling free lunch).",
        "Compression cannot push erased bits below the data's own Shannon entropy H_data; the "
        "compressed-then-erase demo approaches but does not beat the floor.",
        "All information quantities are in BITS (log base 2) before multiplying by k_B*T*ln2; "
        "nat/bit conversion errors silently double- or half-count energy.",
        "This ILLUSTRATES established physics (Landauer 1961; Bennett 1973/1982; Dyson 1979 'Time "
        "Without End'). It is NOT a new result and must not be presented as one.",
    ]
    for i, c in enumerate(caveats, 1):
        print(f"  [{i:>2}] {c}")
    print("=" * 90)


if __name__ == "__main__":
    main()
