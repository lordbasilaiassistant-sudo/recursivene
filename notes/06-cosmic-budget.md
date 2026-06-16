# How much knowing is left — the cosmic compute budget

`experiments/cosmic_budget.py` · real numbers from SI constants · THEORETICAL upper bounds (several
heuristic OOM), not hardware predictions. Sources: Landauer 1961, Bekenstein 1981, Margolus–Levitin
1998, Lloyd 2000/2002, Gibbons–Hawking 1977, Krauss–Starkman 2000.

## The numbers

| Limit | Value | What it means |
|---|---|---|
| **Landauer** floor (erase 1 bit) | 2.87×10⁻²¹ J @ 300 K · **2.61×10⁻²³ J @ 2.725 K** | forgetting is ~110× cheaper in the cold of space, and cheaper still as the universe cools (Dyson) |
| **Margolus–Levitin** speed | 6.0×10³³ ops/s per joule · **5.4×10⁵⁰ ops/s** for a 1 kg "ultimate laptop" | a **real wall** (#2): unitarity fixes the rate; the only "escape" is spending more energy |
| **Bekenstein** info density | ~3.9×10⁴² bits in a human head · vastly more in the universe | a **real wall** (#2): degrees of freedom scale with **area**, not volume (holography) |
| **Lloyd** — ops the universe has done | ~10¹²⁰–10¹²² operations on ~10⁹⁰ bits | the observable universe is, in effect, a ~10¹²⁰-operation computer |
| **de Sitter horizon** ceiling | r=1.66×10²⁶ m (17.5 Gly), **S=3.3×10¹²², ~4.8×10¹²² bits**, ~10¹²⁰–10¹²³ total ops | with dark energy, a causal patch has a **finite total number of operations, ever** |

## The bottom line — and the answer to your heat-death question

The deepest result is #5, and it directly settles the earlier "can mind outlast the universe?" thread:

- **Energy is not the binding constraint.** Reversible computing drives the *energy* of knowing toward
  zero (paid in memory), and riding the cooling universe makes each irreversible bit ever cheaper
  (cost ∝ T). On the *energy* axis, a mind can stretch a fixed budget enormously — Dyson's eternal
  intelligence is real.
- **But dark energy imposes a hard ceiling on the TOTAL.** The de Sitter horizon caps the number of
  distinguishable operations a causal patch can ever perform at ~10¹²² — not because energy runs out,
  but because the accessible information is finite. **A finite number of thoughts, not heat death, is
  the true wall.** Removing the energy cost of forgetting does *not* remove it.
- **The only escape is one level down:** if Λ is not a true constant, the horizon need not close. And
  *why Λ has the value it does* is one of the genuinely uncracked #2 boundary-condition problems
  (`notes/05-limits-of-knowing.md`). So the wall to immortal cognition inherits the deepest open
  question in physics.

## The honest floor
These are theoretical bounds under stated assumptions; the Lloyd and horizon figures are
order-of-magnitude; the de Sitter numbers assume Λ is a true constant — the cosmological-constant
problem (off by ~10¹²⁰ from the QFT vacuum estimate) is **unsolved**. This illustrates established
physics; it is not a new result.
