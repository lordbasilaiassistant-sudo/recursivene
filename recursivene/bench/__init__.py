"""Non-saturating benchmarks for RecursiveNe.

A fixed scorecard saturates: once the learner aces it, the number stops moving and tells
you nothing about further improvement. These benchmarks instead measure the EXPANDING
FRONTIER — the rate at which cost-for-competence falls, the point at which a weak-RSI
ratchet plateaus and a stronger lever breaks past it, and the growth of the repertoire of
mastered activities. The deliverable is the CONTRAST: a fixed test flatlines while a
frontier metric keeps climbing.

Public surface (per CONTRACTS.md):
    race_to_zero_curve(log_path) -> {slope, points, halflife, ...}
    open_ended_report() -> dict
    plateau_break_demo() -> dict
    saturation_contrast() -> dict
"""

from .benchmark import (
    race_to_zero_curve,
    plateau_break_demo,
    open_ended_report,
    saturation_contrast,
)

__all__ = [
    "race_to_zero_curve",
    "plateau_break_demo",
    "open_ended_report",
    "saturation_contrast",
]
