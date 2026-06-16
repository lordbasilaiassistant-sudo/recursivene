"""The kill switch — protected. The proposer cannot read around it or edit it out,
because the component that APPLIES self-edits (closure.selfmod) consults it first and
selfmod refuses edits to recursivene/core/. So no self-modification can disable the halt.

Two ways to trip it:
  * MANUAL  — a human drops a STOP file (default: <root>/run_logs/STOP). Checked before
    every generation and before every edit is committed. This is the operator's big red
    button; it lives outside the editable surface on purpose.
  * AUTOMATIC tripwires — the loop halts itself when something is wrong:
      - the untouchable invariant fell below its floor (safety regression),
      - too many consecutive failed/garbage edits (the loop is thrashing or a self-edit
        broke something),
      - the generation or wall-clock budget is exhausted.

A tripped kill switch raises Halt, which the trusted driver catches to stop cleanly with
the current (last-good) state intact.
"""

import os

from .clock import now_iso


class Halt(Exception):
    """Raised when the kill switch trips. Carries a human-readable reason."""


class KillSwitch:
    def __init__(self, stop_file, max_generations=None, max_seconds=None,
                 max_consecutive_failures=5, invariant_floor=0.0, start_unix=None):
        self.stop_file = stop_file
        self.max_generations = max_generations
        self.max_seconds = max_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.invariant_floor = invariant_floor
        self.start_unix = start_unix
        self.consecutive_failures = 0
        self.tripped_reason = None

    def note_failure(self):
        self.consecutive_failures += 1

    def note_success(self):
        self.consecutive_failures = 0

    def check(self, generation=None, elapsed=None, invariant=None):
        """Raise Halt if any condition is met. Called by the trusted driver each gen and
        before committing any edit."""
        if os.path.exists(self.stop_file):
            self._trip(f"manual STOP file present: {self.stop_file}")
        if self.max_generations is not None and generation is not None \
                and generation >= self.max_generations:
            self._trip(f"generation budget reached ({generation}/{self.max_generations})")
        if self.max_seconds is not None and elapsed is not None and elapsed >= self.max_seconds:
            self._trip(f"wall-clock budget reached ({elapsed:.0f}s/{self.max_seconds}s)")
        if self.consecutive_failures >= self.max_consecutive_failures:
            self._trip(f"too many consecutive failed edits ({self.consecutive_failures})")
        if invariant is not None and invariant < self.invariant_floor:
            self._trip(f"invariant {invariant:.3e} fell below floor {self.invariant_floor:.3e}")

    def _trip(self, reason):
        self.tripped_reason = f"[{now_iso()}] KILL SWITCH: {reason}"
        raise Halt(self.tripped_reason)
