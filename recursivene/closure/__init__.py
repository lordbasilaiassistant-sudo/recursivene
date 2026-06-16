"""recursivene.closure — the META-META level: the system editing the code that improves
the system, including the proposer's own definition.

This is where strong RSI lives. Weak RSI tunes an artifact with a frozen optimizer;
here the optimizer (harness) and the proposer are themselves inside the editable surface,
and an edit to them is kept only if RUNNING the edited improver finds cheaper,
generalizing models — gated by the protected objective, the untouchable invariant, the
kill switch, and a rollback sandbox. selfmod.py is itself PROTECTED (the kernel cannot be
casually rewritten by what it runs), which is the trust root that makes the rest safe.
"""

from .selfmod import SelfModifier, Edit, PROTECTED_PATHS, STAGE_SCOPE
from .catalog import constant_edits

__all__ = ["SelfModifier", "Edit", "PROTECTED_PATHS", "STAGE_SCOPE", "constant_edits"]
