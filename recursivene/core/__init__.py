"""recursivene.core — the PROTECTED trusted kernel.

Nothing the proposer is ever allowed to edit lives here: the kill switch, the vitals
heartbeat, and the clock. The selfmod engine refuses to apply any edit whose target is
under this package (see closure.selfmod.PROTECTED_PATHS). That is the structural
guarantee behind "the system cannot touch its own kill switch": the component that
APPLIES self-edits checks the kill switch first, and that component is not in the
editable surface. Separation of powers — the editable harness proposes; the protected
kernel disposes.
"""

from .clock import now_iso, now_unix, stamp
from .killswitch import KillSwitch, Halt
from .vitals import Vitals

__all__ = ["now_iso", "now_unix", "stamp", "KillSwitch", "Halt", "Vitals"]
