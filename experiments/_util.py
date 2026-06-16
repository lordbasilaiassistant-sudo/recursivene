"""Shared helpers for experiment scripts (path setup + tiny ASCII plotting)."""

import os
import sys

# Make `recursivene` importable when running scripts directly from the repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Unicode sparklines die on the Windows cp1252 console; force UTF-8 with a safe fallback.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, lo=None, hi=None):
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-12:
        return _SPARK[0] * len(values)
    out = []
    for v in values:
        t = (v - lo) / (hi - lo)
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        out.append(_SPARK[int(t * (len(_SPARK) - 1))])
    return "".join(out)


def bar(frac, width=20):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)
