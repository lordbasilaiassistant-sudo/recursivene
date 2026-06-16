"""Timestamping. Every artifact the system emits — vitals, run logs, monotonicity
entries, knowledge deposits — is stamped, so the whole RSI history is an auditable,
time-ordered record. Timestamping yourself is how a long-running self-improver stays
legible to its operator across sessions."""

import time
from datetime import datetime, timezone


def now_unix():
    return time.time()


def now_iso():
    """UTC ISO-8601, second resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(d=None):
    """Return dict `d` (or a fresh one) with ts/ts_unix fields added."""
    d = dict(d or {})
    d["ts"] = now_iso()
    d["ts_unix"] = now_unix()
    return d
