"""tools/dashboard.py — read-only vitals dashboard for the RecursiveNe closure loop.

Reads vitals/parent.jsonl, vitals/child.jsonl, run_logs/monotonicity.jsonl and
run_logs/closure_summary.json and prints a single timestamped status panel covering
BOTH entities:

  * the PARENT — the orchestrator / three-stage improvement loop (model -> harness ->
    proposer), which writes a heartbeat each time a stage finishes or an edit lands;
  * the CHILD — the RSI learner being improved, which writes a heartbeat each generation
    of the stage-1 config ratchet (cost-for-competence, n_params, competence, noise).

This is the OBSERVABILITY view: at any moment you can answer "is the parent alive, what
stage/generation, what is the child's competence/cost right now, is the race-to-0 trend
healthy, and has the kill switch tripped?" without attaching a debugger. It is strictly
read-only over the rest of the repo and depends only on the frozen artifact formats; it
never imports the model/harness/closure code, only core.clock for its own timestamp.

Run from the repo root:  python tools/dashboard.py
Degrades gracefully when files are missing or sparse ("no heartbeats yet").
"""

import json
import os
import sys
import time

# Prefer UTF-8 on the console so the sparkline/box-rule glyphs render; on a terminal that
# refuses UTF-8 we fall back to pure ASCII everywhere (see _ascii_safe / _sparkline).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError, OSError):
    pass

# --- locate the repo root and make core.clock importable for our own stamp -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from recursivene.core.clock import now_iso
except Exception:  # pragma: no cover - dashboard must run even if the package is broken
    from datetime import datetime, timezone

    def now_iso():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


VITALS_DIR = os.path.join(_REPO, "vitals")
RUN_LOGS_DIR = os.path.join(_REPO, "run_logs")

# unicode sparkline ramp, low -> high; we down-shift to ASCII if the terminal can't encode
_SPARK = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = ".:-=+*#@"


# ---------------------------------------------------------------------------- readers
def _read_jsonl(path):
    """Return a list of parsed JSON records from a .jsonl file; [] if missing/unreadable.

    Tolerates partial last lines (a writer mid-append) by skipping records that don't
    parse, so the dashboard never crashes on a half-written heartbeat."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------------------- formatting
def _fmt_age(ts_unix):
    """Human 'how long ago' for a unix timestamp, e.g. '12s ago', '3m 4s ago'."""
    if ts_unix is None:
        return "unknown"
    dt = max(0.0, time.time() - float(ts_unix))
    if dt < 60:
        return f"{dt:.0f}s ago"
    if dt < 3600:
        return f"{int(dt // 60)}m {int(dt % 60)}s ago"
    if dt < 86400:
        return f"{int(dt // 3600)}h {int((dt % 3600) // 60)}m ago"
    return f"{int(dt // 86400)}d {int((dt % 86400) // 3600)}h ago"


def _fmt_num(x):
    """Compact number: big values in scientific notation, small ones with sig figs."""
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:  # NaN
        return "nan"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1e6 or ax < 1e-3:
        return f"{x:.3e}"
    if ax >= 1000:
        return f"{x:,.0f}"
    return f"{x:.4g}"


def _sparkline(values, width=40):
    """ASCII/unicode sparkline of a numeric series, newest on the right.

    Non-finite values are dropped. Uses a unicode ramp but falls back to ASCII if the
    output stream can't encode it (common on Windows code pages)."""
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    if not vals:
        return "(no data)"
    if len(vals) > width:
        vals = vals[-width:]
    ramp = _SPARK
    try:
        ramp.encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError, TypeError):
        ramp = _SPARK_ASCII
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return ramp[0] * len(vals)  # flat line
    span = hi - lo
    n = len(ramp) - 1
    return "".join(ramp[int((v - lo) / span * n + 0.5)] for v in vals)


def _trend_arrow(values):
    """Direction of the series end-to-end: down (good, racing to 0), up, or flat."""
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    if len(vals) < 2:
        return "n/a"
    first, last = vals[0], vals[-1]
    if last < first * 0.999:
        return "v down (good)"
    if last > first * 1.001:
        return "^ up"
    return "= flat"


def _hr(char="─", width=64):
    line = char * width
    try:
        line.encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError, TypeError):
        line = "-" * width
    return line


# ----------------------------------------------------------------------------- panels
def _child_panel(child_beats):
    out = ["CHILD  (RSI learner - stage-1 config ratchet, race cost-for-competence to 0)"]
    if not child_beats:
        out.append("  no heartbeats yet (vitals/child.jsonl absent or empty)")
        return out
    last = child_beats[-1]
    out.append(f"  last beat   : {last.get('ts', '?')}  ({_fmt_age(last.get('ts_unix'))})")
    out.append(f"  stage / gen : {last.get('stage', '?')} / generation {last.get('gen', '?')}"
               f"   ({len(child_beats)} beat(s) total)")
    out.append(f"  cost/comp   : {_fmt_num(last.get('cost'))}   "
               f"(n_params={last.get('n_params', '?')})")
    out.append(f"  competence  : {_fmt_num(last.get('competence'))}    "
               f"noise_fraction={_fmt_num(last.get('noise_fraction'))}")
    costs = [b.get("cost") for b in child_beats]
    out.append(f"  cost trend  : {_sparkline(costs)}   {_trend_arrow(costs)}")
    if len(costs) >= 2:
        c0 = next((c for c in costs if c is not None), None)
        cN = costs[-1]
        if c0 and cN is not None and c0 != 0:
            out.append(f"                start {_fmt_num(c0)} -> now {_fmt_num(cN)}"
                       f"   ({(cN / c0):.2%} of start)")
    return out


def _parent_panel(parent_beats, summary):
    out = ["PARENT (orchestrator - three-stage closure: model -> harness -> proposer)"]
    if not parent_beats:
        out.append("  no heartbeats yet (vitals/parent.jsonl absent or empty)")
    else:
        last = parent_beats[-1]
        out.append(f"  last beat   : {last.get('ts', '?')}  ({_fmt_age(last.get('ts_unix'))})")
        out.append(f"  stage       : {last.get('stage', '?')}   ({len(parent_beats)} beat(s) total)")
        # meta_cost shows up on harness/proposer beats; track the latest one we've seen.
        meta = next((b.get("meta_cost") for b in reversed(parent_beats)
                     if b.get("meta_cost") is not None), None)
        if meta is not None:
            out.append(f"  meta_cost   : {_fmt_num(meta)}  (held-out improvement objective)")
        # model-stage beat carries the stage-1 pass/fail summary.
        model_beat = next((b for b in reversed(parent_beats)
                           if b.get("stage") == "model"), None)
        if model_beat is not None:
            out.append(f"  stage-1     : {_fmt_num(model_beat.get('start_cost'))} -> "
                       f"{_fmt_num(model_beat.get('best_cost'))}   "
                       f"pass={model_beat.get('passed')}")
    # self-edit accept/reject tally comes from the monotonicity log (authoritative).
    return out


def _selfedit_panel(mono):
    out = ["SELF-EDITS (monotonicity log - every attempt, sandboxed + invariant-gated)"]
    if not mono:
        out.append("  no edit attempts logged yet (run_logs/monotonicity.jsonl absent)")
        return out, 0
    accepts = [m for m in mono if m.get("accepted") is True or m.get("event") == "accept"]
    rejects = [m for m in mono if not (m.get("accepted") is True or m.get("event") == "accept")]
    # a sandbox crash / garbage edit is what the kill switch counts toward thrashing
    failures = [m for m in rejects if "crash" in str(m.get("reason", "")).lower()
                or "no result" in str(m.get("reason", "")).lower()]
    out.append(f"  attempts    : {len(mono)}   accepted={len(accepts)}   "
               f"rejected={len(rejects)}   (crash/garbage={len(failures)})")
    for m in accepts[-3:]:
        out.append(f"   + ACCEPT [{m.get('stage', '?')}] {m.get('descr', '')}  "
                   f"meta {_fmt_num(m.get('meta_cost_before'))} -> "
                   f"{_fmt_num(m.get('meta_cost_after'))}")
    for m in rejects[-3:]:
        out.append(f"   - reject [{m.get('stage', '?')}] {m.get('descr', '')}  "
                   f"({m.get('reason', 'no reason')})")
    return out, len(failures)


def _killswitch_panel(mono, crash_failures):
    """Infer kill-switch state from observable artifacts.

    The dashboard cannot import the live KillSwitch object, but it can read the two
    things that trip it: the manual STOP file, and the monotonicity log (a run of
    crash/garbage edits is the 'thrashing' tripwire; an invariant-floor breach shows as
    'invariant degraded' rejections)."""
    out = ["KILL SWITCH"]
    stop_path = os.path.join(RUN_LOGS_DIR, "STOP")
    if os.path.exists(stop_path):
        out.append(f"  *** STOPPED ***  manual STOP file present: {stop_path}")
    else:
        out.append("  manual STOP : not present (run_logs/STOP absent) — clear")
    # consecutive crash/garbage rejects at the tail = thrashing tripwire pressure
    streak = 0
    for m in reversed(mono):
        if m.get("accepted") is True or m.get("event") == "accept":
            break
        reason = str(m.get("reason", "")).lower()
        if "crash" in reason or "no result" in reason:
            streak += 1
        else:
            break
    if streak:
        out.append(f"  tripwire    : {streak} consecutive crash/garbage edit(s) at tail "
                   f"(loop may be thrashing)")
    inv_breaches = [m for m in mono if "invariant" in str(m.get("reason", "")).lower()]
    if inv_breaches:
        out.append(f"  invariant   : {len(inv_breaches)} edit(s) rejected for degrading the "
                   f"invariant (gate working)")
    if not streak and not inv_breaches and not os.path.exists(stop_path):
        out.append("  tripwires   : none detected — run looks healthy")
    return out


# ------------------------------------------------------------------------------- main
def render():
    """Build the full dashboard as a single string (so watch.py can reuse it)."""
    child = _read_jsonl(os.path.join(VITALS_DIR, "child.jsonl"))
    parent = _read_jsonl(os.path.join(VITALS_DIR, "parent.jsonl"))
    mono = _read_jsonl(os.path.join(RUN_LOGS_DIR, "monotonicity.jsonl"))
    summary = _read_json(os.path.join(RUN_LOGS_DIR, "closure_summary.json"))

    lines = []
    lines.append(_hr("═"))
    lines.append(f"  RecursiveNe vitals — {now_iso()}")
    lines.append(_hr("═"))
    lines.append("")
    lines.extend(_parent_panel(parent, summary))
    lines.append("")
    lines.extend(_child_panel(child))
    lines.append("")
    se_lines, crash_failures = _selfedit_panel(mono)
    lines.extend(se_lines)
    lines.append("")
    lines.extend(_killswitch_panel(mono, crash_failures))
    lines.append("")
    # closure summary footer (only present after a full run completes)
    if summary:
        lines.append(_hr())
        if summary.get("halted"):
            lines.append(f"  RUN HALTED: {summary['halted']}")
        passed = {k: v.get("passed") for k, v in (summary.get("stages") or {}).items()}
        lines.append(f"  closure_summary.json: stages passed = {passed}"
                     f"   elapsed={_fmt_num(summary.get('elapsed_s'))}s")
    else:
        lines.append(_hr())
        lines.append("  closure_summary.json: not written yet (run in progress or not started)")
    lines.append(_hr("═"))
    return "\n".join(lines)


def main():
    text = render()
    try:
        print(text)
    except UnicodeEncodeError:
        # last-resort: strip to ascii so we never die on a code-page mismatch
        print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
