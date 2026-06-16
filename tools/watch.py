"""tools/watch.py — live-watch wrapper around tools/dashboard.py.

Re-renders the vitals panel every N seconds (default 5) until interrupted, clearing the
screen between frames so you get a stable, updating view of the closure loop while it
runs. Pure stdlib, read-only; delegates all reading/formatting to dashboard.render() so
there is exactly one source of truth for the panel.

Usage (from repo root):
  python tools/watch.py            # refresh every 5s
  python tools/watch.py 2          # refresh every 2s
  python tools/watch.py --once     # render a single frame and exit (same as dashboard.py)

Ctrl-C to stop.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dashboard  # noqa: E402  (local sibling module)


def _clear():
    """Clear the terminal. ANSI works on modern Windows Terminal / PowerShell and *nix;
    os.system fallback covers older consoles."""
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()
    if os.name == "nt":
        # ANSI may be off on legacy conhost; cls is cheap and harmless otherwise.
        os.system("")  # enables VT processing on Win10+ without spawning a new console


def main(argv):
    interval = 5.0
    once = False
    for arg in argv:
        if arg in ("--once", "-1"):
            once = True
        else:
            try:
                interval = max(0.5, float(arg))
            except ValueError:
                print(f"ignoring unrecognized arg: {arg}")
    if once:
        print(dashboard.render())
        return 0
    try:
        while True:
            _clear()
            try:
                print(dashboard.render())
            except UnicodeEncodeError:
                text = dashboard.render()
                print(text.encode("ascii", "replace").decode("ascii"))
            print(f"\n(refreshing every {interval:g}s — Ctrl-C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
