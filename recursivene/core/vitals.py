"""Vitals — heartbeats for the PARENT (the orchestrator / improvement loop) and the
CHILD (the RSI learner being improved). Protected.

Every generation the loop writes one stamped line per entity to vitals/<entity>.jsonl,
so at any moment you can answer 'is the parent alive, what generation, what is the
child's competence/cost right now, and is the trend healthy?' without attaching a
debugger. Keeping vitals on both yourself and your RSI child is what makes a long
autonomous run observable instead of a black box."""

import json
import os

from .clock import stamp


class Vitals:
    def __init__(self, vitals_dir):
        self.dir = vitals_dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, entity):
        return os.path.join(self.dir, f"{entity}.jsonl")

    def beat(self, entity, **fields):
        """Append one timestamped heartbeat record for `entity` (e.g. 'parent','child')."""
        rec = stamp({"entity": entity, **fields})
        with open(self._path(entity), "a") as f:
            f.write(json.dumps(rec) + "\n")
        # also keep a 'latest' snapshot for cheap status reads
        with open(os.path.join(self.dir, f"{entity}.latest.json"), "w") as f:
            json.dump(rec, f, indent=2)
        return rec

    def last(self, entity):
        p = os.path.join(self.dir, f"{entity}.latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)
