"""
core/pipeline.py -- runs every available stream on each window (Stage 5.1).

This is the plug-in point for all modules. A stream joins the pipeline by
adding one line to STREAMS. Nothing else changes: whatever a stream exposes
as last_* attributes (besides last_quality) is picked up automatically and
stored under 'extras', so new modules do not need a new database column.

One record per stream per window:
    {"t": seconds, "stream": "motion", "score": 0..1, "quality": 0..1,
     "extras": {"impact": .., "stillness": .., ...}}

A stream that crashes or returns something invalid gives score 0 and
quality 0 for that window (contract.py rules); the run carries on.

Fusion (Stage 7) will read these records. The pipeline itself makes no
decision and raises no alert.

    py -m core.pipeline            # which streams load, and from which model
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# name: (module, class, saved model or None). Stage 6 swaps or adds lines here.
STREAMS = {
    "motion": ("streams.motion", "MotionStream", "models/motion.joblib"),
    "activity": ("streams.context", "ContextStream", "models/context.joblib"),
    "physiological": ("streams.physiological", "PhysiologicalStream", None),
}

EXTRA_ATTRS = ("gate_open", "time_since_impact")      # besides every last_*
_SKIP = object()


def _jsonable(v):
    """Plain JSON value, or _SKIP for things that should not be stored."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, dict):
        out = {str(k): _jsonable(x) for k, x in v.items()}
        return {k: x for k, x in out.items() if x is not _SKIP}
    if isinstance(v, (list, tuple)) and len(v) <= 16:
        out = [_jsonable(x) for x in v]
        return [x for x in out if x is not _SKIP]
    return _SKIP                                      # arrays, models, ...


def _extras(stream) -> dict:
    out = {}
    for name, v in vars(stream).items():
        if (name.startswith("last_") and name != "last_quality") or name in EXTRA_ATTRS:
            j = _jsonable(v)
            if j is not _SKIP:
                out[name[5:] if name.startswith("last_") else name] = j
    return out


def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


class Pipeline:
    """Load the streams once, then call step(window) for every window."""

    def __init__(self, streams: dict | None = None, root: Path = ROOT):
        self.streams, self.status = {}, {}
        for name, (mod, cls, model) in (streams or STREAMS).items():
            try:
                C = getattr(importlib.import_module(mod), cls)
                path = root / model if model else None
                if path is not None and path.exists():
                    self.streams[name] = C.load(str(path))
                    self.status[name] = f"loaded {model}"
                else:
                    self.streams[name] = C()
                    self.status[name] = ("no model file, running untrained" if model
                                         else "placeholder (no model)")
            except Exception as e:                                # noqa: BLE001
                self.status[name] = f"NOT LOADED: {e}"

    def step(self, window: dict) -> list[dict]:
        out = []
        for name, s in self.streams.items():
            try:
                score = float(s.score(window))
                q = float(getattr(s, "last_quality", 0.0))
                if not (math.isfinite(score) and math.isfinite(q)):
                    score, q = 0.0, 0.0
                ex = _extras(s)
            except Exception as e:                                # noqa: BLE001
                score, q, ex = 0.0, 0.0, {"error": str(e)}
            out.append(dict(t=float(window["t"]), stream=name, score=_clip01(score),
                            quality=_clip01(q), extras=ex))
        return out

    def describe(self) -> str:
        return "\n".join(f"  {k:14s} {v}" for k, v in self.status.items())


if __name__ == "__main__":
    print("streams:\n" + Pipeline().describe())
