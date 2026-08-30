"""
streams/physiology.py

REAL implementation of the physiology (heart-rate deviation) stream.
Rule-based, not ML.

DATA SOURCE
-----------
Public wrist fall-detection datasets (UMAFall, FallAllD) do not include
heart-rate channels. Heart rate here is meant to come from a SCRIPTED
trace (see data/generate_hr_trace.py -> data/hr_trace.csv), stitched
into window["hr"] by the replay loader.

KNOWN GAP (not fixed here -- out of scope for physiology.py):
data/loader.py's replay_stub() currently hardcodes hr=72.0 for every
window and does NOT read data/hr_trace.csv. The CSV's shape (t, hr
columns; resting -> rise -> peak -> recovery) is compatible with this
module, but nothing in the pipeline exercises it yet. Flagged for
whoever wires loader.py during integration.

Wherever HR is displayed (dashboard, report), it must be labelled
"scripted physiological trace -- hardware pending".

SCORING MODEL
-------------
No personalised baseline exists yet (core/baseline.py is a separate
component), so this stream uses FIXED resting-range thresholds:
50-100 bpm is normal. Deviation outside that range maps to a
continuous risk value via a saturating exponential -- never a hard
binary cutoff -- scaled by DEVIATION_SCALE_BPM.

A single noisy spike must not trigger a high score, so raw deviation
risk is additionally weighted by how long the deviation has been
sustained (dwell), ramping from 0 to full weight over DWELL_SECONDS
(~30s). Two independent axes -- how far outside normal, and for how
long -- are multiplied together.

MOTION GATING
-------------
This module does NOT import streams/context.py (streams only meet in
core/fusion.py, per the contract). It also does NOT take context as a
separate score() argument -- that broke the Stream contract's fixed
score(window) -> float signature and made the gate unreachable from
any generic per-stream loop (dashboard, fusion, shared test harness)
that calls score(window) uniformly across all four streams.

Instead, context state is read from window.get("context_state") if
present. The window dict produced by data/loader.py does not currently
carry this key -- it is expected to be injected by whoever assembles
the window for this stream (fusion, in the real pipeline), by copying
the loader's window and adding "context_state" before calling score().
If the key is absent, context_state is None and gating does not apply
(matches the base contract: physiology behaves standalone/testable
without needing a context stream present).

When context_state == "ambulating": return 0.0 / last_quality 0.0.
Wrist PPG/HR is not trustworthy during movement. The dwell clock is
left untouched (paused, not reset) during ambulating windows, so a
brief walk doesn't erase genuine sustained elevation on either side
of it.

INTEGRATION NOTE: streams/context.py, at stub stage, does not yet
expose a state-label attribute analogous to last_quality. Whoever owns
context.py needs to expose one before fusion can read it and inject
"context_state" into windows passed to this stream.

Do NOT use SpO2 as a trigger: peripheral SpO2 stays flat through
fainting; it is cerebral oxygenation that drops, and that cannot be
measured from a wrist sensor.
"""

from __future__ import annotations
import math

from contract import Stream


PHYSIOLOGY_THRESHOLDS = {
    "resting_low_bpm": 50.0,
    "resting_high_bpm": 100.0,
    "deviation_scale_bpm": 12.0,   # larger = risk ramps up more slowly with distance
    "dwell_seconds": 30.0,         # time for a sustained deviation to count fully
    "default_dt": 1.25,            # assumed spacing on the very first window
    "max_dt": 5.0,                 # sanity cap against replay time-gaps
}


def _deviation_risk(hr: float) -> float:
    """Continuous (non-binary) risk from HR distance outside the resting range."""
    low = PHYSIOLOGY_THRESHOLDS["resting_low_bpm"]
    high = PHYSIOLOGY_THRESHOLDS["resting_high_bpm"]
    scale = PHYSIOLOGY_THRESHOLDS["deviation_scale_bpm"]

    dist = max(low - hr, hr - high, 0.0)
    if dist <= 0.0:
        return 0.0
    return 1.0 - math.exp(-dist / scale)


def _is_valid_hr(hr) -> bool:
    """None, NaN, inf, non-numeric, and non-positive values are all invalid."""
    if hr is None:
        return False
    try:
        val = float(hr)
    except (TypeError, ValueError):
        return False
    return math.isfinite(val) and val > 0.0


class PhysiologyStream(Stream):
    def __init__(self) -> None:
        super().__init__()
        self._sustained_seconds = 0.0
        self._last_t = None

    def score(self, window: dict) -> float:
        hr = window.get("hr")
        context_state = window.get("context_state")  # optional; may be absent
        t = window.get("t")

        # Missing/invalid HR: never guess. Reset dwell so a gap doesn't
        # quietly count toward "sustained" once valid data returns.
        if not _is_valid_hr(hr):
            self._sustained_seconds = 0.0
            self._last_t = t if t is not None else self._last_t
            self.last_quality = 0.0
            return 0.0

        # Motion gating: wrist PPG/HR is unreliable while moving.
        # Dwell is paused, not reset -- a brief walk shouldn't erase
        # genuine sustained elevation either side of it.
        if context_state == "ambulating":
            self._last_t = t if t is not None else self._last_t
            self.last_quality = 0.0
            return 0.0

        hr = float(hr)

        if t is not None and self._last_t is not None:
            dt = t - self._last_t
        else:
            dt = PHYSIOLOGY_THRESHOLDS["default_dt"]
        dt = max(0.0, min(dt, PHYSIOLOGY_THRESHOLDS["max_dt"]))
        self._last_t = t

        raw_risk = _deviation_risk(hr)

        if raw_risk > 0.0:
            self._sustained_seconds += dt
        else:
            self._sustained_seconds = 0.0

        dwell_fraction = min(1.0, self._sustained_seconds / PHYSIOLOGY_THRESHOLDS["dwell_seconds"])
        score = raw_risk * dwell_fraction
        score = max(0.0, min(1.0, score))

        if math.isnan(score):
            self.last_quality = 0.0
            return 0.0

        self.last_quality = 1.0
        return score