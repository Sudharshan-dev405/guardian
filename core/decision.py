"""
core/decision.py

STUB. Real implementation (Stage 6) will be a 7-state machine
(normal / motion event / possible incident / post-event monitoring /
physiological confirmation / escalate / sensor uncertainty) with
hysteresis (entry 0.55, exit 0.40) and a 30s dwell before escalating.
Manual SOS will bypass all of it.

For now: a single threshold, no hysteresis, no dwell, no memory
between calls -- just enough to produce a state string for the
dashboard to display.
"""

THRESHOLD_STUB = 0.5


def decide_stub(risk: float) -> str:
    return "escalate" if risk >= THRESHOLD_STUB else "normal"