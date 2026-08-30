"""
core/fusion.py

REAL fusion implementation (Review 1 scope): plain weighted average of
three evidence streams -- motion, context, physiology. No quality
weighting in this version; streams/quality.py is not one of the three
evidence streams used here (that's a later-stage addition, not part
of this review's fusion).

WEIGHTS
-------
Centralized below so they're tunable in one place.

    risk = sum(weight_i * score_i) / sum(weight_i)   [over valid streams]

Invalid/missing scores (None, NaN, out of [0,1]) are excluded from
both numerator and denominator -- they degrade the result gracefully
rather than dragging risk toward zero.

SUPPRESSION RULE
-----------------
When context reports "seated hand activity" with confidence > 0.7,
AND no impact has just occurred (motion_stream.time_since_impact is
None/unset), the motion CONTRIBUTION (not the raw score, and not the
denominator weight) is halved. Halving only the numerator -- while the
denominator keeps the full original weight -- is what actually lowers
the overall risk average; this is the mechanism behind demo scenario 1
(a hard hand impact during seated activity gets suppressed).

The impact guard is critical: if time_since_impact IS set (an impact
was just detected), suppression never applies, even if context
mislabels the state as seated hand activity. Motion evidence after a
real impact must never be silently discounted.

INTERFACE
---------
fuse(scores, context_stream, motion_stream) -> (risk, contributions)

- scores: dict with keys "motion", "context", "physiology" -> float,
  or None/NaN for missing/invalid.
- context_stream / motion_stream: the actual stream OBJECTS, not their
  scores -- fusion needs attributes beyond the score itself:
  ContextStream.last_state, ContextStream.last_confidence,
  MotionStream.time_since_impact. Context/motion metadata is read from
  these objects directly; it is never smuggled into the window dict.
- Both objects are read via getattr() with safe defaults, because
  context.py/motion.py may still be stub-stage (no last_state /
  last_confidence / time_since_impact attributes yet) -- fusion must
  not crash before the real branch is merged. In that case suppression
  simply never triggers (last_state defaults to None, which never
  equals "seated hand activity").

Returns:
- risk: float in [0,1]
- contributions: dict of the EXACT numbers used in the weighted sum
  (post-suppression), keyed like scores. core/explain.py must use
  these values directly rather than recomputing anything.
"""

import math

WEIGHTS = {
    "motion": 0.50,
    "context": 0.20,
    "physiology": 0.30,
}

SUPPRESSION_CONTEXT_STATE = "seated hand activity"
SUPPRESSION_CONFIDENCE_THRESHOLD = 0.7
SUPPRESSION_FACTOR = 0.5


def _is_valid_score(value) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and 0.0 <= v <= 1.0


def fuse(scores: dict, context_stream=None, motion_stream=None):
    """
    Combine motion/context/physiology scores into a single risk value.

    Returns (risk: float in [0,1], contributions: dict).
    contributions is empty ({}) when no stream had a usable score.
    """
    raw_contributions = {}
    included_weight = 0.0

    for name, weight in WEIGHTS.items():
        value = scores.get(name) if scores else None
        if not _is_valid_score(value):
            continue
        raw_contributions[name] = weight * float(value)
        included_weight += weight

    if included_weight == 0.0:
        return 0.0, {}

    last_state = getattr(context_stream, "last_state", None)
    last_confidence = getattr(context_stream, "last_confidence", 0.0) or 0.0
    time_since_impact = getattr(motion_stream, "time_since_impact", None)

    suppression_active = (
        "motion" in raw_contributions
        and last_state == SUPPRESSION_CONTEXT_STATE
        and last_confidence > SUPPRESSION_CONFIDENCE_THRESHOLD
        and time_since_impact is None
    )

    contributions = dict(raw_contributions)
    if suppression_active:
        contributions["motion"] = raw_contributions["motion"] * SUPPRESSION_FACTOR

    risk = sum(contributions.values()) / included_weight
    risk = max(0.0, min(1.0, risk))

    return risk, contributions