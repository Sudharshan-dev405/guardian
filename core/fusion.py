"""
core/fusion.py -- combines the evidence streams into one decision. PLACEHOLDER.

Owner: Pranavah. Designed, not built (Stage 7).
Full design: core/FUSION_DESIGN.md

Inputs (one set per window): activity (streams/context.py), motion
(streams/motion.py), physiological (streams/physiological.py), and context
derived from those three. Each gives a score in [0, 1] and a quality.

It has to cover every emergency a wrist band can reasonably see, not just
falls (full table in testing/outputs/motion_module_log.txt):
    hard fall, long lie, faint or soft collapse, unresponsive without a fall,
    heart-rate emergency, seizure (needs its own detector), overheating or
    cold (skin temperature trend, weak), and the SOS button.

Rules already fixed:
    - no emergency path depends on the motion gate alone
    - every alert says which pattern triggered it
    - no SpO2 claims, no personal baseline
"""


def fuse(streams: dict) -> dict:
    raise NotImplementedError("fusion is not built yet")
