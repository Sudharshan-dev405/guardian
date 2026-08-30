"""
streams/motion.py

STUB. Real implementation (Stage: Pranavah) will be a two-stage
impact-gate + calibrated fall-vs-ADL classifier, combined with
post-impact stillness scoring.

For now: deterministic constant so fusion/dashboard can be built
against a stable value.
"""

from contract import Stream


class MotionStream(Stream):
    def score(self, window: dict) -> float:
        self.last_quality = 0.9
        return 0.6