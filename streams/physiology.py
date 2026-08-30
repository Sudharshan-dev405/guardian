"""
streams/physiology.py

STUB. Real implementation (Stage: Mithuna) will score deviation of
a scripted HR trace from a per-context/time-of-day robust baseline,
and will return 0.0 / quality 0.0 whenever context is "ambulating"
(HRV is unreliable during motion).

For now: deterministic constant so fusion/dashboard can be built
against a stable value.
"""

from contract import Stream


class PhysiologyStream(Stream):
    def score(self, window: dict) -> float:
        self.last_quality = 0.7
        return 0.2