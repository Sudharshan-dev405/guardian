"""
streams/context.py

STUB. Real implementation (Stage: Pranavah) will be a 4-state
classifier (stationary / ambulating / seated hand activity /
lying-immobile) trained on mapped dataset activity labels.

For now: deterministic constant so fusion/dashboard can be built
against a stable value.
"""

from contract import Stream


class ContextStream(Stream):
    def score(self, window: dict) -> float:
        self.last_quality = 0.8
        return 0.3