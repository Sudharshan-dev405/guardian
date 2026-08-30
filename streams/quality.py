"""
streams/quality.py

STUB. Real implementation (Sudharshan, Stage 4) will compute PPG
indices (skewness, perfusion index, template-match correlation) and
IMU indices (gravity magnitude, clipping, dropout), each mapped to
[0,1] via piecewise-linear functions with thresholds centralized in
one dict.

Note: "quality" here is itself a Stream in the fusion sense --
it produces a sensor-confidence score/contribution, separate from
each individual stream's own self.last_quality.

For now: deterministic constant so fusion/dashboard can be built
against a stable value.
"""

from contract import Stream


class QualityStream(Stream):
    def score(self, window: dict) -> float:
        self.last_quality = 1.0
        return 0.5