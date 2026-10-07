"""
streams/physiological.py -- heart-rate stream. PLACEHOLDER.

Owner: Mithuna. Not built yet.

What it will do: read heart rate (and HRV, if usable) from the band's PPG
sensor and say how unusual it looks, e.g. very high, very low or very
irregular. Thresholds will be population-level (no personal baseline) and
need sources. No SpO2.

Until it is built, this class follows contract.Stream but always returns
0.0 with last_quality = 0.0, so the rest of the pipeline can run and fusion
knows this stream has nothing to say.

The exact outputs fusion will read are defined in the module contract
(next step after the repo cleanup).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Stream  # noqa: E402


class PhysiologicalStream(Stream):
    def score(self, window: dict) -> float:
        self.last_quality = 0.0
        return 0.0
