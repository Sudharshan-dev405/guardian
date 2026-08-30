"""
data/loader.py

STUB. Real implementation (Pranavah) will replay real public wrist
dataset files (UMAFall / FallAllD) as a stream of windows, matching
the format in contract.py.

For now: yields a fixed number of synthetic windows so the rest of
the pipeline (streams -> fusion -> decision -> explain -> dashboard)
can be built and tested without real data.

IMPORTANT: this is synthetic data for wiring/testing the PIPELINE,
not for evaluation. Evaluation must use real datasets (see build plan).
"""

import numpy as np


def replay_stub(n_windows: int = 20):
    """Yield n_windows fake windows, one at a time, matching the contract."""
    t = 0.0
    for i in range(n_windows):
        window = {
            "t": t,
            "acc": np.zeros((125, 3)),
            "gyro": np.zeros((125, 3)),
            "hr": 72.0,
            "temp": 36.5,
            "fs": 50,
        }
        yield window
        t += 1.25  # one window every 1.25s, per the contract