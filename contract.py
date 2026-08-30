"""
contract.py

Universal interface every Guardian stream module must implement.

WINDOW FORMAT
-------------
Produced by data/loader.py, one dict per replay time step:

    window = {
        "t":    float,          # seconds since replay start
        "acc":  np.ndarray,     # shape (N, 3), units of g
        "gyro": np.ndarray,     # shape (N, 3), deg/s
        "hr":   float | None,   # bpm, scripted trace
        "temp": float | None,   # degrees C, scripted
        "fs":   50,             # Hz
    }

Windows are 2.5s at 50 Hz (N=125 samples), 50% overlap,
one window every 1.25s.

RULES
-----
- score() always returns a float in [0.0, 1.0]. Never None, never NaN.
- If a stream cannot compute a score, return 0.0 and set last_quality = 0.0.
- After every score() call, self.last_quality must be set, in [0.0, 1.0].
- Missing sensor data is represented as None -- never silently zeroed
  or interpolated. The quality stage is supposed to see gaps as gaps.
- No stream module imports another stream module. They only meet
  in core/fusion.py.
"""

from abc import ABC, abstractmethod


class Stream(ABC):
    """Base class for every evidence stream in the Guardian pipeline."""

    def __init__(self) -> None:
        # Every stream must expose this after score() is called.
        self.last_quality: float = 0.0

    def fit(self, df_normal):
        """
        Optional. Train the stream on a dataframe of "normal" windows
        (e.g. to build baselines). Default is a no-op -- not every
        stream needs training. Must return self.
        """
        return self

    @abstractmethod
    def score(self, window: dict) -> float:
        """
        Compute this stream's risk/confidence contribution for one window.

        Must return a float in [0.0, 1.0]. Never None, never NaN.
        Must also set self.last_quality in [0.0, 1.0] before returning.
        If the stream cannot compute (missing required data, etc.),
        return 0.0 and set self.last_quality = 0.0.
        """
        raise NotImplementedError