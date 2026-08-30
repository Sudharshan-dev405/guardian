"""
Generate the scripted HR trace used by Guardian's physiology stream.

IMPORTANT:
This is NOT real physiological sensor data.

The public wrist datasets used by Guardian do not contain heart-rate
measurements, so this trace is used only as a clearly labelled placeholder
until hardware is available.

Scenario:
    ~72 bpm resting
    -> gradual rise toward ~105 bpm
    -> recovery toward ~72 bpm
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Guardian replay:
# 50 Hz sampling, 62-sample hop = 1.24 seconds.
WINDOW_INTERVAL_S = 62 / 50

# Total duration of the scripted trace.
DURATION_S = 150.0

# Reproducible noise.
RANDOM_SEED = 42


def generate_hr_trace() -> pd.DataFrame:
    """Generate the scripted HR trace."""

    rng = np.random.default_rng(RANDOM_SEED)

    times = np.arange(
        0.0,
        DURATION_S + WINDOW_INTERVAL_S,
        WINDOW_INTERVAL_S,
    )

    hr = np.full(len(times), 72.0)

    for i, t in enumerate(times):

        # Resting period.
        if t < 40:
            target = 72.0

        # Gradual rise toward ~105 bpm.
        elif t < 70:
            progress = (t - 40.0) / 30.0
            target = 72.0 + progress * (105.0 - 72.0)

        # Sustained elevated HR.
        elif t < 100:
            target = 105.0

        # Recovery.
        elif t < 125:
            progress = (t - 100.0) / 25.0
            target = 105.0 - progress * (105.0 - 72.0)

        # Resting again.
        else:
            target = 72.0

        # Small realistic measurement noise.
        hr[i] = target + rng.normal(0.0, 1.5)

    return pd.DataFrame(
        {
            "t": times,
            "hr": hr,
        }
    )


def main() -> None:
    output_path = Path(__file__).with_name("hr_trace.csv")

    df = generate_hr_trace()
    df.to_csv(output_path, index=False)

    print(f"Generated: {output_path}")
    print(f"Samples: {len(df)}")
    print(f"Duration: {df['t'].iloc[-1]:.2f} seconds")
    print(f"HR range: {df['hr'].min():.1f}–{df['hr'].max():.1f} bpm")


if __name__ == "__main__":
    main()