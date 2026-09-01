"""
WEDA-FALL adapter for Guardian.

Reads a WEDA-FALL accelerometer/gyroscope recording and converts it
into Guardian's existing Record format.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from data.loader import (
    Record,
    _acc_to_g,
    _gyro_to_dps,
    _resample,
    _est_fs,
    map_activity,
)


def read_wedafall(accel_path: Path) -> Record:
    accel_path = Path(accel_path)

    if not accel_path.name.endswith("_accel.csv"):
        raise ValueError(f"Expected an accelerometer file: {accel_path}")

    # Example:
    # U01_R01_accel.csv -> U01_R01_gyro.csv
    gyro_path = accel_path.with_name(
        accel_path.name.replace("_accel.csv", "_gyro.csv")
    )

    acc_df = pd.read_csv(accel_path)

    required_acc = [
        "accel_time_list",
        "accel_x_list",
        "accel_y_list",
        "accel_z_list",
    ]

    if not all(col in acc_df.columns for col in required_acc):
        raise ValueError(
            f"Unexpected WEDA-FALL accelerometer columns: "
            f"{list(acc_df.columns)}"
        )

    t = acc_df["accel_time_list"].to_numpy(dtype=float)
    acc = acc_df[
        ["accel_x_list", "accel_y_list", "accel_z_list"]
    ].to_numpy(dtype=float)

    notes = []

    # Normalize timestamps.
    t = t - t[0]

    fs_raw = _est_fs(t)

    # Convert acceleration to Guardian's g units.
    acc = _acc_to_g(acc, notes)

    # Resample acceleration to Guardian's 50 Hz grid.
    grid, acc_r = _resample(t, acc)

    # Gyroscope is optional in the Guardian contract.
    gyro_r = None

    if gyro_path.exists():
        gyro_df = pd.read_csv(gyro_path)

        required_gyro = [
            "gyro_time_list",
            "gyro_x_list",
            "gyro_y_list",
            "gyro_z_list",
        ]

        if all(col in gyro_df.columns for col in required_gyro):
            gyro_t = gyro_df["gyro_time_list"].to_numpy(dtype=float)
            gyro = gyro_df[
                ["gyro_x_list", "gyro_y_list", "gyro_z_list"]
            ].to_numpy(dtype=float)

            gyro_t = gyro_t - gyro_t[0]

            gyro = _gyro_to_dps(gyro, notes)

            gyro_r = np.empty(
                (len(grid), 3),
                dtype=np.float32,
            )

            for k in range(3):
                gyro_r[:, k] = np.interp(
                    grid,
                    gyro_t,
                    gyro[:, k],
                )
        else:
            notes.append(
                f"unexpected gyroscope columns in {gyro_path.name}"
            )
    else:
        notes.append("no matching gyroscope file")

    # Extract WEDA-FALL naming information.
    # Example: D01/U01_R01_accel.csv
    activity = accel_path.parent.name

    stem = accel_path.name.replace("_accel.csv", "")

    parts = stem.split("_")

    subject = parts[0] if parts else "unknown"
    trial = parts[1] if len(parts) > 1 else "unknown"

    # WEDA-FALL uses D01-D11 for ADLs and F01-F08 for falls.
    if activity.upper().startswith("F"):
        state = None
        is_fall = True
    elif activity.upper().startswith("D"):
        # WEDA-FALL ADLs are not directly named like Guardian's
        # generic activity map, so leave state unmapped for now.
        state = None
        is_fall = False
    else:
        state, is_fall = map_activity(activity)

    return Record(
        path=accel_path,
        subject=subject,
        activity=activity,
        trial=trial,
        t=grid,
        acc=acc_r,
        gyro=gyro_r,
        fs_raw=fs_raw,
        state=state,
        is_fall=is_fall,
        notes=notes,
    )