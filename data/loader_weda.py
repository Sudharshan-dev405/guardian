"""
data/loader_weda.py -- WEDA-FALL loader, same Record and window contract as
data/loader.py (50 Hz, acc in g, gyro in deg/s).

What the dataset is (from its README):
- Fitbit Sense smartwatch on the wrist, recorded at 50 Hz. The 40/25/10/5 Hz
  folders are derived from the 50 Hz data, so only dataset/50Hz is read.
- dataset/50Hz/<code>/U<user>_R<trial>_<sensor>.csv, sensor = accel, gyro,
  orientation, vertical_accel. We read accel and gyro.
  Columns: <sensor>_time_list (s), <sensor>_x_list, _y_list, _z_list.
- Falls F01-F08, ADLs D01-D11. Every fall was onto a mattress.
- Young participants U01-U14 did falls and ADLs. Elder participants U21-U31
  (ages 77-95) did ADLs only (D01, D03, D04, D09, D10, D11).
- F06-F08 are falls while sitting "caused by fainting or falling asleep".

Units (checked by --scan, not assumed silently):
- accelerometer in m/s^2 (rows at rest read ~9.8) -> divided by 9.80665
- gyroscope in rad/s (Fitbit sensor API) -> multiplied by 180/pi.
  Decided ONCE for the whole dataset; --scan prints the fall peak so it can
  be checked (a fall should reach hundreds of deg/s, not tens of thousands).
- Timestamps arrive in bursts (Fitbit batches), so samples are spread evenly
  over each file's time span (TIMING = "even"), then put on the uniform 50 Hz
  grid. See the TIMING note below for the measurements behind this.

CLI (repo root):
    py -m data.loader_weda --scan
    py -m data.loader_weda --inspect data\\raw\\WEDA-FALL\\dataset\\50Hz\\F07\\U02_R03_accel.csv
    py -m data.loader_weda --mapping
    py -m data.loader_weda --timing
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import Record, iter_windows, replay, TARGET_FS  # noqa: E402,F401

DEFAULT_ROOT = Path(r"data/raw/WEDA-FALL/dataset/50Hz")
G_MS2 = 9.80665
GYRO_RAD_TO_DEG = 180.0 / np.pi       # dataset-wide unit decision, see docstring
# How sample times are taken. Measured with --timing on all 969 files:
#   rows / timestamp span = 49.6 Hz (median), but 68% of the gaps between
#   stamps are under 5 ms -> stamps are batch delivery times, not sample times.
#   Fall peaks: 4.97 g with the stamps, 5.69 g with even spacing (the stamps
#   smear the impact).
# "even" (used): spread the samples evenly over the recording's own time span.
#   Keeps each file's true length (some run nearer 40 Hz) and removes the bursts.
# "index": assume exactly 50 Hz. "timestamps": trust the file.
TIMING = "even"
NOMINAL_FS = 50.0
NAME_RE = re.compile(r"^U(\d+)_R(\d+)_accel\.csv$", re.I)   # not *_vertical_accel

# code -> (name, Guardian state or None for falls, is_fall, direction, cause)
ACTIVITIES = {
    "F01": ("forward fall while walking (slip)", None, True, "forward", "slip"),
    "F02": ("lateral fall while walking (slip)", None, True, "lateral", "slip"),
    "F03": ("backward fall while walking (slip)", None, True, "backward", "slip"),
    "F04": ("forward fall while walking (trip)", None, True, "forward", "trip"),
    "F05": ("backward fall trying to sit down", None, True, "backward", "sit-down"),
    "F06": ("forward fall while sitting (faint or asleep)", None, True, "forward", "faint"),
    "F07": ("backward fall while sitting (faint or asleep)", None, True, "backward", "faint"),
    "F08": ("lateral fall while sitting (faint or asleep)", None, True, "lateral", "faint"),
    "D01": ("walking", "ambulating", False, "", ""),
    "D02": ("jogging", "ambulating", False, "", ""),
    "D03": ("walking up and downstairs", "ambulating", False, "", ""),
    "D04": ("sitting on a chair, wait, get up", "stationary", False, "", ""),
    "D05": ("attempt to get up, collapse into chair", "stationary", False, "", ""),
    "D06": ("crouching to tie shoes, get up", "stationary", False, "", ""),
    "D07": ("stumble while walking", "ambulating", False, "", ""),
    "D08": ("gentle jump to reach high object", "ambulating", False, "", ""),
    "D09": ("hit table with hand", "seated hand activity", False, "", ""),
    "D10": ("clapping hands", "seated hand activity", False, "", ""),
    "D11": ("opening and closing door", "seated hand activity", False, "", ""),
}
MAPPING_NOTES = {
    "D05": "JUDGEMENT CALL: a hard sit is a postural transition with a static base; "
           "kept as stationary like UMAFall sitting. It is also a hard negative for falls.",
    "D07": "JUDGEMENT CALL: stumble without falling happens during gait -> ambulating.",
    "D08": "whole-body jump, mapped like UMAFall hopping -> ambulating.",
    "D11": "standing, not seated; kept as 'seated hand activity' for consistency with UMAFall.",
}


def mapping_table_markdown() -> str:
    rows = ["| WEDA code | Activity | Guardian state | Note |", "|---|---|---|---|"]
    for code, (name, state, is_fall, *_rest) in ACTIVITIES.items():
        st = "(fall - motion stream)" if is_fall else state
        rows.append(f"| {code} | {name} | {st} | {MAPPING_NOTES.get(code, '')} |")
    return "\n".join(rows)


def _read_raw(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path.name}: expected 4 columns, got {data.shape}")
    return data[~np.isnan(data).any(axis=1)]


def _read_xyz(path: Path, timing=None):
    data = _read_raw(path)
    timing = timing or TIMING
    if timing == "index":                         # file order, fixed 50 Hz spacing
        t = np.arange(len(data)) / NOMINAL_FS
        return t, data[:, 1:4]
    if timing == "even":                          # file order, even over the real span
        t0, t1 = float(data[:, 0].min()), float(data[:, 0].max())
        t = np.linspace(t0, t1, len(data)) if t1 > t0 else np.arange(len(data)) / NOMINAL_FS
        return t, data[:, 1:4]
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    t, keep = np.unique(data[:, 0], return_index=True)   # drop duplicate stamps
    return t, data[keep, 1:4]


def _est_fs(t):
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else float("nan")


def read_record(acc_path, timing=None) -> Record:
    acc_path = Path(acc_path)
    m = NAME_RE.search(acc_path.name)
    code = acc_path.parent.name.upper()
    if not m or code not in ACTIVITIES:
        raise ValueError(f"{acc_path}: not a WEDA <code>/U<id>_R<n>_accel.csv file")
    user, trial = int(m.group(1)), m.group(2)
    name, state, is_fall, direction, cause = ACTIVITIES[code]
    notes = []

    t_a, acc = _read_xyz(acc_path, timing)
    if len(t_a) < 10:
        raise ValueError(f"{acc_path.name}: only {len(t_a)} accelerometer rows")
    fs_raw = _est_fs(t_a)
    acc = acc / G_MS2

    t0 = t_a[0]
    t_end = t_a[-1]
    gyro_path = acc_path.with_name(acc_path.name.replace("_accel", "_gyro"))
    if gyro_path.exists():
        t_g, gyr = _read_xyz(gyro_path, timing)
        gyr = gyr * GYRO_RAD_TO_DEG
        t0, t_end = max(t0, t_g[0]), min(t_end, t_g[-1])
    else:
        t_g, gyr = None, None
        notes.append("no gyroscope file for this trial")

    if t_end - t0 < 1.0:
        raise ValueError(f"{acc_path.name}: accel/gyro overlap under 1 s")
    grid = np.arange(t0, t_end, 1.0 / TARGET_FS)
    acc_r = np.column_stack([np.interp(grid, t_a, acc[:, k]) for k in range(3)]).astype(np.float32)
    gyro_r = (np.column_stack([np.interp(grid, t_g, gyr[:, k]) for k in range(3)]).astype(np.float32)
              if gyr is not None else None)

    activity = f"{code} {name}"          # contains forward/backward/lateral for falls
    rec = Record(path=acc_path, subject=f"U{user:02d}", activity=activity, trial=trial,
                 t=grid - grid[0], acc=acc_r, gyro=gyro_r, fs_raw=fs_raw,
                 state=state, is_fall=is_fall, notes=notes)
    rec.code, rec.direction, rec.cause = code, direction, cause
    rec.elder = user >= 21
    return rec


def scan(root=DEFAULT_ROOT, limit=None, subjects=None, codes=None):
    """Yield Records for every accelerometer file under root (50 Hz folder)."""
    root = Path(root)
    files = sorted(f for f in root.glob("*/U*_R*_accel.csv") if NAME_RE.match(f.name))
    if codes is not None:
        files = [f for f in files if f.parent.name.upper() in set(codes)]
    if subjects is not None:
        keep = set(subjects)
        files = [f for f in files if (m := NAME_RE.search(f.name))
                 and f"U{int(m.group(1)):02d}" in keep]
    if limit:
        files = files[:limit]
    for f in files:
        try:
            yield read_record(f)
        except Exception as e:                       # noqa: BLE001
            print(f"[skip] {f.parent.name}/{f.name}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _inspect(path):
    p = Path(path)
    print(f"--- first 8 lines of {p.name} ---")
    with open(p, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 8:
                break
            print(f"{i:3d} | {line.rstrip()}")
    rec = read_record(p)
    print(rec.summary())
    print(f"  direction={rec.direction} cause={rec.cause} elder={rec.elder}")
    print(f"  windows: {sum(1 for _ in iter_windows(rec))}")


def _scan_report(root):
    per = defaultdict(lambda: dict(falls=0, adls=0, faint_falls=0))
    durations, fs, med_g, max_g, fall_gyro, adl_gyro = [], [], [], [], [], []
    for rec in scan(root):
        d = per[rec.subject]
        d["falls" if rec.is_fall else "adls"] += 1
        d["faint_falls"] += int(rec.cause == "faint")
        durations.append(rec.duration)
        fs.append(rec.fs_raw)
        mag = np.linalg.norm(rec.acc, axis=1)
        med_g.append(float(np.median(mag)))
        max_g.append(float(mag.max()))
        if rec.gyro is not None:
            pk = float(np.percentile(np.abs(rec.gyro), 99.5))
            (fall_gyro if rec.is_fall else adl_gyro).append(pk)

    print("subject  falls  faint-type  ADLs")
    for s in sorted(per):
        d = per[s]
        print(f"  {s}    {d['falls']:4d}   {d['faint_falls']:4d}       {d['adls']:4d}")
    n_f = sum(d["falls"] for d in per.values())
    n_a = sum(d["adls"] for d in per.values())
    print(f"\nrecords {n_f + n_a}  (falls {n_f}, ADLs {n_a})  subjects {len(per)}  "
          f"with falls {sum(1 for d in per.values() if d['falls'])}")
    print(f"duration s: median {np.median(durations):.1f}, min {min(durations):.1f}, "
          f"max {max(durations):.1f}")
    print(f"raw rate Hz: median {np.median(fs):.1f}, min {min(fs):.1f}, max {max(fs):.1f}")
    print(f"UNIT CHECK accel: median |a| per record, median {np.median(med_g):.3f} g "
          f"(should be ~1.0)")
    print(f"           peak |a|: median {np.median(max_g):.2f} g, max {max(max_g):.2f} g "
          f"(a flat ceiling here would mean clipping)")
    if fall_gyro:
        print(f"UNIT CHECK gyro: 99.5th-pct |w| per record, falls median "
              f"{np.median(fall_gyro):.0f} deg/s, ADLs median {np.median(adl_gyro):.0f} deg/s "
              f"(falls should be in the hundreds)")


def _timing_report(root, limit=None):
    """Are the timestamps real sample times? Compare count/duration with the
    gaps between stamps, and how sharp fall peaks look under each timing."""
    files = sorted(f for f in Path(root).glob("*/U*_R*_accel.csv") if NAME_RE.match(f.name))
    if limit:
        files = files[:limit]
    eff, med_dt, burst, dup, span, n_rows = [], [], [], [], [], []
    peaks = {"timestamps": [], "index": [], "even": []}
    for f in files:
        d = _read_raw(f)
        t = d[:, 0]
        dt = np.diff(t)
        n_rows.append(len(t))
        span.append(t[-1] - t[0])
        eff.append(len(t) / (t[-1] - t[0]) if t[-1] > t[0] else np.nan)
        med_dt.append(float(np.median(dt)))
        burst.append(float(np.mean(dt < 0.005)))
        dup.append(float(np.mean(dt <= 0)))
        if f.parent.name.startswith("F"):
            for mode in peaks:
                try:
                    r = read_record(f, timing=mode)
                    peaks[mode].append(float(np.linalg.norm(r.acc, axis=1).max()))
                except Exception:  # noqa: BLE001
                    pass
    q = lambda x: f"median {np.nanmedian(x):.3f}  p10 {np.nanpercentile(x, 10):.3f}  p90 {np.nanpercentile(x, 90):.3f}"  # noqa: E731
    print(f"{len(files)} accelerometer files")
    print(f"rows per file            {q(n_rows)}")
    print(f"timestamp span (s)       {q(span)}")
    print(f"rows / span (Hz)         {q(eff)}   <- ~50 means 50 samples per real second")
    print(f"median gap (s)           {q(med_dt)}   <- 0.020 for even 50 Hz")
    print(f"share of gaps < 5 ms     {q(burst)}   <- high = samples arrive in bursts")
    print(f"share of gaps <= 0       {q(dup)}")
    for mode, v in peaks.items():
        if v:
            print(f"fall peak |a| with '{mode}' timing: median {np.median(v):.2f} g")


def main(argv=None):
    ap = argparse.ArgumentParser(description="WEDA-FALL loader")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--inspect", metavar="ACC_FILE")
    ap.add_argument("--mapping", action="store_true")
    ap.add_argument("--timing", action="store_true", help="check how trustworthy the timestamps are")
    a = ap.parse_args(argv)
    if a.scan:
        _scan_report(a.root)
    elif a.inspect:
        _inspect(a.inspect)
    elif a.mapping:
        print(mapping_table_markdown())
    elif a.timing:
        _timing_report(a.root)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
