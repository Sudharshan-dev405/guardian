"""
data/loader.py -- Guardian dataset loader.

Reads raw dataset files, extracts the wrist accelerometer and gyroscope
channels, resamples to 50 Hz, windows at 2.5 s with 50% overlap, and yields
window dicts in the format fixed by contract.py.

Two consumption modes:
    iter_windows(record)          -> (window, meta) pairs, as fast as possible.
                                     Use this for training and evaluation.
    replay(record, speed=1.0)     -> window dicts only, paced in wall-clock
                                     time. Use this for the dashboard demo.

The window dict is exactly the contract, nothing extra:
    {"t": float, "acc": (125,3) g, "gyro": (125,3) deg/s | None,
     "hr": float|None, "temp": float|None, "fs": 50}

Labels live in `meta`, never in `window`, so a stream module cannot
accidentally see ground truth.

CLI (run from the repo root, C:\\Users\\mvpva\\guardian):
    py -m data.loader --inspect data\\raw\\UMAFall\\<some file>.csv
    py -m data.loader --scan    data\\raw\\UMAFall
    py -m data.loader --mapping
    py -m data.loader --selftest
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Constants -- the contract, in one place
# --------------------------------------------------------------------------

TARGET_FS = 50               # Hz
WINDOW_SEC = 2.5
WINDOW_N = int(round(WINDOW_SEC * TARGET_FS))   # 125 samples
HOP_N = WINDOW_N // 2                            # 62 samples -> 1.24 s
G_MS2 = 9.80665
DEG_PER_RAD = 180.0 / np.pi

# UMAFall column semantics (from the '%' header block in each file).
UMA_SENSOR_TYPE = {0: "acc", 1: "gyro", 2: "mag"}
UMA_WRIST_ID = 2             # 0 pocket, 1 chest, 2 wrist, 3 waist, 4 ankle


# --------------------------------------------------------------------------
# Context label mapping -- this table goes in the report
# --------------------------------------------------------------------------

STATES = ("stationary", "ambulating", "seated hand activity", "lying/immobile")

# (substring matched case-insensitively against the dataset activity name,
#  target state, rationale for the report)
ACTIVITY_MAP = [
    ("walking",     "ambulating",           "level gait, 1.4-2.3 Hz cadence"),
    ("jogging",     "ambulating",           "gait, higher cadence; merged per HAR70+ practice"),
    ("godownstairs","ambulating",           "stairs share the walking cadence band; merging makes the known confusion intra-class"),
    ("goupstairs",  "ambulating",           "as above"),
    ("hopping",     "ambulating",           "whole-body periodic locomotion; not a fall, must not read as impact"),

    ("sitting",     "stationary",           "trunk static; sit/stand differ only by a tilt a free wrist destroys, so both are 'stationary'"),
    ("standing",    "stationary",           "as above"),
    ("bending",     "stationary",           "postural transition with a static base of support"),

    ("aplausing",   "seated hand activity", "hand motion, body static -- the class exists to suppress hand impacts"),
    ("clapping",    "seated hand activity", "as above"),
    ("makingacall", "seated hand activity", "hand raised to head, body static"),
    ("handsup",     "seated hand activity", "large forearm excursion, body static"),
    ("openingdoor", "seated hand activity", "JUDGEMENT CALL: subject is standing, not seated. Class is operationally "
                                            "'hand activity with a static body'; the name is kept for team consistency"),

    ("lying",       "lying/immobile",       "trunk horizontal and still"),
]

FALL_KEYS = ("fall", "syncope", "trip", "slip")


def map_activity(activity: str):
    """Return (state, is_fall). state is None for falls and unmapped names."""
    a = (activity or "").lower().replace("_", "").replace(" ", "")
    for key in FALL_KEYS:
        if key in a:
            return None, True
    for key, state, _ in ACTIVITY_MAP:
        if key in a:
            return state, False
    return None, False


def mapping_table_markdown() -> str:
    rows = ["| Dataset activity (match) | Guardian state | Rationale |",
            "|---|---|---|"]
    for key, state, why in ACTIVITY_MAP:
        rows.append(f"| `*{key}*` | {state} | {why} |")
    rows.append("| `*fall*`, `*trip*`, `*slip*` | (excluded from context training) "
                "| falls are transitions, not activity states; they train the motion stream instead |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------

@dataclass
class Record:
    """One raw trial, resampled to TARGET_FS."""
    path: Path
    subject: str
    activity: str
    trial: str
    t: np.ndarray                    # (M,) seconds from 0
    acc: np.ndarray                  # (M, 3) g
    gyro: np.ndarray | None          # (M, 3) deg/s, or None if absent
    fs_raw: float
    state: str | None = None
    is_fall: bool = False
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def summary(self) -> str:
        g = "none" if self.gyro is None else f"{self.gyro.shape[0]}x3"
        return (f"{self.path.name}\n"
                f"  subject={self.subject}  activity={self.activity}  trial={self.trial}\n"
                f"  state={self.state}  is_fall={self.is_fall}\n"
                f"  raw fs~{self.fs_raw:.1f} Hz -> {TARGET_FS} Hz, {self.duration:.1f} s\n"
                f"  acc={self.acc.shape[0]}x3 g   gyro={g}\n"
                + "".join(f"  ! {n}\n" for n in self.notes))


# --------------------------------------------------------------------------
# Unit auto-detection -- so a wrong assumption is loud, not silent
# --------------------------------------------------------------------------

def _acc_to_g(a: np.ndarray, notes: list) -> np.ndarray:
    """Accept m/s^2 or g. Decide from the median resultant, which sits near
    1 g or near 9.81 m/s^2 for any real recording."""
    med = float(np.median(np.linalg.norm(a, axis=1)))
    if 5.0 < med < 15.0:
        notes.append(f"accelerometer read as m/s^2 (median |a|={med:.2f}), converted to g")
        return a / G_MS2
    if 0.5 < med < 1.6:
        return a
    notes.append(f"UNKNOWN ACCELEROMETER UNITS: median |a|={med:.3f}, left unscaled -- check this")
    return a


def _gyro_to_dps(w: np.ndarray, notes: list) -> np.ndarray:
    """Accept deg/s or rad/s. A staged fall reaches several hundred deg/s;
    the same motion in rad/s is single digits."""
    pk = float(np.percentile(np.abs(w), 99.5))
    if pk < 20.0 and pk > 0.0:
        notes.append(f"gyroscope read as rad/s (99.5th pct |w|={pk:.2f}), converted to deg/s")
        return w * DEG_PER_RAD
    return w


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def _resample(t: np.ndarray, x: np.ndarray, fs: int = TARGET_FS):
    """Linear interpolation onto a uniform grid. Upsampling recovers no
    information -- it only makes windows the right shape."""
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    n = int(np.floor(t[-1] * fs)) + 1
    grid = np.arange(n) / fs
    out = np.empty((n, x.shape[1]), dtype=np.float32)
    for k in range(x.shape[1]):
        out[:, k] = np.interp(grid, t, x[:, k])
    return grid, out


def _est_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else float("nan")


# --------------------------------------------------------------------------
# Reader: UMAFall
# --------------------------------------------------------------------------

def read_umafall(path: Path, sensor_id: int = UMA_WRIST_ID) -> Record:
    """
    UMAFall CSV: a block of '%' comment lines, then rows of
        TimeStamp; SampleNo; X; Y; Z; SensorType; SensorID
    separated by ';'. One file interleaves every sensor at every body
    position, so we filter to SensorID == wrist and split by SensorType.
    """
    path = Path(path)
    notes: list = []
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%") or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(";")]
            if len(parts) < 7:
                parts = [p.strip() for p in s.split(",")]
            if len(parts) < 7:
                continue
            try:
                rows.append([float(parts[0]), float(parts[2]), float(parts[3]),
                             float(parts[4]), int(float(parts[5])), int(float(parts[6]))])
            except ValueError:
                continue  # header text line

    if not rows:
        raise ValueError(f"{path.name}: no parseable data rows. "
                         f"Run --inspect on this file and send me the first 20 lines.")

    arr = np.asarray(rows, dtype=float)
    at_wrist = arr[arr[:, 5] == sensor_id]
    if len(at_wrist) == 0:
        present = sorted(set(arr[:, 5].astype(int).tolist()))
        raise ValueError(f"{path.name}: no rows with SensorID={sensor_id} (wrist). "
                         f"Present IDs: {present}")

    def channel(stype: int):
        sub = at_wrist[at_wrist[:, 4] == stype]
        if len(sub) < 4:
            return None, None
        order = np.argsort(sub[:, 0])
        sub = sub[order]
        t = sub[:, 0]
        if t[-1] - t[0] > 1000:       # timestamps are milliseconds
            t = t / 1000.0
        return t - t[0], sub[:, 1:4]

    t_a, acc = channel(0)
    if acc is None:
        raise ValueError(f"{path.name}: wrist accelerometer channel empty.")
    t_w, gyro = channel(1)

    fs_raw = _est_fs(t_a)
    acc = _acc_to_g(acc, notes)
    grid, acc_r = _resample(t_a, acc)

    if gyro is None:
        gyro_r = None
        notes.append("no wrist gyroscope in this file -- Stage 1 gate falls back to the accelerometer only")
    else:
        gyro = _gyro_to_dps(gyro, notes)
        gyro_r = np.empty((len(grid), 3), dtype=np.float32)
        for k in range(3):
            gyro_r[:, k] = np.interp(grid, t_w, gyro[:, k])

    if fs_raw < 40:
        notes.append(f"raw wrist rate is {fs_raw:.0f} Hz; upsampled to {TARGET_FS} Hz. "
                     f"Impact peaks are attenuated at this rate -- retune the motion gate on this data")

    subject, activity, trial = _parse_umafall_name(path.name)
    state, is_fall = map_activity(activity)
    return Record(path=path, subject=subject, activity=activity, trial=trial,
                  t=grid, acc=acc_r, gyro=gyro_r, fs_raw=fs_raw,
                  state=state, is_fall=is_fall, notes=notes)


def _parse_umafall_name(name: str):
    """UMAFall_Subject_01_ADL_Walking_1_2016-06-13_20-23-52.csv"""
    stem = Path(name).stem
    parts = stem.split("_")
    subject, activity, trial = "unknown", stem, "1"
    for i, p in enumerate(parts):
        if p.lower() == "subject" and i + 1 < len(parts):
            subject = f"S{parts[i+1]}"
        if p.upper() in ("ADL", "FALL") and i + 1 < len(parts):
            activity = parts[i + 1]
            if i + 2 < len(parts) and parts[i + 2].isdigit():
                trial = parts[i + 2]
    return subject, activity, trial


# --------------------------------------------------------------------------
# Reader: generic CSV -- for FallAllD or anything else, once we know the shape
# --------------------------------------------------------------------------

def read_generic_csv(path: Path, *, acc_cols, gyro_cols=None, time_col=None,
                     fs=None, subject="unknown", activity="unknown",
                     trial="1", delimiter=",") -> Record:
    """Column names or indices. If time_col is None, fs must be given."""
    import csv as _csv
    path = Path(path)
    notes: list = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rdr = _csv.reader(fh, delimiter=delimiter)
        rows = [r for r in rdr if r]
    header = rows[0]
    numeric_first = all(_is_num(c) for c in header)
    body = rows if numeric_first else rows[1:]

    def idx(c):
        if isinstance(c, int):
            return c
        return header.index(c)

    ai = [idx(c) for c in acc_cols]
    gi = [idx(c) for c in gyro_cols] if gyro_cols else None
    ti = idx(time_col) if time_col is not None else None

    data = np.asarray([[float(r[k]) for k in range(len(r))] for r in body
                       if _row_ok(r)], dtype=float)
    acc = data[:, ai]
    t = data[:, ti] if ti is not None else np.arange(len(data)) / float(fs)
    if t[-1] - t[0] > 100000:
        t = t / 1000.0
    t = t - t[0]

    fs_raw = _est_fs(t)
    acc = _acc_to_g(acc, notes)
    grid, acc_r = _resample(t, acc)
    if gi:
        gyro = _gyro_to_dps(data[:, gi], notes)
        gyro_r = np.empty((len(grid), 3), dtype=np.float32)
        for k in range(3):
            gyro_r[:, k] = np.interp(grid, t, gyro[:, k])
    else:
        gyro_r = None
        notes.append("no gyroscope columns given")

    state, is_fall = map_activity(activity)
    return Record(path=path, subject=subject, activity=activity, trial=trial,
                  t=grid, acc=acc_r, gyro=gyro_r, fs_raw=fs_raw,
                  state=state, is_fall=is_fall, notes=notes)


def _is_num(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _row_ok(r):
    return all(_is_num(c) for c in r)


# --------------------------------------------------------------------------
# Scripted HR / temperature trace (Mithuna's CSV, optional)
# --------------------------------------------------------------------------

class ScriptedTrace:
    """CSV with columns t,hr[,temp]. Returns the most recent sample at or
    before the query time, or None if the trace has not started."""

    def __init__(self, path):
        import csv as _csv
        self.t, self.hr, self.temp = [], [], []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh):
                self.t.append(float(row["t"]))
                self.hr.append(float(row["hr"]) if row.get("hr") not in (None, "", "nan") else None)
                v = row.get("temp")
                self.temp.append(float(v) if v not in (None, "", "nan") else None)
        self.t = np.asarray(self.t, dtype=float)

    def at(self, t: float):
        if len(self.t) == 0 or t < self.t[0]:
            return None, None
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        return self.hr[i], self.temp[i]


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def iter_windows(record: Record, trace: "ScriptedTrace | None" = None,
                 t0: float = 0.0):
    """
    Yield (window, meta) as fast as the CPU allows.

    window follows contract.py exactly. meta carries ground truth and
    provenance and must never be passed to a stream's score().
    """
    n = record.acc.shape[0]
    if n < WINDOW_N:
        return
    for start in range(0, n - WINDOW_N + 1, HOP_N):
        stop = start + WINDOW_N
        t = t0 + start / TARGET_FS
        hr, temp = trace.at(t) if trace is not None else (None, None)
        window = {
            "t": float(t),
            "acc": record.acc[start:stop],
            "gyro": None if record.gyro is None else record.gyro[start:stop],
            "hr": hr,
            "temp": temp,
            "fs": TARGET_FS,
        }
        meta = {
            "subject": record.subject,
            "activity": record.activity,
            "state": record.state,
            "is_fall": record.is_fall,
            "trial": record.trial,
            "source": record.path.name,
            "start_sample": start,
        }
        yield window, meta


def replay(record: Record, speed: float = 1.0,
           trace: "ScriptedTrace | None" = None, t0: float = 0.0):
    """
    Yield windows paced in wall-clock time, for the dashboard.
    speed=1.0 is real time; speed=4.0 is four times faster.
    Yields the window dict only -- no labels reach the pipeline in a demo.
    """
    period = (HOP_N / TARGET_FS) / max(speed, 1e-6)
    next_due = time.perf_counter()
    for window, _meta in iter_windows(record, trace=trace, t0=t0):
        now = time.perf_counter()
        if next_due > now:
            time.sleep(next_due - now)
        next_due += period
        yield window


# --------------------------------------------------------------------------
# Directory scanning
# --------------------------------------------------------------------------

def scan(root, pattern="*.csv", reader=read_umafall, limit=None):
    """Yield Records for every readable file under root. Files that fail are
    reported and skipped rather than killing the run."""
    root = Path(root)
    files = sorted(root.rglob(pattern))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            yield reader(f)
        except Exception as e:                      # noqa: BLE001
            print(f"[skip] {f.name}: {e}", file=sys.stderr)


def load_dataset(root, pattern="*.csv", reader=read_umafall, falls=None):
    """
    Collect windows for training. Returns (X_meta, windows) as two lists,
    same length and order.
      falls=None  -> everything
      falls=False -> ADLs only (context classifier training)
      falls=True  -> falls only
    """
    windows, metas = [], []
    for rec in scan(root, pattern, reader):
        if falls is True and not rec.is_fall:
            continue
        if falls is False and rec.is_fall:
            continue
        for w, m in iter_windows(rec):
            windows.append(w)
            metas.append(m)
    return windows, metas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _inspect(path):
    p = Path(path)
    print(f"--- first 20 raw lines of {p.name} ---")
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 20:
                break
            print(f"{i:3d} | {line.rstrip()}")
    print("--- parsed as UMAFall ---")
    try:
        rec = read_umafall(p)
        print(rec.summary())
        n = sum(1 for _ in iter_windows(rec))
        print(f"  -> {n} windows of {WINDOW_N} samples")
    except Exception as e:                          # noqa: BLE001
        print(f"  FAILED: {e}")


def _scan_report(root):
    counts, subjects, bad = {}, set(), 0
    total_windows = 0
    for rec in scan(root):
        key = f"{rec.state or ('FALL' if rec.is_fall else 'UNMAPPED:' + rec.activity)}"
        counts[key] = counts.get(key, 0) + 1
        subjects.add(rec.subject)
        total_windows += sum(1 for _ in iter_windows(rec))
        if rec.state is None and not rec.is_fall:
            bad += 1
    print(f"subjects: {len(subjects)}  {sorted(subjects)}")
    for k in sorted(counts):
        print(f"  {k:24s} {counts[k]} files")
    print(f"total windows: {total_windows}")
    if bad:
        print(f"\n{bad} files did not match the mapping table -- add them to ACTIVITY_MAP.")


def _selftest():
    """Build a synthetic file in UMAFall's on-disk shape and push it through.
    This tests the machinery only. It is never used as project data."""
    import tempfile
    fs, secs = 20, 12
    n = fs * secs
    t = (np.arange(n) / fs * 1000).astype(int)
    rng = np.random.default_rng(0)
    walk = np.column_stack([
        0.4 * np.sin(2 * np.pi * 1.9 * np.arange(n) / fs),
        0.3 * np.cos(2 * np.pi * 1.9 * np.arange(n) / fs),
        9.81 + 0.5 * np.sin(2 * np.pi * 3.8 * np.arange(n) / fs),
    ]) + rng.normal(0, 0.05, (n, 3))
    gyr = rng.normal(0, 30, (n, 3))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "UMAFall_Subject_01_ADL_Walking_1_2016-06-13_20-23-52.csv"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("% UMAFall synthetic fixture\n")
            fh.write("% TimeStamp; Sample No; X-Axis; Y-Axis; Z-Axis; Sensor Type; Sensor ID\n")
            for i in range(n):
                fh.write(f"{t[i]}; {i}; {walk[i,0]:.4f}; {walk[i,1]:.4f}; {walk[i,2]:.4f}; 0; 2\n")
                fh.write(f"{t[i]}; {i}; {gyr[i,0]:.4f}; {gyr[i,1]:.4f}; {gyr[i,2]:.4f}; 1; 2\n")
                fh.write(f"{t[i]}; {i}; 0.1; 0.1; 0.1; 0; 3\n")   # waist, must be ignored
        rec = read_umafall(p)
        print(rec.summary())
        ws = list(iter_windows(rec))
        assert ws, "no windows produced"
        w, m = ws[0]
        assert w["acc"].shape == (WINDOW_N, 3), w["acc"].shape
        assert w["gyro"].shape == (WINDOW_N, 3)
        assert w["fs"] == TARGET_FS and w["hr"] is None
        assert set(w) == {"t", "acc", "gyro", "hr", "temp", "fs"}, "window has extra keys"
        assert 0.8 < np.median(np.linalg.norm(w["acc"], axis=1)) < 1.3, "unit conversion wrong"
        assert m["state"] == "ambulating" and m["is_fall"] is False
        assert abs(ws[1][0]["t"] - ws[0][0]["t"] - HOP_N / TARGET_FS) < 1e-9
        print(f"windows: {len(ws)}, hop {HOP_N/TARGET_FS:.2f} s, "
              f"last t={ws[-1][0]['t']:.2f} s")
    print("SELFTEST PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guardian dataset loader")
    ap.add_argument("--inspect", metavar="FILE")
    ap.add_argument("--scan", metavar="DIR")
    ap.add_argument("--mapping", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.inspect:
        _inspect(a.inspect)
    elif a.scan:
        _scan_report(a.scan)
    elif a.mapping:
        print(mapping_table_markdown())
    elif a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
