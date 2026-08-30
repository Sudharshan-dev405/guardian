"""
data/loader_fallalld.py -- FallAllD loader, same output contract as
data/loader.py.

Everything below was established empirically or from the source paper
(Saleh, Abbas and Le Bouquin Jeannes, IEEE Sensors Journal 21(2), 2021),
not assumed:

- WRIST IS DEVICE 2. The paper states twelve hand-activity ADLs concern only
  the wrist-worn device. A009 (clapping hands), A010 (waving hands) and A011
  (hand shaking) have 49, 25 and 50 records on D2 and ZERO on D1 and D3.
  Structural, not statistical.
- ADL = A001..A044, FALL = A101..A135. Confirmed by exact file counts:
  4883 ADL and 1722 fall accelerometer files, matching the published totals.
- SCALING. LSM9DS1 at +/-8 g on a 16-bit signed axis is 4096 LSB per g;
  at +/-2000 dps it is 0.07 dps per LSB. Verified: median resultant lands at
  0.980 g and no samples sit at the rail.
- One file per sensor: S<subj>_D<dev>_A<act>_T<trial>_{A,G,M,B}.dat, three
  integer columns, no header. We read A and G, ignore M and B.
- Records are 20 s with the fall impact / ADL transition centred at 10 s.

DOWNSAMPLING, which is the step that can silently invent a result:
238 Hz -> 50 Hz is a ratio of 4.76, not an integer, so plain decimation is
not available and would alias regardless. We use scipy.signal.resample_poly
with up=25, down=119 (25/119 = 0.21008, and 238 * 25/119 = 50.0 exactly).
resample_poly applies a Kaiser-windowed FIR anti-alias filter before
downsampling, so content above the new 25 Hz Nyquist is removed rather than
folded back into the band. This matters: a fall impact has energy well above
25 Hz, and aliasing it would fabricate low-frequency structure that the
motion classifier would happily score.

Contrast with data/loader.py, where UMAFall's ~20 Hz wrist data is UPSAMPLED
to 50 Hz by linear interpolation. That recovers nothing; this removes
something. Different signal paths, which is exactly why the cross-dataset
comparison is worth running.

CLI:
    py -m data.loader_fallalld --inspect data\\raw\\FallAllD\\FallAllD\\S02_D2_A026_T01_A.dat
    py -m data.loader_fallalld --scan
    py -m data.loader_fallalld --mapping
    py -m data.loader_fallalld --selftest
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TARGET_FS = 50
WINDOW_SEC = 2.5
WINDOW_N = int(round(WINDOW_SEC * TARGET_FS))       # 125
HOP_N = WINDOW_N // 2                                # 62 -> 1.24 s

RAW_FS = 238.0
RESAMPLE_UP, RESAMPLE_DOWN = 25, 119                 # 238 * 25/119 = 50.0
TRIM_N = 50                                          # 1 s trimmed each end

# Matched-path mode. 238 -> 20 Hz (238 * 10/119 = 20.0 exactly), then 20 -> 50
# by linear interpolation -- the identical signal path data/loader.py puts
# UMAFall through. Used to test whether the cross-dataset context drop is
# caused by the sampling path or by subject and protocol differences.
UMAFALL_FS = 20.0
MATCH_UP, MATCH_DOWN = 10, 119                       # 238 * 10/119 = 20.0

WRIST_DEVICE = 2
ACC_LSB_PER_G = 4096.0                               # +/-8 g, 16-bit signed
GYR_DPS_PER_LSB = 0.07                               # +/-2000 dps
CLIP_G = 8.0                                         # sensor rail

ADL_RANGE = (1, 44)
FALL_RANGE = (101, 135)

NAME_RE = re.compile(
    r"S(\d+)_D(\d+)_A(\d+)_T(\d+)_([AGMB])\.dat$", re.I)

DEFAULT_ROOT = Path("data/raw/FallAllD/FallAllD")

# --------------------------------------------------------------------------
# Activity table, transcribed from Figure 4 and Figure 8 of the paper
# --------------------------------------------------------------------------

ACTIVITY_NAMES = {
    1: "walking slowly", 2: "walking quickly",
    3: "jogging slowly", 4: "jogging quickly",
    5: "climbing stairs up slowly", 6: "climbing stairs up quickly",
    7: "climbing stairs down slowly", 8: "climbing stairs down quickly",
    9: "clapping hands", 10: "waving hands", 11: "hand shaking",
    12: "sitting down (low/high chair)", 13: "standing up (low/high chair)",
    14: "fail to stand up after half standing",
    15: "lying down on a bed", 16: "rising up from a bed",
    17: "changing position while lying",
    18: "stumbling while walking without falling",
    19: "jumping slightly", 20: "jumping strongly",
    21: "bending down",
    22: "clap hands one time", 23: "raising hand up", 24: "moving hand down",
    25: "moving hand up then down immediately",
    26: "beating a table with your hand",
    27: "start walking", 28: "stop walking",
    29: "start jogging", 30: "stop jogging",
    31: "start clapping hands", 32: "stop clapping hands",
    33: "start waving hands", 34: "stop waving hands",
    35: "start climbing stairs up", 36: "stop climbing stairs up",
    37: "start climbing stairs down", 38: "stop climbing stairs down",
    39: "lift ascending start", 40: "lift ascending stop",
    41: "lift descending start", 42: "lift descending stop",
    43: "standing in a moving bus/metro", 44: "sitting in a moving bus/metro",
}

# --------------------------------------------------------------------------
# Context mapping -- FallAllD's own version of the table. Goes in the report.
# --------------------------------------------------------------------------

STATES = ("stationary", "ambulating", "seated hand activity", "lying/immobile")

CONTEXT_MAP = {
    1:  ("ambulating",           "level gait, slow"),
    2:  ("ambulating",           "level gait, fast"),
    3:  ("ambulating",           "jogging; merged into ambulating per HAR70+ practice"),
    4:  ("ambulating",           "as above"),
    5:  ("ambulating",           "stairs share the walking cadence band; merged so the known confusion is intra-class"),
    6:  ("ambulating",           "as above"),
    7:  ("ambulating",           "as above"),
    8:  ("ambulating",           "as above"),
    9:  ("seated hand activity", "hand motion, body static"),
    10: ("seated hand activity", "hand motion, body static"),
    11: ("seated hand activity", "hand motion, body static"),
    15: ("lying/immobile",       "on a bed, trunk horizontal"),
    17: ("lying/immobile",       "still lying, only repositioning"),
    19: ("ambulating",           "whole-body periodic motion; must not read as impact"),
    20: ("ambulating",           "as above"),
    22: ("seated hand activity", "single hand impact, body static"),
    23: ("seated hand activity", "forearm excursion, body static"),
    24: ("seated hand activity", "forearm excursion, body static"),
    25: ("seated hand activity", "forearm excursion, body static"),
    26: ("seated hand activity", "HARD hand impact with a static body -- the "
                                 "false-alarm-suppression case UMAFall lacks entirely"),
    43: ("stationary",           "standing, vehicle vibration only"),
    44: ("stationary",           "seated, vehicle vibration only"),
}

# Deliberately UNMAPPED, with reasons. These are transitions, not states, and
# forcing them into a four-state scheme would manufacture label noise and then
# report it as a generalisation failure.
UNMAPPED_REASONS = {
    12: "sitting down -- a transition between two states, not a state",
    13: "standing up -- transition",
    14: "fail to stand up -- transition, and the paper notes it is arguably a fall",
    16: "rising from a bed -- transition",
    18: "stumbling -- transition, and a known false-alarm source",
    21: "bending down -- transition",
    27: "start walking -- transient phase of a cyclic ADL",
    28: "stop walking -- transient phase",
    29: "start jogging -- transient phase",
    30: "stop jogging -- transient phase",
    31: "start clapping -- transient phase",
    32: "stop clapping -- transient phase",
    33: "start waving -- transient phase",
    34: "stop waving -- transient phase",
    35: "start stairs up -- transient phase",
    36: "stop stairs up -- transient phase",
    37: "start stairs down -- transient phase",
    38: "stop stairs down -- transient phase",
    39: "lift ascending start -- transient, and vertical acceleration is lift motion",
    40: "lift ascending stop -- transient",
    41: "lift descending start -- transient",
    42: "lift descending stop -- transient",
}

# Records that make good scenario-1 candidates: hard hand impact, static body.
SCENARIO1_ACTIVITIES = (26, 22, 9, 25)


def map_activity(activity_id: int):
    """Return (state, is_fall). state is None for falls and unmapped IDs."""
    if FALL_RANGE[0] <= activity_id <= FALL_RANGE[1]:
        return None, True
    entry = CONTEXT_MAP.get(activity_id)
    return (entry[0] if entry else None), False


def mapping_table_markdown() -> str:
    rows = ["| FallAllD activity | ID | Guardian state | Rationale |",
            "|---|---|---|---|"]
    for aid in sorted(CONTEXT_MAP):
        state, why = CONTEXT_MAP[aid]
        rows.append(f"| {ACTIVITY_NAMES.get(aid, '?')} | A{aid:03d} | {state} | {why} |")
    rows.append(f"| all fall types | A101-A135 | (excluded from context training) "
                f"| falls are transitions, not activity states |")
    rows.append("")
    rows.append("Deliberately unmapped, with reasons:")
    rows.append("")
    rows.append("| FallAllD activity | ID | Why not mapped |")
    rows.append("|---|---|---|")
    for aid in sorted(UNMAPPED_REASONS):
        rows.append(f"| {ACTIVITY_NAMES.get(aid, '?')} | A{aid:03d} | {UNMAPPED_REASONS[aid]} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------

@dataclass
class Record:
    path: Path
    subject: str
    activity: str
    activity_id: int
    trial: str
    t: np.ndarray
    acc: np.ndarray                  # (M, 3) g at TARGET_FS
    gyro: np.ndarray | None          # (M, 3) deg/s at TARGET_FS
    fs_raw: float
    state: str | None = None
    is_fall: bool = False
    clipped_frac: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def summary(self):
        g = "none" if self.gyro is None else f"{self.gyro.shape[0]}x3"
        return (f"{self.path.name}\n"
                f"  subject={self.subject}  A{self.activity_id:03d} "
                f"= {self.activity}  trial={self.trial}\n"
                f"  state={self.state}  is_fall={self.is_fall}\n"
                f"  {self.fs_raw:.0f} Hz -> {TARGET_FS} Hz, {self.duration:.1f} s\n"
                f"  acc={self.acc.shape[0]}x3 g   gyro={g}   "
                f"clipped={self.clipped_frac*100:.2f}%\n"
                + "".join(f"  ! {n}\n" for n in self.notes))


# --------------------------------------------------------------------------
# Downsampling -- the step to get right
# --------------------------------------------------------------------------

def downsample(x: np.ndarray, up=RESAMPLE_UP, down=RESAMPLE_DOWN) -> np.ndarray:
    """238 Hz -> 50 Hz via polyphase resampling.

    resample_poly upsamples by `up`, applies a Kaiser-windowed FIR low-pass at
    the lower of the two Nyquists, then decimates by `down`. The anti-alias
    filter is the point: without it, impact energy above 25 Hz folds back into
    the band and fabricates low-frequency structure.
    """
    return signal.resample_poly(x, up, down, axis=0).astype(np.float32)


def downsample_trimmed(x, trim=TRIM_N):
    """downsample(), then drop `trim` samples from each end.

    The polyphase FIR rings at the record boundaries: a 40 Hz tone that is
    attenuated to 0.0003 in the interior still shows 0.215 in the first and
    last few samples. Records are 20 s with the event centred at 10 s, so
    trimming 1 s from each end removes the artefact and costs nothing.
    """
    y = downsample(x)
    return y[trim:-trim] if len(y) > 2 * trim + WINDOW_N else y


def downsample_umafall_path(x, trim=TRIM_N):
    """238 Hz -> 20 Hz -> 50 Hz, reproducing UMAFall's signal path exactly.

    Why this exists: the context classifier was trained on UMAFall wrist data,
    which is ~20 Hz linearly interpolated up to 50 Hz. That signal has no real
    content above 10 Hz. FallAllD at a true 50 Hz carries genuine energy to
    25 Hz, so the spectral features (dominant frequency, spectral entropy,
    band power, jerk RMS) see a different world and the classifier
    misclassifies confidently.

    Running FallAllD through UMAFall's path isolates the cause. If context
    accuracy recovers, the sampling path was responsible. If it does not, the
    gap is subject and protocol differences, which is a different claim.

    Note this DEGRADES the data on purpose: the 20 Hz stage discards real
    information that the direct path keeps. It is a diagnostic, not the
    preferred loader.
    """
    lo = signal.resample_poly(x, MATCH_UP, MATCH_DOWN, axis=0)
    lo = lo[10:-10] if len(lo) > 20 + 10 else lo        # trim FIR edge ringing
    t_lo = np.arange(len(lo)) / UMAFALL_FS
    n = int(np.floor(t_lo[-1] * TARGET_FS)) + 1
    grid = np.arange(n) / TARGET_FS
    out = np.empty((n, x.shape[1]), dtype=np.float32)
    for k in range(x.shape[1]):
        out[:, k] = np.interp(grid, t_lo, lo[:, k])     # linear, as loader.py
    return out


def parse_name(name):
    m = NAME_RE.search(name)
    if not m:
        return None
    s, d, a, t, sensor = m.groups()
    return dict(subject=f"S{s}", device=int(d), activity_id=int(a),
                trial=t, sensor=sensor.upper())


def read_record(acc_path, device=WRIST_DEVICE, match_umafall_path=False) -> Record:
    acc_path = Path(acc_path)
    meta = parse_name(acc_path.name)
    if meta is None:
        raise ValueError(f"{acc_path.name}: filename does not match "
                         f"S<n>_D<n>_A<n>_T<n>_<S>.dat")
    if meta["sensor"] != "A":
        raise ValueError(f"{acc_path.name}: pass the accelerometer (_A) file")
    if meta["device"] != device:
        raise ValueError(f"{acc_path.name}: device D{meta['device']}, "
                         f"wanted D{device} (wrist)")

    notes = []
    raw_acc = np.loadtxt(acc_path, delimiter=",", dtype=np.float64)
    if raw_acc.ndim != 2 or raw_acc.shape[1] != 3:
        raise ValueError(f"{acc_path.name}: expected 3 columns, "
                         f"got shape {raw_acc.shape}")

    rail = 8.0 * ACC_LSB_PER_G
    clipped = float(np.mean(np.abs(raw_acc) >= rail - 2))
    if clipped > 0.001:
        notes.append(f"{clipped*100:.2f}% of accelerometer samples at the "
                     f"+/-8 g rail")

    resample = downsample_umafall_path if match_umafall_path else downsample_trimmed
    acc = resample(raw_acc / ACC_LSB_PER_G)
    if match_umafall_path:
        notes.append("MATCHED PATH: 238 -> 20 -> 50 Hz, reproducing UMAFall's "
                     "signal path (diagnostic; discards real content)")

    gyro_path = acc_path.with_name(acc_path.name[:-6] + "_G.dat")
    if gyro_path.exists():
        raw_gyr = np.loadtxt(gyro_path, delimiter=",", dtype=np.float64)
        if raw_gyr.ndim == 2 and raw_gyr.shape[1] == 3:
            gyr = resample(raw_gyr * GYR_DPS_PER_LSB)
            n = min(len(acc), len(gyr))
            acc, gyro = acc[:n], gyr[:n]
        else:
            gyro = None
            notes.append("gyroscope file malformed, ignored")
    else:
        gyro = None
        notes.append("no gyroscope file for this record")

    t = np.arange(len(acc)) / TARGET_FS
    aid = meta["activity_id"]
    state, is_fall = map_activity(aid)
    name = ACTIVITY_NAMES.get(aid, f"fall type {aid}" if is_fall else "unknown")
    if state is None and not is_fall:
        notes.append(f"A{aid:03d} deliberately unmapped: "
                     f"{UNMAPPED_REASONS.get(aid, 'not in the mapping table')}")

    return Record(path=acc_path, subject=meta["subject"], activity=name,
                  activity_id=aid, trial=meta["trial"], t=t, acc=acc,
                  gyro=gyro, fs_raw=RAW_FS, state=state, is_fall=is_fall,
                  clipped_frac=clipped, notes=notes)


# --------------------------------------------------------------------------
# Windowing -- identical contract to data/loader.py
# --------------------------------------------------------------------------

def iter_windows(record: Record, t0: float = 0.0):
    """Yield (window, meta). window has exactly the six contract keys."""
    n = record.acc.shape[0]
    if n < WINDOW_N:
        return
    for start in range(0, n - WINDOW_N + 1, HOP_N):
        stop = start + WINDOW_N
        t = t0 + start / TARGET_FS
        window = {
            "t": float(t),
            "acc": record.acc[start:stop],
            "gyro": None if record.gyro is None else record.gyro[start:stop],
            "hr": None,
            "temp": None,
            "fs": TARGET_FS,
        }
        meta = {
            "subject": record.subject,
            "activity": record.activity,
            "activity_id": record.activity_id,
            "state": record.state,
            "is_fall": record.is_fall,
            "trial": record.trial,
            "source": record.path.name,
            "start_sample": start,
            "dataset": "FallAllD",
        }
        yield window, meta


def scan(root=DEFAULT_ROOT, device=WRIST_DEVICE, activities=None, limit=None,
         match_umafall_path=False):
    """Yield Records for wrist accelerometer files. activities is an optional
    iterable of activity IDs to keep."""
    root = Path(root)
    files = sorted(root.glob(f"S*_D{device}_A*_T*_A.dat"))
    if activities is not None:
        keep = set(activities)
        files = [f for f in files
                 if (m := parse_name(f.name)) and m["activity_id"] in keep]
    if limit:
        files = files[:limit]
    for f in files:
        try:
            yield read_record(f, device, match_umafall_path)
        except Exception as e:                              # noqa: BLE001
            print(f"[skip] {f.name}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _inspect(path):
    p = Path(path)
    print(f"--- first 8 raw lines of {p.name} ---")
    with open(p, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 8:
                break
            print(f"{i:3d} | {line.rstrip()}")
    print("--- parsed ---")
    rec = read_record(p)
    print(rec.summary())
    n = sum(1 for _ in iter_windows(rec))
    svm = np.linalg.norm(rec.acc, axis=1)
    print(f"  -> {n} windows of {WINDOW_N} samples")
    print(f"  peak |acc| = {svm.max():.2f} g   median = {np.median(svm):.3f} g")
    if rec.gyro is not None:
        print(f"  peak |gyro| = {np.linalg.norm(rec.gyro, axis=1).max():.0f} deg/s")


def _scan_report(root, device):
    counts, subjects, unmapped = {}, set(), {}
    total_windows = 0
    n = 0
    for rec in scan(root, device):
        key = rec.state or ("FALL" if rec.is_fall
                            else f"UNMAPPED A{rec.activity_id:03d}")
        counts[key] = counts.get(key, 0) + 1
        if rec.state is None and not rec.is_fall:
            unmapped[rec.activity_id] = unmapped.get(rec.activity_id, 0) + 1
        subjects.add(rec.subject)
        total_windows += sum(1 for _ in iter_windows(rec))
        n += 1
    print(f"device D{device} (wrist)   {n} records   "
          f"{len(subjects)} subjects: {sorted(subjects)}")
    for k in sorted(counts):
        if k.startswith("UNMAPPED"):
            continue
        print(f"  {k:24s} {counts[k]:>5} records")
    if unmapped:
        tot = sum(unmapped.values())
        print(f"  {'(unmapped, by design)':24s} {tot:>5} records across "
              f"{len(unmapped)} activity types")
    print(f"total windows: {total_windows}")
    print()
    print("scenario-1 candidates (hard hand impact, static body):")
    for aid in SCENARIO1_ACTIVITIES:
        c = sum(1 for _ in scan(root, device, activities=[aid]))
        print(f"  A{aid:03d} {ACTIVITY_NAMES.get(aid,'?'):<40} {c:>4} records")


def _selftest():
    """Verify the resampling ratio and contract conformance without needing
    the dataset on disk."""
    assert RAW_FS * RESAMPLE_UP / RESAMPLE_DOWN == 50.0, "resample ratio wrong"
    print(f"resample: {RAW_FS} Hz * {RESAMPLE_UP}/{RESAMPLE_DOWN} = "
          f"{RAW_FS*RESAMPLE_UP/RESAMPLE_DOWN} Hz")

    # A 40 Hz tone is above the 25 Hz Nyquist of the target rate. With a
    # correct anti-alias filter it is attenuated; naive decimation would fold
    # it to 10 Hz at full amplitude.
    n = int(RAW_FS * 20)
    t = np.arange(n) / RAW_FS
    tone = np.sin(2 * np.pi * 40 * t)[:, None].repeat(3, axis=1)
    raw = downsample(tone)
    out = downsample_trimmed(tone)
    naive = tone[::5]
    print(f"  40 Hz tone (above the 25 Hz Nyquist):")
    print(f"    trimmed resample_poly {np.abs(out).max():.5f}  (want ~0)")
    print(f"    untrimmed             {np.abs(raw).max():.5f}  (edge ringing)")
    print(f"    naive decimation      {np.abs(naive).max():.5f}  (ALIASED, "
          f"folds to 10 Hz at full amplitude)")
    assert np.abs(out).max() < 0.01, "anti-alias filter is not working"

    # A 5 Hz tone is in band and must survive.
    tone5 = np.sin(2 * np.pi * 5 * t)[:, None].repeat(3, axis=1)
    o5 = downsample_trimmed(tone5)
    print(f"  5 Hz tone (in band): amplitude {np.abs(o5).max():.4f}  (want ~1)")
    assert np.abs(o5).max() > 0.90, "in-band signal was attenuated"
    print(f"  length {len(tone)} -> {len(out)} "
          f"(20 s at 238 Hz -> {len(out)/TARGET_FS:.0f} s at 50 Hz after trim)")

    # Matched path must also reject the out-of-band tone, and must reach 50 Hz.
    assert UMAFALL_FS * 1 == RAW_FS * MATCH_UP / MATCH_DOWN, "matched ratio wrong"
    mp = downsample_umafall_path(tone)
    mp5 = downsample_umafall_path(tone5)
    print(f"  matched path (238->20->50): 40 Hz tone {np.abs(mp).max():.4f}, "
          f"5 Hz tone {np.abs(mp5).max():.4f}, length {len(mp)}")
    assert np.abs(mp).max() < 0.05, "matched path aliasing"

    # Contract shape.
    rec = Record(path=Path("synthetic.dat"), subject="S00",
                 activity="synthetic", activity_id=26, trial="01",
                 t=np.arange(1000) / TARGET_FS,
                 acc=np.ones((1000, 3), np.float32) / np.sqrt(3),
                 gyro=np.zeros((1000, 3), np.float32), fs_raw=RAW_FS,
                 state="seated hand activity", is_fall=False)
    ws = list(iter_windows(rec))
    w, m = ws[0]
    assert set(w) == {"t", "acc", "gyro", "hr", "temp", "fs"}, "extra keys"
    assert w["acc"].shape == (WINDOW_N, 3) and w["fs"] == TARGET_FS
    assert w["hr"] is None and w["temp"] is None
    assert m["state"] == "seated hand activity" and m["dataset"] == "FallAllD"
    assert abs(ws[1][0]["t"] - ws[0][0]["t"] - HOP_N / TARGET_FS) < 1e-9
    print(f"  windows: {len(ws)}, contract keys correct")
    print("SELFTEST PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description="FallAllD loader (wrist = D2)")
    ap.add_argument("--inspect", metavar="ACC_FILE")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--device", type=int, default=WRIST_DEVICE)
    ap.add_argument("--mapping", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.inspect:
        _inspect(a.inspect)
    elif a.scan:
        _scan_report(a.root, a.device)
    elif a.mapping:
        print(mapping_table_markdown())
    elif a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
