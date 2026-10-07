"""scratch_counts.py -- read-only counts for the preprocessing/sampling plan.
Not part of any package. Imports the loaders, changes nothing.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data.loader as UMA
import data.loader_fallalld as FAD
from streams.motion import GATE_SVM_G, EVENT_HOLD_SEC

STATES = ("stationary", "ambulating", "seated hand activity", "lying/immobile")


def tally(records, iter_windows, fs):
    win, rec_n = defaultdict(int), defaultdict(int)
    d = dict(total_w=0, fall_w=0, adl_w=0, adl_mapped_w=0, unmapped_w=0,
             no_gyro_recs=0, no_gyro_w=0, gated=0, ungated=0,
             hold_w=0, hold_w_fall=0, too_short=0, n_recs=0, max_dur=0.0)
    for rec in records:
        d["n_recs"] += 1
        key = rec.state or ("FALL" if rec.is_fall else "UNMAPPED")
        rec_n[key] += 1
        if rec.gyro is None:
            d["no_gyro_recs"] += 1
        d["max_dur"] = max(d["max_dur"], rec.duration)

        svm = np.linalg.norm(np.asarray(rec.acc, dtype=float), axis=1)
        hit = svm > GATE_SVM_G
        if hit.any():
            d["gated"] += 1
            trig_t = float(np.flatnonzero(hit)[0]) / fs
        else:
            d["ungated"] += 1
            trig_t = None

        nw = 0
        for w, _m in iter_windows(rec):
            nw += 1
            d["total_w"] += 1
            win[key] += 1
            if rec.is_fall:
                d["fall_w"] += 1
            else:
                d["adl_w"] += 1
                d["adl_mapped_w" if rec.state else "unmapped_w"] += 1
            if rec.gyro is None:
                d["no_gyro_w"] += 1
            if trig_t is not None and trig_t <= w["t"] <= trig_t + EVENT_HOLD_SEC:
                d["hold_w"] += 1
                if rec.is_fall:
                    d["hold_w_fall"] += 1
        if nw == 0:
            d["too_short"] += 1
    return d, dict(win), dict(rec_n)


def report(name, d, win, rec_n):
    print(f"\n===== {name} =====")
    print(f"records read              {d['n_recs']}")
    print(f"records yielding 0 windows {d['too_short']}  (shorter than 2.5 s after resample)")
    print(f"longest record             {d['max_dur']:.2f} s")
    print(f"total windows             {d['total_w']}")
    print("\n| bucket | records | windows |")
    print("|---|---|---|")
    for k in list(STATES) + ["FALL", "UNMAPPED"]:
        if k in rec_n or k in win:
            print(f"| {k} | {rec_n.get(k, 0)} | {win.get(k, 0)} |")
    print(f"\nfall windows              {d['fall_w']}")
    print(f"ADL windows (all non-fall) {d['adl_w']}")
    print(f"ADL windows (mapped only)  {d['adl_mapped_w']}")
    print(f"unmapped windows           {d['unmapped_w']}")
    if d["fall_w"]:
        print(f"ratio fall:ADL(all)        1 : {d['adl_w'] / d['fall_w']:.6f}")
        print(f"ratio fall:ADL(mapped)     1 : {d['adl_mapped_w'] / d['fall_w']:.6f}")
    else:
        print("ratio                      EMPTY - no fall windows")
    print(f"\ngate fired (SVM > {GATE_SVM_G} g)   {d['gated']} records")
    print(f"gate never fired           {d['ungated']} records")
    print(f"windows inside {EVENT_HOLD_SEC:.0f} s hold  {d['hold_w']}  (of which fall records: {d['hold_w_fall']})")
    print(f"\nrecords with gyro=None    {d['no_gyro_recs']}")
    print(f"windows with gyro=None    {d['no_gyro_w']}")


if __name__ == "__main__":
    report("UMAFall wrist (SensorID=2)",
           *tally(UMA.scan("data/raw/UMAFall"), UMA.iter_windows, UMA.TARGET_FS))
    report(f"FallAllD wrist (D{FAD.WRIST_DEVICE})",
           *tally(FAD.scan(FAD.DEFAULT_ROOT, FAD.WRIST_DEVICE),
                  FAD.iter_windows, FAD.TARGET_FS))