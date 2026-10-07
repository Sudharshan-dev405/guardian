"""a7_descriptives.py -- read-only. Descriptive stats for five context
features, UMAFall wrist non-fall windows, grouped by activity state.
Not part of any package. Modifies nothing, trains nothing.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from data.loader import scan, iter_windows
from streams.context import extract_features, FEATURE_NAMES, GAIT_BAND, STATES

FEATURES = ("tilt_mean", "tilt_std", "svm_var", "cadence_hz", "jerk_rms")
OUT_DIR = Path(__file__).resolve().parent / "a7_outputs"


def main():
    missing = [f for f in FEATURES if f not in FEATURE_NAMES]
    if missing:
        print(f"ABORT: features not in FEATURE_NAMES: {missing}")
        print(f"available: {FEATURE_NAMES}")
        return
    cols = {f: FEATURE_NAMES.index(f) for f in FEATURES}

    vals = defaultdict(lambda: defaultdict(list))   # state -> feature -> list
    n_rec = 0
    for rec in scan("data/raw/UMAFall"):
        if rec.is_fall or rec.state is None:
            continue
        n_rec += 1
        for w, m in iter_windows(rec):
            f = extract_features(w, GAIT_BAND)
            for name, i in cols.items():
                vals[m["state"]][name].append(float(f[i]))

    if not vals:
        print("ABORT: no non-fall windows collected")
        return

    print(f"non-fall records read: {n_rec}")
    print("\nwindow count per state (check against 969 / 1200 / 1776 / 413):")
    for s in STATES:
        n = len(vals[s][FEATURES[0]]) if s in vals else 0
        print(f"  {s:<24} {n}")
    total = sum(len(vals[s][FEATURES[0]]) for s in vals)
    print(f"  {'TOTAL':<24} {total}")

    rows = []
    for feat in FEATURES:
        for s in STATES:
            if s not in vals or not vals[s][feat]:
                continue
            x = np.asarray(vals[s][feat], dtype=float)
            q75, q25 = np.percentile(x, [75, 25])
            rows.append({
                "feature": feat,
                "state": s,
                "n": len(x),
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "std": float(x.std(ddof=1)),
                "iqr": float(q75 - q25),
                "skewness": float(skew(x)),
            })

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "descriptives.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'feature':<12}{'state':<24}{'n':>6}{'mean':>12}{'median':>12}"
          f"{'std':>12}{'IQR':>12}{'skew':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['feature']:<12}{r['state']:<24}{r['n']:>6}"
              f"{r['mean']:>12.6f}{r['median']:>12.6f}{r['std']:>12.6f}"
              f"{r['iqr']:>12.6f}{r['skewness']:>10.4f}")
    print(f"\nsaved -> {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()