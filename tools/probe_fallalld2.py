"""
tools/probe_fallalld2.py -- decisive wrist-device identification for FallAllD.

The first probe was inconclusive: it ranked devices by angular rate and by
within-record orientation spread, and the two rankings disagreed. The spread
metric was contaminated by linear acceleration because it used the raw
normalised accelerometer instead of a low-passed gravity estimate.

This uses a sharper discriminator.

A trunk or neck device is strapped in a FIXED orientation. Across every
upright record, its mean gravity direction lands in nearly the same place.
A WRIST device points wherever the forearm happens to be, so its mean gravity
direction scatters all over the sphere between records.

So: estimate the gravity direction per record (low-passed, then averaged),
and measure how much those per-record directions disagree WITH EACH OTHER.

  between-record dispersion   wrist HIGH, trunk LOW      <- the decisive one
  mean pairwise angle (deg)   wrist near 60-90, trunk small
  within-record tilt change   wrist high
  median gyro resultant       wrist high

Run per subject and pooled, because a subject who wore a device loosely could
skew a pooled number.

    py -m tools.probe_fallalld2
    py -m tools.probe_fallalld2 --subjects S01 S02 S03 --n 60
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal

ACC_LSB_PER_G = 4096.0
GYR_DPS_PER_LSB = 0.07
FS = 238.0
NAME_RE = re.compile(r"S(\d+)_D(\d+)_A(\d+)_T(\d+)_A\.dat", re.I)


def gravity(acc, fs=FS, corner=0.5):
    b, a = signal.butter(2, min(corner / (fs / 2), 0.99), btype="low")
    if acc.shape[0] <= 3 * max(len(a), len(b)):
        return acc.mean(axis=0, keepdims=True).repeat(acc.shape[0], axis=0)
    return signal.filtfilt(b, a, acc, axis=0)


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def analyse(root, subjects, n_per):
    root = Path(root)
    out = {}
    for device in (1, 2, 3):
        mean_dirs, within, gyros = [], [], []
        for subj in subjects:
            files = sorted(root.glob(f"{subj}_D{device}_A0*_T*_A.dat"))[:n_per]
            for pa in files:
                pg = pa.with_name(pa.name[:-6] + "_G.dat")
                try:
                    acc = np.loadtxt(pa, delimiter=",", dtype=float) / ACC_LSB_PER_G
                except Exception:
                    continue
                if acc.ndim != 2 or acc.shape[1] != 3 or len(acc) < 500:
                    continue
                g = gravity(acc)
                gu = unit(g)
                mean_dirs.append(unit(gu.mean(axis=0)))
                # within-record: mean angle from the record's own mean direction
                cosang = np.clip(gu @ unit(gu.mean(axis=0)), -1, 1)
                within.append(float(np.degrees(np.arccos(cosang)).mean()))
                if pg.exists():
                    try:
                        gyr = np.loadtxt(pg, delimiter=",", dtype=float) * GYR_DPS_PER_LSB
                        gyros.append(float(np.median(np.linalg.norm(gyr, axis=1))))
                    except Exception:
                        pass
        if len(mean_dirs) < 5:
            out[device] = None
            continue
        M = np.vstack(mean_dirs)
        # Between-record dispersion: 1 - |mean resultant length|.
        # 0 = every record points the same way, 1 = uniformly scattered.
        R = float(np.linalg.norm(M.mean(axis=0)))
        disp = 1.0 - R
        # Mean pairwise angle between record directions.
        C = np.clip(M @ M.T, -1, 1)
        iu = np.triu_indices(len(M), 1)
        pair = float(np.degrees(np.arccos(C[iu])).mean())
        out[device] = dict(n=len(M), disp=disp, pair=pair,
                           within=float(np.median(within)),
                           gyro=float(np.median(gyros)) if gyros else float("nan"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data\raw\FallAllD\FallAllD")
    ap.add_argument("--subjects", nargs="*",
                    default=["S01", "S02", "S03", "S04", "S05"])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)

    if not Path(a.root).exists():
        print(f"not found: {a.root}")
        return

    print("=" * 74)
    print("WRIST DEVICE IDENTIFICATION -- between-record gravity dispersion")
    print("A fixed trunk/neck mount points the same way every record (low).")
    print("A wrist points wherever the forearm is (high).")
    print("=" * 74)

    # Per subject, so one badly-worn device cannot carry the pooled result.
    votes = defaultdict(int)
    for subj in a.subjects:
        res = analyse(a.root, [subj], a.n)
        line = [f"  {subj}:"]
        best, bestv = None, -1
        for d in (1, 2, 3):
            r = res.get(d)
            if not r:
                line.append(f" D{d} n/a")
                continue
            line.append(f" D{d} disp={r['disp']:.3f}")
            if r["disp"] > bestv:
                best, bestv = d, r["disp"]
        if best:
            votes[best] += 1
            line.append(f"  -> D{best}")
        print("".join(line))

    print()
    res = analyse(a.root, a.subjects, a.n)
    print(f"  POOLED over {', '.join(a.subjects)}")
    print(f"  {'dev':>4} {'recs':>5} {'between-rec disp':>18} "
          f"{'mean pair angle':>16} {'within-rec deg':>15} {'med gyro dps':>13}")
    for d in (1, 2, 3):
        r = res.get(d)
        if not r:
            print(f"  D{d:<3} no data")
            continue
        print(f"  D{d:<3} {r['n']:>5} {r['disp']:>18.3f} "
              f"{r['pair']:>15.1f}d {r['within']:>14.1f}d {r['gyro']:>13.2f}")

    ok = {d: r for d, r in res.items() if r}
    if len(ok) == 3:
        rank = sorted(ok, key=lambda d: -ok[d]["disp"])
        lead = ok[rank[0]]["disp"] / max(ok[rank[1]]["disp"], 1e-9)
        agree = max(votes, key=votes.get) if votes else None
        print()
        print(f"  dispersion ranking: D{rank[0]} > D{rank[1]} > D{rank[2]}"
              f"   (lead {lead:.2f}x)")
        print(f"  per-subject votes:  " +
              ", ".join(f"D{d}:{votes[d]}" for d in sorted(votes)))
        if agree == rank[0] and votes[agree] >= max(3, len(a.subjects) - 1) and lead > 1.3:
            print(f"\n  CONCLUSIVE: D{rank[0]} is the WRIST")
        else:
            print("\n  STILL INCONCLUSIVE -- get the device table from the paper.")


if __name__ == "__main__":
    main()
