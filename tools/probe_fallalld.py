"""
tools/probe_fallalld.py -- work out FallAllD's structure empirically before
writing a loader against it.

Answers three questions:

1. Do activity IDs 001-044 and 101-135 correspond to ADLs and falls?
   Checked against the published counts: 4883 ADL and 1722 fall instances.

2. Which of D1/D2/D3 is the WRIST? Published sources disagree on whether the
   three loggers sit at neck/wrist/waist or neck/chest/waist. A wrist device
   sees much higher angular rate and much more orientation change during
   ordinary ADLs than a trunk-mounted one, so this is measurable rather than
   something to guess. Reported per device over ADL records:
     - median gyro resultant                 wrist should be highest
     - spread of the gravity direction       wrist should be highest
     - median |acc| resultant                all should sit near 1 g
   The wrist is the device that is clearly separated on the first two.

3. Are the raw counts scaled as expected? LSM9DS1 at +/-8 g full scale on a
   16-bit signed axis gives 4096 LSB per g; at +/-2000 dps, 0.07 dps per LSB.
   If median |acc| lands near 1.0 g after dividing by 4096, that is confirmed.

    py -m tools.probe_fallalld
    py -m tools.probe_fallalld --root data\\raw\\FallAllD\\FallAllD --n 60
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ACC_LSB_PER_G = 4096.0          # +/-8 g on a 16-bit signed axis
GYR_DPS_PER_LSB = 0.07          # +/-2000 dps, LSM9DS1 datasheet
NAME_RE = re.compile(r"S(\d+)_D(\d+)_A(\d+)_T(\d+)_([AGMB])\.dat", re.I)


def parse_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    s, d, a, t, sensor = m.groups()
    return dict(subject=f"S{s}", device=int(d), activity=int(a),
                trial=int(t), sensor=sensor.upper())


def read_dat(path):
    """Three integer columns, no header."""
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def q1_activity_split(root):
    print("=" * 70)
    print("1. ACTIVITY ID SPLIT")
    counts = defaultdict(int)
    per_id = defaultdict(int)
    for p in Path(root).glob("*_A.dat"):
        meta = parse_name(p.name)
        if not meta:
            continue
        band = "001-044" if meta["activity"] <= 44 else "101-135"
        counts[band] += 1
        per_id[meta["activity"]] += 1
    total = sum(counts.values())
    print(f"   accelerometer files: {total}   (expected 6605)")
    for band in sorted(counts):
        print(f"   activity {band}: {counts[band]:>5} instances")
    print()
    print("   published: 4883 ADL, 1722 fall")
    lo = counts.get("001-044", 0)
    hi = counts.get("101-135", 0)
    if abs(lo * 3 - 4883 * 3) < 1e-9 or lo in (4883, 4883 * 3):
        pass
    print(f"   -> 001-044 is {'ADL' if lo > hi else 'FALL'}, "
          f"101-135 is {'FALL' if lo > hi else 'ADL'}"
          f"   (per device: {lo//3} and {hi//3})")
    print()
    print("   instances per activity id (all devices):")
    ids = sorted(per_id)
    for i in range(0, len(ids), 10):
        chunk = ids[i:i + 10]
        print("     " + "  ".join(f"A{k:03d}:{per_id[k]:>3}" for k in chunk))


def q2_which_device_is_wrist(root, n_per_device):
    print()
    print("=" * 70)
    print("2. WHICH DEVICE IS THE WRIST")
    root = Path(root)
    stats = defaultdict(lambda: dict(gyro=[], tiltspread=[], accmed=[], n=0))

    for device in (1, 2, 3):
        # ADL records only -- falls would swamp the comparison.
        files = sorted(root.glob(f"S*_D{device}_A0*_T*_A.dat"))[:n_per_device]
        for pa in files:
            pg = pa.with_name(pa.name[:-6] + "_G.dat")
            if not pg.exists():
                continue
            try:
                acc = read_dat(pa) / ACC_LSB_PER_G
                gyr = read_dat(pg) * GYR_DPS_PER_LSB
            except Exception:                               # noqa: BLE001
                continue
            if acc.ndim != 2 or acc.shape[1] != 3 or len(acc) < 100:
                continue
            s = stats[device]
            s["accmed"].append(float(np.median(np.linalg.norm(acc, axis=1))))
            s["gyro"].append(float(np.median(np.linalg.norm(gyr, axis=1))))
            # Spread of the gravity direction: how much the device's own
            # orientation moves over the record.
            g = acc / np.clip(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9, None)
            s["tiltspread"].append(float(np.mean(np.std(g, axis=0))))
            s["n"] += 1

    print(f"   {'dev':>4} {'files':>6} {'med |acc| g':>12} "
          f"{'med |gyro| dps':>15} {'orientation spread':>19}")
    summary = {}
    for device in (1, 2, 3):
        s = stats[device]
        if not s["n"]:
            print(f"   D{device:<3} {'0':>6}  no readable pairs")
            continue
        a = float(np.median(s["accmed"]))
        g = float(np.median(s["gyro"]))
        t = float(np.median(s["tiltspread"]))
        summary[device] = (g, t)
        print(f"   D{device:<3} {s['n']:>6} {a:>12.3f} {g:>15.2f} {t:>19.4f}")

    if len(summary) == 3:
        by_gyro = sorted(summary, key=lambda d: -summary[d][0])
        by_tilt = sorted(summary, key=lambda d: -summary[d][1])
        print()
        print(f"   ranked by angular rate:        D{by_gyro[0]} > D{by_gyro[1]} > D{by_gyro[2]}")
        print(f"   ranked by orientation spread:  D{by_tilt[0]} > D{by_tilt[1]} > D{by_tilt[2]}")
        if by_gyro[0] == by_tilt[0]:
            lead = summary[by_gyro[0]][0] / max(summary[by_gyro[1]][0], 1e-9)
            print(f"\n   -> D{by_gyro[0]} is the WRIST "
                  f"({lead:.1f}x the angular rate of the next device)")
            if lead < 1.5:
                print("      WARNING: margin is small. Do not trust this; "
                      "get the device table from the paper instead.")
        else:
            print("\n   -> INCONCLUSIVE: the two rankings disagree. "
                  "Get the device table from the paper before proceeding.")


def q3_scaling(root):
    print()
    print("=" * 70)
    print("3. SCALING CHECK")
    root = Path(root)
    pa = sorted(root.glob("S01_D*_A0*_T*_A.dat"))[:1]
    if not pa:
        print("   no files found")
        return
    raw = read_dat(pa[0])
    print(f"   file: {pa[0].name}")
    print(f"   shape: {raw.shape}   duration at 238 Hz: {len(raw)/238:.1f} s "
          f"(expected ~20 s)")
    print(f"   raw range per axis: "
          f"{raw.min(axis=0).astype(int).tolist()} .. "
          f"{raw.max(axis=0).astype(int).tolist()}")
    g = raw / ACC_LSB_PER_G
    print(f"   after /{ACC_LSB_PER_G:.0f}: median |acc| = "
          f"{np.median(np.linalg.norm(g, axis=1)):.3f} g   (expected ~1.0)")
    print(f"   saturation: +/-8 g is +/-{8*ACC_LSB_PER_G:.0f} counts; "
          f"{float(np.mean(np.abs(raw) >= 8*ACC_LSB_PER_G - 2))*100:.2f}% "
          f"of samples at rail")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Probe FallAllD structure")
    ap.add_argument("--root", default=r"data\raw\FallAllD\FallAllD")
    ap.add_argument("--n", type=int, default=40,
                    help="ADL records per device to sample for the wrist test")
    a = ap.parse_args(argv)
    if not Path(a.root).exists():
        print(f"not found: {a.root}")
        return
    q1_activity_split(a.root)
    q2_which_device_is_wrist(a.root, a.n)
    q3_scaling(a.root)


if __name__ == "__main__":
    main()
