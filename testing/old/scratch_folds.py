"""scratch_folds.py -- read-only. Subject spans + UMAFall fall-direction split."""
import re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data.loader as UMA
import data.loader_fallalld as FAD
from streams.motion import GATE_SVM_G

DIRS = ("forwardFall", "backwardFall", "lateralFall")

def main():
    u_subj, f_subj = set(), set()
    dir_rec, dir_win = defaultdict(int), defaultdict(int)
    other_fall = []
    per_subj = defaultdict(lambda: dict(rec=0, fall_rec=0, adl_rec=0,
                                        win=0, fall_win=0, gated=0))
    for rec in UMA.scan("data/raw/UMAFall"):
        u_subj.add(rec.subject)
        s = per_subj[rec.subject]
        s["rec"] += 1
        s["fall_rec" if rec.is_fall else "adl_rec"] += 1
        nw = sum(1 for _ in UMA.iter_windows(rec))
        s["win"] += nw
        if rec.is_fall:
            s["fall_win"] += nw
        svm = np.linalg.norm(np.asarray(rec.acc, dtype=float), axis=1)
        if (svm > GATE_SVM_G).any():
            s["gated"] += 1
        if rec.is_fall:
            hit = next((d for d in DIRS if d.lower() in rec.path.name.lower()), None)
            if hit is None:
                other_fall.append(rec.path.name)
            else:
                dir_rec[hit] += 1
                dir_win[hit] += nw
    for rec in FAD.scan(FAD.DEFAULT_ROOT, FAD.WRIST_DEVICE):
        f_subj.add(rec.subject)

    print(f"UMAFall subjects in 617 records: {len(u_subj)}  {sorted(u_subj)}")
    print(f"FallAllD subjects in 2515 records: {len(f_subj)}  {sorted(f_subj)}")
    print("\n| subject | records | ADL rec | fall rec | windows | fall windows | gated rec |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(per_subj):
        s = per_subj[k]
        print(f"| {k} | {s['rec']} | {s['adl_rec']} | {s['fall_rec']} | "
              f"{s['win']} | {s['fall_win']} | {s['gated']} |")
    print("\n| fall direction | records | windows |")
    print("|---|---|---|")
    tr = tw = 0
    for d in DIRS:
        print(f"| {d} | {dir_rec[d]} | {dir_win[d]} |")
        tr += dir_rec[d]; tw += dir_win[d]
    print(f"| (total matched) | {tr} | {tw} |")
    if other_fall:
        print(f"\nfall records matching no direction token: {len(other_fall)}")
        for n in sorted(set(other_fall)):
            print("  " + n)
    else:
        print("\nall fall records matched a direction token")

if __name__ == "__main__":
    main()