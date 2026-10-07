"""a7_chisquare.py -- read-only. 3x2 chi-square: UMAFall fall direction vs
whether a fall record has >=1 post-impact window hitting the fusion
suppression rule. Definitions copied from testing/tools/context_on_falls.py.
Modifies nothing, trains nothing.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from data.loader import scan, iter_windows, TARGET_FS
from streams.context import ContextStream

SUPPRESS_STATE = "seated hand activity"   # context_on_falls.py
SUPPRESS_CONF = 0.7                       # context_on_falls.py
MODEL = "models/context.joblib"
DIRECTIONS = ("forward", "backward", "lateral")
OUT_DIR = Path(__file__).resolve().parent / "a7_outputs"


def impact_time(rec):
    """Seconds to peak resultant acceleration. Same as context_on_falls.py."""
    svm = np.linalg.norm(np.asarray(rec.acc, dtype=float), axis=1)
    return float(int(np.argmax(svm)) / TARGET_FS)


def direction_of(rec):
    a = (rec.activity or "").lower()
    for d in DIRECTIONS:
        if d in a:
            return d
    return None


def main():
    ctx = ContextStream.load(MODEL)

    counts = {d: [0, 0] for d in DIRECTIONS}   # direction -> [yes, no]
    unmatched = []
    n_rec = 0
    total_affected = 0

    for rec in scan("data/raw/UMAFall"):
        if not rec.is_fall:
            continue
        n_rec += 1
        d = direction_of(rec)
        if d is None:
            unmatched.append(rec.path.name)
            continue
        t_imp = impact_time(rec)
        hit = False
        for w, m in iter_windows(rec):
            ctx.score(w)                    # score first, then read state
            if w["t"] < t_imp:
                continue
            if ctx.last_state == SUPPRESS_STATE and ctx.last_confidence > SUPPRESS_CONF:
                hit = True
                break
        counts[d][0 if hit else 1] += 1
        if hit:
            total_affected += 1

    if n_rec == 0:
        print("ABORT: no fall records found")
        return

    print(f"fall records read: {n_rec}   (check value: 189)")
    if unmatched:
        print(f"records with no direction token: {len(unmatched)}")
        for nm in unmatched:
            print("  " + nm)

    print(f"\nrecords affected: {total_affected} / {n_rec}   (check value: 77 / 189)")
    print("\nper direction (check values: forward 47.7%, backward 1.5%, lateral 78.9%)")
    for d in DIRECTIONS:
        yes, no = counts[d]
        tot = yes + no
        pct = yes / tot * 100 if tot else float("nan")
        print(f"  {d:<10} {yes:>3} yes / {tot:<3} records  ({pct:.1f}%)")

    observed = np.array([counts[d] for d in DIRECTIONS], dtype=float)
    print("\nOBSERVED 3x2 table   rows = direction, cols = [affected, not affected]")
    print(f"{'':<10}{'affected':>10}{'not_affected':>14}{'total':>8}")
    for i, d in enumerate(DIRECTIONS):
        print(f"{d:<10}{observed[i,0]:>10.0f}{observed[i,1]:>14.0f}"
              f"{observed[i].sum():>8.0f}")
    print(f"{'total':<10}{observed[:,0].sum():>10.0f}"
          f"{observed[:,1].sum():>14.0f}{observed.sum():>8.0f}")

    if (observed.sum(axis=1) == 0).any() or (observed.sum(axis=0) == 0).any():
        print("\nABORT: a row or column total is zero -- chi-square undefined")
        return

    chi2, p, dof, expected = chi2_contingency(observed, correction=False)

    print(f"\nchi-square statistic : {chi2!r}")
    print(f"p-value              : {p:.6e}")
    print(f"p-value (repr)       : {p!r}")
    print(f"degrees of freedom   : {dof}")
    print("\nEXPECTED counts")
    print(f"{'':<10}{'affected':>12}{'not_affected':>16}")
    for i, d in enumerate(DIRECTIONS):
        print(f"{d:<10}{expected[i,0]:>12.6f}{expected[i,1]:>16.6f}")
    n_small = int((expected < 5).sum())
    print(f"\nexpected cells below 5: {n_small} of {expected.size}")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "contingency_observed.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["direction", "affected", "not_affected", "total", "pct_affected"])
        for i, d in enumerate(DIRECTIONS):
            tot = observed[i].sum()
            wr.writerow([d, int(observed[i, 0]), int(observed[i, 1]), int(tot),
                         observed[i, 0] / tot * 100 if tot else ""])
    with open(OUT_DIR / "contingency_expected.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["direction", "affected_expected", "not_affected_expected"])
        for i, d in enumerate(DIRECTIONS):
            wr.writerow([d, expected[i, 0], expected[i, 1]])
    with open(OUT_DIR / "chisquare_result.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["metric", "value"])
        wr.writerow(["chi2_statistic", repr(chi2)])
        wr.writerow(["p_value", f"{p:.6e}"])
        wr.writerow(["p_value_repr", repr(p)])
        wr.writerow(["degrees_of_freedom", dof])
        wr.writerow(["n_records", int(observed.sum())])
        wr.writerow(["yates_correction", "False"])
        wr.writerow(["expected_cells_below_5", n_small])
        wr.writerow(["suppress_state", SUPPRESS_STATE])
        wr.writerow(["suppress_conf_threshold", SUPPRESS_CONF])
        wr.writerow(["post_impact_definition", "window t >= argmax(SVM)/50"])
    print(f"\nsaved -> {OUT_DIR}\\contingency_observed.csv, "
          f"contingency_expected.csv, chisquare_result.csv")


if __name__ == "__main__":
    main()