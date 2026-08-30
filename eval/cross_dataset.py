"""
eval/cross_dataset.py -- run the ALREADY-TRAINED UMAFall models, unchanged,
over FallAllD. Cross-dataset generalisation measurement.

NO RETRAINING. NO FINE-TUNING. NO THRESHOLD ADJUSTMENT. The models are loaded
from models/*.joblib exactly as they were fitted on UMAFall, and this file
never calls fit(). A large drop is the expected result and IS the finding: a
motion AUC of 0.998 on staged UMAFall falls is not presentable as fall
detection performance, and measuring what happens on another lab's data is
the closest available substitute for real-world falls.

If the motion AUC holds above ~0.95 here, be suspicious rather than pleased,
and check that the loader is not accidentally reading UMAFall.

Reports:
  - motion AUC on FallAllD vs the UMAFall figure
  - gate coverage: does 2.5 g still catch ~91% of falls, and what is the ADL
    trigger rate on this dataset
  - where the drop comes from: stage-1 gate vs stage-2 classifier
  - context macro F1 and per-state F1 vs the UMAFall figures
  - calibration on new data (ECE), since a model calibrated on one dataset is
    usually not calibrated on another
  - optionally the same numbers with FallAllD forced through UMAFall's
    238->20->50 Hz signal path, to separate "sampling path" from "different
    subjects and protocol" as the cause of any context drop

    py -m eval.cross_dataset
    py -m eval.cross_dataset --matched      (adds the matched-path comparison)
    py -m eval.cross_dataset --limit 300    (quick pass)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader_fallalld import (scan, iter_windows, STATES,          # noqa: E402
                                  DEFAULT_ROOT)
from streams.context import ContextStream, expected_calibration_error  # noqa: E402
from streams.motion import MotionStream, segments_from_record          # noqa: E402
from streams.motion import GATE_SVM_G                                   # noqa: E402

# UMAFall reference figures, for the comparison column.
REF = {
    "motion_auc": 0.999,
    "gate_fall": 0.910,
    "gate_adl": 0.147,
    "macro_f1": 0.71,
    "f1": {"ambulating": 0.90, "seated hand activity": 0.75,
           "lying/immobile": 0.70, "stationary": 0.51},
    "ece": 0.0483,
}


def _delta(new, ref):
    d = new - ref
    return f"{new:.3f}  (UMAFall {ref:.3f}, {d:+.3f})"


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------

def eval_motion(root, matched, limit):
    print("=" * 74)
    print("MOTION STREAM on FallAllD -- UMAFall-trained model, not retrained")
    print("=" * 74)

    ms = MotionStream.load("models/motion.joblib")
    X, y, subj, gated = [], [], [], []
    for rec in scan(root, limit=limit, match_umafall_path=matched):
        for f, lab, s, hit in segments_from_record(rec):
            X.append(f); y.append(lab); subj.append(s); gated.append(hit)
    if not X:
        print("  no segments")
        return
    X = np.asarray(X); y = np.asarray(y)
    gated = np.asarray(gated)

    fall_i = list(ms.classes_).index(1)
    p_raw = ms.model.predict_proba(X)
    from streams.motion import _softmax
    p = _softmax(np.log(np.clip(p_raw, 1e-8, 1.0)) / ms.temperature)[:, fall_i]

    print(f"  {len(X)} records  falls={int(y.sum())}  ADLs={int((1-y).sum())}  "
          f"subjects={len(set(subj))}")

    # Stage 1: the gate, which involves no learned model at all.
    gf, ga = gated[y == 1].mean(), gated[y == 0].mean()
    print(f"\n  STAGE 1 -- {GATE_SVM_G} g accelerometer gate")
    print(f"    catches {gf*100:.1f}% of falls   (UMAFall {REF['gate_fall']*100:.1f}%)")
    print(f"    fires on {ga*100:.1f}% of ADLs   (UMAFall {REF['gate_adl']*100:.1f}%)")

    # Stage 2: the classifier, on everything and on gated records only.
    def roc(mask, label):
        m = mask & np.ones(len(y), bool)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            print(f"    {label}: too few samples")
            return None
        auc = roc_auc_score(y[m], p[m])
        fpr, tpr, thr = roc_curve(y[m], p[m])
        k = fpr <= 0.05
        s95 = float(tpr[k].max()) if k.any() else 0.0
        print(f"    {label}  n={int(m.sum())} "
              f"(falls {int(y[m].sum())}, ADLs {int((1-y[m]).sum())})")
        print(f"      AUC {auc:.3f}   sens@95%spec {s95:.3f}")
        return auc

    print(f"\n  STAGE 2 -- calibrated classifier")
    auc_all = roc(np.ones(len(y), bool), "ALL RECORDS     ")
    auc_gate = roc(gated, "GATE-CONDITIONAL")

    if auc_gate is not None:
        print(f"\n  gate-conditional AUC: {_delta(auc_gate, REF['motion_auc'])}")
        drop = REF["motion_auc"] - auc_gate
        if drop < 0.05:
            print("    Held up. Be suspicious before being pleased -- check the")
            print("    loader is actually reading FallAllD, and remember staged")
            print("    falls are stereotyped in both datasets.")
        else:
            print(f"    Dropped {drop:.3f}. This is the expected result and it is")
            print("    the finding: performance on staged falls from one lab does")
            print("    not transfer intact to another.")

    # Attribute the drop.
    print(f"\n  WHERE THE DROP COMES FROM")
    print(f"    gate recall on falls:  {gf*100:.1f}% vs {REF['gate_fall']*100:.1f}% "
          f"({(gf-REF['gate_fall'])*100:+.1f} pts)")
    if auc_gate is not None:
        print(f"    classifier AUC given gated: {auc_gate:.3f} vs "
              f"{REF['motion_auc']:.3f} ({auc_gate-REF['motion_auc']:+.3f})")
    print("    A gate loss means impact magnitudes differ between labs.")
    print("    A classifier loss means the waveform shape differs.")


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

def eval_context(root, matched, limit):
    print()
    print("=" * 74)
    print("CONTEXT STREAM on FallAllD -- UMAFall-trained model, not retrained")
    print("=" * 74)

    cs = ContextStream.load("models/context.joblib")

    # Rejected windows are excluded from F1 but counted and reported.
    y_true, y_pred, probs = [], [], []
    rejected = 0
    for rec in scan(root, limit=limit, match_umafall_path=matched):
        if rec.is_fall or rec.state is None:
            continue
        for w, m in iter_windows(rec):
            state, conf, pdict = cs.predict(w)
            if state == "unknown":
                rejected += 1
                continue
            y_true.append(m["state"])
            y_pred.append(state)
            probs.append([pdict.get(c, 0.0) for c in cs.classes_])

    total = len(y_true) + rejected
    if len(y_true) < 20:
        print("  too few scored windows")
        return

    print(f"  {total} mapped ADL windows, {rejected} open-set rejected "
          f"({rejected/max(total,1)*100:.1f}%), {len(y_true)} scored")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))
    labels = [s for s in STATES if s in set(y_true) | set(y_pred)]
    print("confusion matrix, rows true, order:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    print(f"\n  macro F1: {_delta(macro, REF['macro_f1'])}")
    for lab, v in zip(labels, per):
        if lab in REF["f1"]:
            print(f"    {lab:<22} {_delta(v, REF['f1'][lab])}")

    P = np.asarray(probs)
    idx = {c: i for i, c in enumerate(cs.classes_)}
    yi = np.asarray([idx[v] for v in y_true if v in idx])
    if len(yi) == len(P):
        ece = expected_calibration_error(P, yi)
        print(f"\n  ECE on FallAllD: {_delta(ece, REF['ece'])}")
        print("    A model calibrated on one dataset is usually not calibrated")
        print("    on another. Fusion multiplies by this confidence, so a")
        print("    calibration failure matters as much as an accuracy failure.")

    print(f"\n  predicted-state distribution vs true:")
    tc, pc = Counter(y_true), Counter(y_pred)
    for s in labels:
        print(f"    {s:<22} true {tc[s]:>5}   predicted {pc[s]:>5}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-dataset evaluation")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap records read, for a quick pass")
    ap.add_argument("--matched", action="store_true",
                    help="ALSO run with FallAllD forced through UMAFall's "
                         "238->20->50 Hz path, to separate sampling path from "
                         "subject/protocol as the cause of any context drop")
    a = ap.parse_args(argv)

    for mp in ([False, True] if a.matched else [False]):
        if mp:
            print("\n\n" + "#" * 74)
            print("# MATCHED PATH: FallAllD forced through 238 -> 20 -> 50 Hz")
            print("# Same models. If context recovers here, the sampling path")
            print("# was the cause. If not, it is subjects and protocol.")
            print("#" * 74 + "\n")
        eval_motion(a.root, mp, a.limit)
        eval_context(a.root, mp, a.limit)


if __name__ == "__main__":
    main()
