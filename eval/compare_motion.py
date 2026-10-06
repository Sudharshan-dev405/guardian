"""
eval/compare_motion.py -- Stage 2: pick the motion model.

Uses the SAME segments the production stream trains on (one per record,
2.0 s before to 1.5 s after the first gate crossing, or the SVM peak for
ungated records), so the winner can drop straight into streams/motion.py.

  1. Locks 3 test subjects in eval/outputs/split.json (fall subjects only,
     never the largest one). They are NOT touched here -- that is Stage 3.
  2. Leave-one-subject-out over every other subject. Subjects with no falls
     stay in, as negatives; they just cannot give a fold AUC.
  3. Six models on identical folds: gate-only threshold, RandomForest,
     XGBoost, SVM, 1D-CNN, CNN-LSTM.

Commands (repo root):
    py -m eval.compare_motion                       # all six
    py -m eval.compare_motion --models threshold rf xgb svm
    py -m eval.compare_motion --test S05 S11 S16    # pick test subjects yourself
    py -m eval.compare_motion --resplit             # rebuild split.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import scan  # noqa: E402
from streams.motion import (  # noqa: E402
    FEATURE_NAMES, GATE_GYRO_DPS, GATE_SVM_G, POST_SEC, PRE_SEC, segment_features)

OUT = Path(__file__).resolve().parent / "outputs"
SEED = 0
FS = 50.0
N_TEST = 3


# --------------------------------------------------------------------------
# Segments: same index logic as streams.motion.segments_from_record, but also
# keeps the raw segment (for the deep models) and the record's metadata.
# --------------------------------------------------------------------------

def direction_of(activity: str) -> str:
    a = activity.lower()
    for d in ("forward", "backward", "lateral"):
        if d in a:
            return d
    return ""


def collect(root, scanner=None, transform=None, **scan_kw):
    """scanner defaults to the UMAFall loader; pass data.loader_fallalld.scan
    for FallAllD (both yield the same Record shape at 50 Hz). transform, if
    given, maps (acc, gyro) -> (acc, gyro) before gating and features
    (Stage 3 passes streams.motion.harmonise)."""
    rows, X, raw = [], [], []
    pre_n, post_n = int(PRE_SEC * FS), int(POST_SEC * FS)
    for rec in (scanner or scan)(root, **scan_kw):
        acc = np.asarray(rec.acc, dtype=float)
        gyro = np.asarray(rec.gyro, dtype=float) if rec.gyro is not None else None
        if transform is not None:
            acc, gyro = transform(acc, gyro)
        n = len(acc)
        if n < pre_n + post_n + 2:
            continue
        svm = np.linalg.norm(acc, axis=1)
        hit = svm > GATE_SVM_G
        if gyro is not None:
            hit = hit | (np.linalg.norm(gyro, axis=1) > GATE_GYRO_DPS)
        idx = int(np.flatnonzero(hit)[0]) if hit.any() else int(np.argmax(svm))
        i = int(np.clip(idx, pre_n, n - post_n - 1))
        sa = acc[i - pre_n:i + post_n + 1]
        sg = gyro[i - pre_n:i + post_n + 1] if gyro is not None else np.zeros_like(sa)
        X.append(segment_features(sa, None if gyro is None else sg, FS, pre_n))
        raw.append(np.column_stack([sa, sg, np.linalg.norm(sa, axis=1)]))
        rows.append(dict(subject=rec.subject, label=int(rec.is_fall),
                         activity=rec.activity, direction=direction_of(rec.activity),
                         gated=bool(hit.any()), seconds=rec.duration,
                         file=rec.path.name))
    meta = pd.DataFrame(rows)
    return meta, np.asarray(X), np.asarray(raw, dtype=np.float32)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def make_split(meta, forced=None):
    per = meta.groupby("subject").agg(
        falls=("label", "sum"),
        forward=("direction", lambda s: (s == "forward").sum()),
        backward=("direction", lambda s: (s == "backward").sum()),
        lateral=("direction", lambda s: (s == "lateral").sum()))
    fall_subj = per[per.falls > 0]
    if forced:
        bad = [s for s in forced if s not in fall_subj.index]
        if bad:
            sys.exit(f"cannot be test subjects (missing or no falls): {bad}")
        test = sorted(forced)
    else:
        # never the subject holding the most falls; cover all three directions;
        # stay close to a typical subject's fall count
        pool = fall_subj.drop(index=fall_subj.falls.idxmax())
        med = pool.falls.median()
        best = max(itertools.combinations(pool.index, N_TEST),
                   key=lambda c: (min(pool.loc[list(c), d].sum()
                                      for d in ("forward", "backward", "lateral")),
                                  -abs(pool.loc[list(c), "falls"].mean() - med),
                                  tuple(c)))
        test = sorted(best)
    dev = sorted(s for s in per.index if s not in test)
    return dict(test_subjects=test, dev_subjects=dev,
                dev_without_falls=sorted(s for s in dev if per.loc[s, "falls"] == 0),
                seed=SEED, note="Test subjects are opened only in Stage 3.")


# --------------------------------------------------------------------------
# Models: fit(X, y, groups) / predict_proba(X) -> p(fall)
# --------------------------------------------------------------------------

class Threshold:
    """Stage-1 gate on its own: score = peak SVM. Threshold that maximises F1
    on the training subjects (reported, not used for AUC)."""
    feature = FEATURE_NAMES.index("svm_max")

    def fit(self, X, y, groups=None):
        s = X[:, self.feature]
        cands = np.unique(np.round(s, 2))
        f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cands]
        self.thr = float(cands[int(np.argmax(f1s))])
        return self

    def predict_proba(self, X):
        # monotone map so that svm_max == thr gives exactly 0.5
        return 1.0 / (1.0 + np.exp(-(X[:, self.feature] - self.thr) * 4.0))


class Sk:
    def __init__(self, make):
        self.make = make

    def fit(self, X, y, groups=None):
        self.m = self.make(y).fit(X, y)
        if "n_jobs" in self.m.get_params():
            self.m.set_params(n_jobs=1)  # one segment at a time, like the live stream
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, list(self.m.classes_).index(1)]


def rf(y):
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def xgb(y):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)


def svm(y):
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=SEED), method="sigmoid", cv=3))


class Deep:
    """Small nets on the raw segment: ax ay az gx gy gz |a|, 176 x 7.
    Channels z-scored with TRAINING-fold statistics (per-segment z-scoring
    would erase the impact size, the main cue)."""

    def __init__(self, kind, epochs=60):
        self.kind, self.epochs = kind, epochs

    def _build(self, shape):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        tf.keras.utils.set_random_seed(SEED)
        inp = layers.Input(shape)
        x = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        if self.kind == "cnn":
            x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.LSTM(32)(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(layers.Dense(32, activation="relu")(x))
        m = models.Model(inp, out)
        m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
        return m

    def fit(self, R, y, groups):
        import tensorflow as tf
        self.mu = R.mean((0, 1), keepdims=True)
        self.sd = R.std((0, 1), keepdims=True) + 1e-6
        Rn = (R - self.mu) / self.sd
        # early stopping on held-out TRAINING subjects, never the fold's test subject
        tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                      .split(Rn, y, groups))
        if y[va].sum() == 0 or y[va].sum() == len(va):
            tr = va = np.arange(len(y))
        pos = max(int(y[tr].sum()), 1)
        self.m = self._build(R.shape[1:])
        self.m.fit(Rn[tr], y[tr], validation_data=(Rn[va], y[va]),
                   epochs=self.epochs, batch_size=32, verbose=0,
                   class_weight={0: 1.0, 1: (len(tr) - pos) / pos},
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       "val_auc", mode="max", patience=8, restore_best_weights=True)])
        return self

    def predict_proba(self, R):
        return self.m.predict((R - self.mu) / self.sd, verbose=0).ravel()


MODELS = {
    "threshold": (lambda: Threshold(), "features"),
    "rf":        (lambda: Sk(rf), "features"),
    "xgb":       (lambda: Sk(xgb), "features"),
    "svm":       (lambda: Sk(svm), "features"),
    "cnn":       (lambda: Deep("cnn"), "raw"),
    "cnn_lstm":  (lambda: Deep("cnn_lstm"), "raw"),
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def sens_at_spec(y, p, spec=0.95):
    fpr, tpr, _ = roc_curve(y, p)
    k = fpr <= 1 - spec
    return float(tpr[k].max()) if k.any() else 0.0


def summarise(name, d, ms):
    y, p = d.label.values, d.prob.values
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    g = d.gated.values
    adl_hours = d.loc[d.label == 0, "seconds"].sum() / 3600
    return dict(
        model=name,
        auc_all=roc_auc_score(y, p),
        auc_gated=roc_auc_score(y[g], p[g]) if len(set(y[g])) == 2 else np.nan,
        sens_at_95spec=sens_at_spec(y, p),
        sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
        f1=f1_score(y, pred, zero_division=0),
        false_alarms_per_hour=fp / adl_hours if adl_hours else np.nan,
        ms_per_segment=ms)


def latency_ms(model, Z):
    one = Z[:1]
    if isinstance(model, Deep):
        f = lambda: model.m((one - model.mu) / model.sd, training=False)  # noqa: E731
    else:
        f = lambda: model.predict_proba(one)  # noqa: E731
    f()
    t = time.perf_counter()
    for _ in range(20):
        f()
    return (time.perf_counter() - t) / 20 * 1000


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data\raw\UMAFall")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--test", nargs="*", help="test subjects, e.g. S05 S11 S16")
    ap.add_argument("--resplit", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    meta, X, R = collect(Path(a.root))
    print(f"{len(meta)} segments from {meta.subject.nunique()} subjects  "
          f"falls={meta.label.sum()}  ADLs={(meta.label == 0).sum()}")

    split_path = OUT / "split.json"
    if split_path.exists() and not (a.resplit or a.test):
        split = json.loads(split_path.read_text())
    else:
        split = make_split(meta, a.test)
        split_path.write_text(json.dumps(split, indent=2))
    print(f"test subjects (held out, untouched): {split['test_subjects']}")
    print(f"dev subjects: {split['dev_subjects']}  "
          f"(no falls, negatives only: {split['dev_without_falls']})\n")

    dev = meta.subject.isin(split["dev_subjects"]).values
    assert not set(split["test_subjects"]) & set(meta.subject[dev])
    md, Xd, Rd = meta[dev].reset_index(drop=True), X[dev], R[dev]
    y, g = md.label.values, md.subject.values

    summary, folds, oof_all = [], [], []
    for name in a.models:
        make, kind = MODELS[name]
        Z = Xd if kind == "features" else Rd
        prob, lat = np.full(len(y), np.nan), []
        t0 = time.time()
        for s in split["dev_subjects"]:
            te, tr = g == s, g != s
            m = make().fit(Z[tr], y[tr], g[tr])
            prob[te] = m.predict_proba(Z[te])
            lat.append(latency_ms(m, Z[te]))
            if len(set(y[te])) == 2:
                folds.append(dict(model=name, subject=s,
                                  auc=roc_auc_score(y[te], prob[te]),
                                  falls=int(y[te].sum()), adls=int((y[te] == 0).sum())))
        d = md.assign(prob=prob, model=name)
        oof_all.append(d)
        row = summarise(name, d, float(np.median(lat)))
        fa = pd.DataFrame(folds)
        fa = fa[fa.model == name].auc
        row.update(fold_auc_mean=fa.mean(), fold_auc_sd=fa.std(), minutes=(time.time() - t0) / 60)
        summary.append(row)
        print(f"{name:10s} AUC {row['auc_all']:.3f} (gated {row['auc_gated']:.3f})  "
              f"Se@95Sp {row['sens_at_95spec']:.3f}  FA/h {row['false_alarms_per_hour']:.1f}  "
              f"[{row['minutes']:.1f} min]")

    S = pd.DataFrame(summary).set_index("model")
    # rank on gate-conditional AUC (the deployment number); fall back to all-records
    # AUC only if no ADL ever crossed the gate
    S["rank_auc"] = S.auc_gated.fillna(S.auc_all)
    S = S.sort_values("rank_auc", ascending=False)
    oof = pd.concat(oof_all, ignore_index=True)
    by_dir = (oof[oof.label == 1].assign(hit=lambda d: d.prob >= 0.5)
              .groupby(["model", "direction"]).hit.mean().unstack())
    S.round(4).to_csv(OUT / "compare_summary.csv")
    pd.DataFrame(folds).to_csv(OUT / "compare_per_fold.csv", index=False)
    by_dir.round(4).to_csv(OUT / "compare_by_direction.csv")
    oof.to_csv(OUT / "compare_oof_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nPooled out-of-fold results, dev subjects only (sorted by gate-conditional AUC):")
    print(S[["auc_all", "auc_gated", "fold_auc_mean", "fold_auc_sd", "sens_at_95spec",
             "sensitivity", "specificity", "f1", "false_alarms_per_hour",
             "ms_per_segment"]].round(3).to_string())
    print("\nFalls caught at p >= 0.5, by direction:")
    print(by_dir.round(3).to_string())

    top = S.rank_auc.max()
    close = S[S.rank_auc >= top - 0.01]
    pick = close.ms_per_segment.idxmin()
    print(f"\nwithin 0.01 of the best gate-conditional AUC: {list(close.index)}")
    print(f"pick (fastest of those): {pick}")
    if "threshold" in S.index and pick != "threshold":
        print(f"margin over the gate-only baseline: "
              f"{S.loc[pick, 'rank_auc'] - S.loc['threshold', 'rank_auc']:+.3f} AUC")
    (OUT / "compare_pick.txt").write_text(pick + "\n")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
