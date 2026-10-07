"""
testing/stage3.py -- Stage 3: tune and finish the motion model (XGBoost).

Data roles, fixed for the whole stage:
    train / validate   UMAFall dev subjects        testing/outputs/split.json
    cross-dataset val  FallAllD 'select' subjects  testing/outputs/fallalld_split.json
    final test         UMAFall test subjects + FallAllD 'report' subjects,
                       opened ONCE, in step 7, by `test`. Nothing before step 7
                       loads them: prep writes the report half to its own file
                       and no other command reads that file.

Every signal goes through streams.motion.harmonise (+/-8 g clip, 20 Hz path).

Steps (run from the repo root):
    py -m testing.stage3 prep        step 0  build harmonised segments, sanity check
    py -m testing.stage3 gate        step 0b re-set the 2.5 g gate for the 20 Hz path
                                          (UMAFall dev subjects only)
    py -m testing.stage3 sweep       step 1  underfit -> overfit sweep (depth x trees)
    py -m testing.stage3 replot      redraw the step 1 figure from step1_sweep.csv
    py -m testing.stage3 fine        step 2  fine sweep around depth 2-3, ~200 trees
    py -m testing.stage3 features    step 3  apply the pick rule to step 2, then compare
                                          the five feature sets with that setting
    py -m testing.stage3 calibrate   steps 4-5  write final_config.json, then calibrate
                                          probabilities and set the alert threshold
                                          from UMAFall dev LOSO predictions only
    py -m testing.stage3 build       step 6  train the final model on all dev subjects
                                          and save models/motion.joblib
    py -m testing.stage3 test        step 7  ONE-TIME test: UMAFall test subjects and the
                                          FallAllD report half (segments + full stream)

Figures worth keeping go to testing/outputs/figures/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_fallalld as fa  # noqa: E402
from testing.compare_motion import MODELS, OUT, collect  # noqa: E402
from streams.motion import harmonise  # noqa: E402

S3 = OUT / "stage3"
FIG = OUT / "figures"          # keepers for slides and the report
UMA_ROOT = r"data\raw\UMAFall"


# --------------------------------------------------------------------------
# shared helpers (later steps reuse these)
# --------------------------------------------------------------------------

def splits():
    return (json.loads((OUT / "split.json").read_text()),
            json.loads((OUT / "fallalld_split.json").read_text()))


def load_dev():
    """UMAFall dev segments (harmonised). Test subjects are filtered out here."""
    meta, X = pd.read_pickle(S3 / "uma.pkl")
    split, _ = splits()
    k = meta.subject.isin(split["dev_subjects"]).values
    return meta[k].reset_index(drop=True), X[k]


def load_fa_select():
    return pd.read_pickle(S3 / "fa_select.pkl")


def gated_auc(meta, p):
    y, g = meta.label.values, meta.gated.values
    if len(set(y[g])) < 2:
        return np.nan
    return roc_auc_score(y[g], p[g])


def loso(make, X, meta):
    """Leave-one-subject-out out-of-fold probabilities."""
    y, g = meta.label.values, meta.subject.values
    p = np.full(len(y), np.nan)
    for s in np.unique(g):
        te = g == s
        p[te] = make().fit(X[~te], y[~te], g[~te]).predict_proba(X[te])
    return p


# --------------------------------------------------------------------------
# step 0
# --------------------------------------------------------------------------

def prep(uma_root, fa_root):
    S3.mkdir(parents=True, exist_ok=True)
    split, fsplit = splits()

    print("UMAFall: building segments with and without harmonise ...")
    m_raw, X_raw, _ = collect(Path(uma_root))
    m_h, X_h, _ = collect(Path(uma_root), transform=harmonise)
    pd.to_pickle((m_h, X_h), S3 / "uma.pkl")

    print("FallAllD: building harmonised segments (238 -> 50 Hz, then the 20 Hz path) ...")
    m_fa, X_fa, _ = collect(Path(fa_root), scanner=fa.scan, transform=harmonise)
    ks = m_fa.subject.isin(fsplit["select"]).values
    kr = m_fa.subject.isin(fsplit["report"]).values
    pd.to_pickle((m_fa[ks].reset_index(drop=True), X_fa[ks]), S3 / "fa_select.pkl")
    pd.to_pickle((m_fa[kr].reset_index(drop=True), X_fa[kr]), S3 / "fa_report_LOCKED.pkl")
    print(f"  select: {ks.sum()} segments   report (locked until step 7): {kr.sum()} segments")

    # sanity: Stage 2 default XGBoost, with vs without the bottleneck
    make = MODELS["xgb"][0]
    dev = lambda m: m.subject.isin(split["dev_subjects"]).values  # noqa: E731
    rows = []
    for tag, m, X in (("without", m_raw, X_raw), ("with", m_h, X_h)):
        d = dev(m)
        md, Xd = m[d].reset_index(drop=True), X[d]
        p = loso(make, Xd, md)
        rows.append(dict(bottleneck=tag, segments=len(md),
                         uma_gate_falls=md.gated[md.label == 1].mean(),
                         uma_gate_adls=md.gated[md.label == 0].mean(),
                         uma_loso_auc_gated=gated_auc(md, p),
                         uma_loso_auc_all=roc_auc_score(md.label, p)))
    md, Xd = m_h[dev(m_h)].reset_index(drop=True), X_h[dev(m_h)]
    model = make().fit(Xd, md.label.values, md.subject.values)
    fs_meta, fs_X = m_fa[ks].reset_index(drop=True), X_fa[ks]
    pf = model.predict_proba(fs_X)
    rows[1].update(fa_select_auc_gated=gated_auc(fs_meta, pf),
                   fa_select_auc_all=roc_auc_score(fs_meta.label, pf),
                   fa_gate_falls=fs_meta.gated[fs_meta.label == 1].mean(),
                   fa_gate_adls=fs_meta.gated[fs_meta.label == 0].mean())

    R = pd.DataFrame(rows).set_index("bottleneck")
    R.round(4).to_csv(S3 / "step0_sanity.csv")
    pd.set_option("display.width", 200)
    print("\nStep 0 sanity check (default XGBoost from Stage 2):")
    print(R.round(3).T.to_string())
    print("\nExpect: UMAFall barely moves (it is already ~20 Hz).")
    print("Stage 2 reference, FallAllD select gated AUC: native 0.607, matched 0.693")
    print(f"\nsaved to {S3}")


# --------------------------------------------------------------------------
# XGBoost with explicit settings (steps 1-6)
# --------------------------------------------------------------------------

BASE = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0)


class XGB:
    def __init__(self, **params):
        self.params = {**BASE, **params}

    def fit(self, X, y, groups=None):
        from xgboost import XGBClassifier
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        self.m = XGBClassifier(**self.params, scale_pos_weight=spw,
                               eval_metric="logloss", random_state=0, n_jobs=-1)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def evaluate(params, dev_meta, dev_X, fa_meta, fa_X):
    """Train fit, UMAFall LOSO and FallAllD-select numbers for one setting."""
    from sklearn.metrics import log_loss
    y = dev_meta.label.values
    full = XGB(**params).fit(dev_X, y)
    p_tr = full.predict_proba(dev_X)
    p_oof = loso(lambda: XGB(**params), dev_X, dev_meta)
    p_fa = full.predict_proba(fa_X)
    return dict(train_auc_gated=gated_auc(dev_meta, p_tr),
                train_logloss=log_loss(y, np.clip(p_tr, 1e-6, 1 - 1e-6)),
                uma_loso_auc_gated=gated_auc(dev_meta, p_oof),
                uma_loso_auc_all=roc_auc_score(y, p_oof),
                fa_select_auc_gated=gated_auc(fa_meta, p_fa),
                fa_select_auc_all=roc_auc_score(fa_meta.label, p_fa))


# --------------------------------------------------------------------------
# step 1: complexity sweep
# --------------------------------------------------------------------------

DEPTHS = [1, 2, 3, 4, 6, 8]
TREES = [25, 50, 100, 200, 400, 800]


def sweep():
    import time
    dm, dX = load_dev()
    fm, fX = load_fa_select()
    rows, t0 = [], time.time()
    total = len(DEPTHS) * len(TREES)
    for i, (d, n) in enumerate([(d, n) for d in DEPTHS for n in TREES], 1):
        r = dict(max_depth=d, n_estimators=n, **evaluate(
            dict(max_depth=d, n_estimators=n), dm, dX, fm, fX))
        rows.append(r)
        print(f"[{i:2d}/{total}] depth {d} trees {n:3d}  train {r['train_auc_gated']:.3f}  "
              f"UMAFall {r['uma_loso_auc_gated']:.3f}  FallAllD-select {r['fa_select_auc_gated']:.3f}"
              f"   ({time.time() - t0:.0f} s)")
    D = pd.DataFrame(rows)
    D.round(4).to_csv(S3 / "step1_sweep.csv", index=False)
    _plot_sweep(D)

    ok = D[D.uma_loso_auc_gated >= 0.98].sort_values("fa_select_auc_gated", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(ok)} of {len(D)} settings keep UMAFall >= 0.98. Top 5 on FallAllD-select:")
    print(ok.head(5).round(3).to_string(index=False))
    print(f"\nsaved {S3 / 'step1_sweep.csv'}\nfigure: {FIG / 'stage3_step1_fitting_curve.png'}")


def replot():
    _plot_sweep(pd.read_csv(S3 / "step1_sweep.csv"))
    print(f"figure: {FIG / 'stage3_step1_fitting_curve.png'}")


def _plot_sweep(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, col, title in zip(ax, ["train_auc_gated", "uma_loso_auc_gated", "fa_select_auc_gated"],
                             ["Training data (UMAFall dev)", "UMAFall, unseen subjects (LOSO)",
                              "FallAllD selection half"]):
        for d in DEPTHS:
            s = D[D.max_depth == d]
            a.plot(s.n_estimators, s[col], marker="o", label=f"depth {d}")
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks(TREES)
        a.set_xticklabels([str(t) for t in TREES])
        a.set_title(title)
        a.set_xlabel("number of trees")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("gated AUC")
    ax[2].legend(fontsize=8)
    fig.suptitle("XGBoost from underfit to overfit: training keeps rising, other data does not")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage3_step1_fitting_curve.png", dpi=150)


# --------------------------------------------------------------------------
# step 2: fine sweep around the step 1 sweet spot
# --------------------------------------------------------------------------

FINE = dict(max_depth=[2, 3], n_estimators=[100, 200, 300],
            learning_rate=[0.03, 0.05, 0.1], min_child_weight=[1, 5],
            reg_lambda=[1, 10])


def fine():
    import itertools
    import time
    dm, dX = load_dev()
    fm, fX = load_fa_select()
    keys = list(FINE)
    grid = list(itertools.product(*FINE.values()))
    rows, t0 = [], time.time()
    for i, vals in enumerate(grid, 1):
        params = dict(zip(keys, vals))
        r = {**params, **evaluate(params, dm, dX, fm, fX)}
        rows.append(r)
        print(f"[{i:2d}/{len(grid)}] " + " ".join(f"{k}={v}" for k, v in params.items())
              + f"  UMAFall {r['uma_loso_auc_gated']:.3f}  FallAllD-select "
              f"{r['fa_select_auc_gated']:.3f}   ({time.time() - t0:.0f} s)")
    D = pd.DataFrame(rows)
    D.round(4).to_csv(S3 / "step2_fine.csv", index=False)
    ok = D[D.uma_loso_auc_gated >= 0.98].sort_values("fa_select_auc_gated", ascending=False)
    pd.set_option("display.width", 220)
    cols = keys + ["uma_loso_auc_gated", "fa_select_auc_gated", "fa_select_auc_all"]
    print(f"\n{len(ok)} of {len(D)} settings keep UMAFall >= 0.98. Top 10 on FallAllD-select:")
    print(ok[cols].head(10).round(3).to_string(index=False))
    print(f"\nsaved {S3 / 'step2_fine.csv'}")


# --------------------------------------------------------------------------
# selection rule (agreed before step 1)
# --------------------------------------------------------------------------

UMA_FLOOR = 0.98
TIE = 0.005
PARAM_KEYS = ["max_depth", "n_estimators", "learning_rate", "min_child_weight", "reg_lambda"]


def pick(D, score="fa_select_auc_gated"):
    """Highest FallAllD-select gated AUC with UMAFall LOSO >= 0.98. Anything
    within 0.005 of the best counts as a tie; among ties take the simplest
    (shallowest, then fewest trees, then the best score)."""
    ok = D[D.uma_loso_auc_gated >= UMA_FLOOR]
    best = ok[score].max()
    ties = ok[ok[score] >= best - TIE]
    sort = [c for c in ("max_depth", "n_estimators") if c in ties]
    return ties.sort_values(sort + [score], ascending=[True] * len(sort) + [False]), best


# --------------------------------------------------------------------------
# step 3: feature sets at the step 2 setting
# --------------------------------------------------------------------------

def features():
    from streams.motion import FEATURE_NAMES, FEATURE_SETS
    D = pd.read_csv(S3 / "step2_fine.csv")
    ties, best = pick(D)
    chosen = ties.iloc[0]
    params = {k: (int(chosen[k]) if k in ("max_depth", "n_estimators") else float(chosen[k]))
              for k in PARAM_KEYS}
    print(f"step 2 best FallAllD-select gated AUC: {best:.4f}; "
          f"{len(ties)} settings within {TIE} (ties):")
    print(ties[PARAM_KEYS + ["uma_loso_auc_gated", "fa_select_auc_gated"]]
          .round(4).to_string(index=False))
    print(f"\nrule picks the simplest tie: {params}")
    (S3 / "step2_pick.json").write_text(json.dumps(params, indent=2))

    dm, dX = load_dev()
    fm, fX = load_fa_select()
    rows = []
    for name, idx in FEATURE_SETS.items():
        r = dict(feature_set=name, n_features=len(idx),
                 **evaluate(params, dm, dX[:, idx], fm, fX[:, idx]))
        rows.append(r)
        print(f"{name:12s} {len(idx):2d} features  UMAFall {r['uma_loso_auc_gated']:.3f}  "
              f"FallAllD-select {r['fa_select_auc_gated']:.3f}")
    F = pd.DataFrame(rows)
    F.round(4).to_csv(S3 / "step3_features.csv", index=False)
    dropped = {n: [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in idx]
               for n, idx in FEATURE_SETS.items()}
    print("\nfeatures removed in each set:")
    for n, d in dropped.items():
        print(f"  {n:12s} {', '.join(d) if d else '(none)'}")
    _plot_features(F)
    print(f"\nsaved {S3 / 'step3_features.csv'}, {S3 / 'step2_pick.json'}")
    print(f"figure: {FIG / 'stage3_step3_feature_sets.png'}")


def _plot_features(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(F))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, F.uma_loso_auc_gated, 0.4, label="UMAFall, unseen subjects (LOSO)")
    ax.bar(x + 0.2, F.fa_select_auc_gated, 0.4, label="FallAllD selection half")
    for i, (u, f) in enumerate(zip(F.uma_loso_auc_gated, F.fa_select_auc_gated)):
        ax.text(i - 0.2, u + 0.005, f"{u:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({k})" for n, k in zip(F.feature_set, F.n_features)])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("gated AUC")
    ax.set_title("Which features carry over to a new dataset")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage3_step3_feature_sets.png", dpi=150)


# --------------------------------------------------------------------------
# steps 4-5: final configuration, calibration, threshold
# --------------------------------------------------------------------------

TARGET_SPEC = 0.95


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def _operating(meta, p, thr):
    """Numbers for one threshold. 'System' counts a fall only if the gate fired
    AND the model scored >= thr; ADL hours come from all dev ADL recordings."""
    y, g = meta.label.values, meta.gated.values
    hit = g & (p >= thr)
    adl_h = meta.loc[meta.label == 0, "seconds"].sum() / 3600
    gy, gp = y[g], p[g]
    return dict(threshold=thr,
                gated_sensitivity=(gp[gy == 1] >= thr).mean(),
                gated_specificity=(gp[gy == 0] < thr).mean(),
                system_sensitivity=hit[y == 1].mean(),
                system_false_alarms=int(hit[y == 0].sum()),
                false_alarms_per_hour=hit[y == 0].sum() / adl_h)


def calibrate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from streams.motion import FEATURE_SETS, GATE_SVM_G, RATE_HZ, ACC_CLIP_G

    # ---- step 4: final configuration ----
    params = json.loads((S3 / "step2_pick.json").read_text())
    F = pd.read_csv(S3 / "step3_features.csv")
    ok = F[F.uma_loso_auc_gated >= UMA_FLOOR]
    best = ok.fa_select_auc_gated.max()
    ties = ok[ok.fa_select_auc_gated >= best - TIE].sort_values(
        ["n_features", "fa_select_auc_gated"], ascending=[True, False])
    fset = ties.iloc[0].feature_set
    print("STEP 4 - final configuration")
    print(f"  XGBoost {params}")
    print(f"  feature set: {fset} ({len(FEATURE_SETS[fset])} features); "
          f"ties within {TIE}: {list(ties.feature_set)}")
    print(f"  signal path: clip +/-{ACC_CLIP_G} g, {RATE_HZ} Hz; gate {GATE_SVM_G} g")

    # ---- step 5: calibration and threshold, UMAFall dev LOSO only ----
    dm, dX = load_dev()
    idx = FEATURE_SETS[fset]
    dX = dX[:, idx]
    y, g = dm.label.values, dm.gated.values
    raw = loso(lambda: XGB(**params), dX, dm)

    if len(set(y[g])) < 2:
        sys.exit("gated segments contain only one class; cannot calibrate on them")
    platt = LogisticRegression(C=1e6).fit(_logit(raw[g]).reshape(-1, 1), y[g])
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    cal = 1 / (1 + np.exp(-(a * _logit(raw) + b)))

    print("\nSTEP 5 - calibration (Platt scaling on gated LOSO predictions)")
    print(f"  a = {a:.4f}, b = {b:.4f}")
    for name, p in (("raw", raw), ("calibrated", cal)):
        print(f"  {name:10s} Brier {brier_score_loss(y[g], p[g]):.4f}   "
              f"ECE {_ece(y[g], p[g]):.4f}   (gated segments)")

    gp, gy = cal[g], y[g]
    adl_scores = np.sort(gp[gy == 0])
    k = int(np.ceil(TARGET_SPEC * len(adl_scores)))
    thr = float(np.nextafter(adl_scores[min(k, len(adl_scores)) - 1], 1.0))
    rows = [_operating(dm, cal, 0.5), _operating(dm, cal, thr)]
    O = pd.DataFrame(rows, index=["at 0.5", f"at {TARGET_SPEC:.0%} spec"])

    fm, fX = load_fa_select()
    model = XGB(**params).fit(dX, y)
    fa_cal = 1 / (1 + np.exp(-(a * _logit(model.predict_proba(fX[:, idx])) + b)))
    O_fa = pd.DataFrame([_operating(fm, fa_cal, 0.5), _operating(fm, fa_cal, thr)],
                        index=["at 0.5", f"at {TARGET_SPEC:.0%} spec"])

    pd.set_option("display.width", 200)
    print(f"\n  threshold for {TARGET_SPEC:.0%} specificity on gated ADLs: {thr:.4f}")
    print("\n  UMAFall dev, LOSO (used to set the threshold):")
    print(O.round(3).to_string())
    print("\n  FallAllD select half (info only, NOT used to set anything):")
    print(O_fa.round(3).to_string())

    cfg = dict(model="xgboost", params=params, feature_set=fset, feature_index=list(idx),
               rate_hz=RATE_HZ, acc_clip_g=ACC_CLIP_G, gate_svm_g=GATE_SVM_G,
               platt=dict(a=a, b=b), threshold=thr, threshold_rule=f"{TARGET_SPEC:.0%} "
               "specificity on gated ADL segments, UMAFall dev LOSO, calibrated scores")
    (S3 / "final_config.json").write_text(json.dumps(cfg, indent=2))
    pd.concat({"umafall_dev_loso": O, "fallalld_select_info": O_fa}).round(4).to_csv(
        S3 / "step5_operating_points.csv")
    _plot_calibration(y[g], raw[g], cal[g])
    print(f"\nsaved {S3 / 'final_config.json'}, {S3 / 'step5_operating_points.csv'}")
    print(f"figure: {FIG / 'stage3_step5_calibration.png'}")


def _plot_calibration(y, raw, cal, bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.linspace(0, 1, bins + 1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab in ((raw, "before"), (cal, "after (Platt)")):
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xlabel("predicted probability of a fall")
    ax.set_ylabel("actual share of falls")
    ax.set_title("Calibration, UMAFall dev (LOSO, gated segments)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage3_step5_calibration.png", dpi=150)


# --------------------------------------------------------------------------
# step 6: build the production model
# --------------------------------------------------------------------------

def build(out="models/motion.joblib"):
    import shutil
    import time
    from streams.motion import MotionStream, GATE_GYRO_DPS

    cfg = json.loads((S3 / "final_config.json").read_text())
    params, idx = cfg["params"], cfg["feature_index"]
    dm, dX = load_dev()
    y, g = dm.label.values, dm.gated.values
    Xs = dX[:, idx]

    # Keep Platt only if it actually improves calibration (rule fixed here,
    # applied to the same UMAFall dev LOSO predictions as step 5).
    raw = loso(lambda: XGB(**params), Xs, dm)
    a, b = cfg["platt"]["a"], cfg["platt"]["b"]
    cal = 1 / (1 + np.exp(-(a * _logit(raw) + b)))
    ece_raw, ece_cal = _ece(y[g], raw[g]), _ece(y[g], cal[g])
    use_platt = ece_cal < ece_raw
    scores = cal if use_platt else raw
    platt = (a, b) if use_platt else (1.0, 0.0)

    adl = np.sort(scores[g & (y == 0)])
    k = int(np.ceil(TARGET_SPEC * len(adl)))
    thr = float(np.nextafter(adl[min(k, len(adl)) - 1], 1.0))
    op = _operating(dm, scores, thr)
    print(f"gated ECE: raw {ece_raw:.4f}, Platt {ece_cal:.4f} -> "
          f"{'keep Platt' if use_platt else 'drop Platt (no improvement), use raw scores'}")
    print(f"threshold ({TARGET_SPEC:.0%} spec on gated ADLs, UMAFall dev LOSO): {thr:.4f}")
    print(f"  at that threshold: system sensitivity {op['system_sensitivity']:.3f}, "
          f"false alarms/hour {op['false_alarms_per_hour']:.2f}")

    final = XGB(**params).fit(Xs, y)
    final.m.set_params(n_jobs=1)
    ms = MotionStream(gate_svm=cfg["gate_svm_g"], gate_gyro=GATE_GYRO_DPS)
    ms.model, ms.classes_ = final.m, [0, 1]
    ms.platt, ms.threshold, ms.feature_index = platt, thr, idx
    ms.config = {**cfg, "platt": dict(a=platt[0], b=platt[1]), "platt_used": use_platt,
                 "threshold": thr, "trained_on": "UMAFall dev subjects",
                 "n_segments": int(len(y)), "n_falls": int(y.sum())}

    out = Path(out)
    if out.exists():
        backup = out.with_name("motion_rf_before_stage3.joblib")
        if not backup.exists():
            shutil.copy(out, backup)
            print(f"old model kept as {backup}")
    ms.save(out)

    ms2 = MotionStream.load(out)
    p_check = ms2.impact_probability(dX)
    p_ref = final.predict_proba(Xs) if not use_platt else \
        1 / (1 + np.exp(-(a * _logit(final.predict_proba(Xs)) + b)))
    assert np.allclose(p_check, p_ref, atol=1e-6), "saved model does not reproduce"
    t0 = time.perf_counter()
    for _ in range(100):
        ms2.impact_probability(dX[:1])
    ms_per = (time.perf_counter() - t0) / 100 * 1000

    cfg_out = ms.config
    (S3 / "final_config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"\nsaved {out}  ({len(y)} segments, {int(y.sum())} falls, "
          f"gate {cfg['gate_svm_g']} g, {len(idx)} features)")
    print(f"reload check passed; {ms_per:.2f} ms per impact score on this machine")
    print(f"final_config.json updated (platt_used={use_platt}, threshold={thr:.4f})")
    print("\nnext: py -m streams.motion --selftest --model models/motion.joblib")


# --------------------------------------------------------------------------
# step 7: the one-time test
# --------------------------------------------------------------------------

def _seg_metrics(meta, p, thr):
    y, g = meta.label.values, meta.gated.values
    gate_score = meta.svm_max.values
    out = dict(falls=int(y.sum()), adls=int((y == 0).sum()),
               gate_falls=g[y == 1].mean(), gate_adls=g[y == 0].mean(),
               auc_gated=gated_auc(meta, p),
               auc_gated_gate_only=gated_auc(meta, gate_score))
    for name, t in (("0.5", 0.5), ("thr", thr)):
        op = _operating(meta, p, t)
        out[f"falls_caught_at_{name}"] = op["system_sensitivity"]
        out[f"false_alarms_per_hour_at_{name}"] = op["false_alarms_per_hour"]
    return out


def _stream_scores(records, model_path):
    """Replay each recording through the live MotionStream, window by window.
    Per recording: max stream score (impact + stillness) and max impact."""
    from streams.motion import MotionStream
    rows = []
    for rec, windows in records:
        ms = MotionStream.load(model_path)
        best, best_imp = 0.0, 0.0
        for w in windows:
            best = max(best, ms.score(w))
            best_imp = max(best_imp, ms.last_impact if ms.gate_open else 0.0)
        rows.append(dict(subject=rec.subject, label=int(rec.is_fall), file=rec.path.name,
                         stream=best, impact_only=best_imp, seconds=rec.duration))
    return pd.DataFrame(rows)


def _stream_metrics(D):
    from sklearn.metrics import roc_curve
    out = dict(recordings=len(D), fall_recordings=int(D.label.sum()))
    for col in ("impact_only", "stream"):
        out[f"auc_{col}"] = roc_auc_score(D.label, D[col])
        fpr, tpr, _ = roc_curve(D.label, D[col])
        k = fpr <= 0.05
        out[f"sens_at_95spec_{col}"] = float(tpr[k].max()) if k.any() else 0.0
    return out


def test(uma_root, fa_root, model_path="models/motion.joblib"):
    from data import loader as uma_loader
    from streams.motion import MotionStream, FEATURE_NAMES
    flag = S3 / "step7_DONE.txt"
    if flag.exists():
        sys.exit(f"step 7 already ran ({flag.read_text().strip()}). "
                 "Re-running after any change would turn the test set into a tuning set.")
    split, fsplit = splits()
    ms = MotionStream.load(model_path)
    thr = float(ms.threshold)
    i_svm = FEATURE_NAMES.index("svm_max")

    um, uX = pd.read_pickle(S3 / "uma.pkl")
    kt = um.subject.isin(split["test_subjects"]).values
    um_t, uX_t = um[kt].reset_index(drop=True).assign(svm_max=uX[kt][:, i_svm]), uX[kt]
    fr, fX = pd.read_pickle(S3 / "fa_report_LOCKED.pkl")
    fr = fr.assign(svm_max=fX[:, i_svm])

    rows = []
    for name, m, X in (("UMAFall test (S02 S06 S11)", um_t, uX_t),
                       ("FallAllD report half", fr, fX)):
        p = ms.impact_probability(X)
        rows.append(dict(set=name, **_seg_metrics(m, p, thr)))
        if name.startswith("UMAFall"):
            hit = m.gated.values & (p >= thr)
            by_dir = (m.assign(hit=hit)[m.label == 1].groupby("direction").hit.mean())
        _roc_data = (m, p) if name.startswith("UMAFall") else _roc_data + (m, p)
    S = pd.DataFrame(rows).set_index("set")

    print("Replaying recordings through the full stream (impact + stillness) ...")
    test_set = set(split["test_subjects"])
    uma_recs = [(r, uma_loader.iter_windows(r)) for r in uma_loader.scan(Path(uma_root))
                if r.subject in test_set]
    uma_recs = [(r, [w for w, _ in ws]) for r, ws in uma_recs]
    fa_files = sorted(Path(fa_root).glob("S*_D2_A*_T*_A.dat"))
    fa_recs = []
    for f in fa_files:
        meta = fa.parse_name(f.name)
        if meta and meta["subject"] in fsplit["report"]:
            try:
                r = fa.read_record(f)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {f.name}: {e}", file=sys.stderr)
                continue
            fa_recs.append((r, [w for w, _ in fa.iter_windows(r)]))
    St = pd.DataFrame([dict(set="UMAFall test (S02 S06 S11)", **_stream_metrics(_stream_scores(uma_recs, model_path))),
                       dict(set="FallAllD report half", **_stream_metrics(_stream_scores(fa_recs, model_path)))]).set_index("set")

    pd.set_option("display.width", 220)
    print(f"\nmodel: {model_path}   threshold {thr:.4f} (set on UMAFall dev)")
    print("\nA. Impact segments (what the XGBoost model scores)")
    print(S.round(3).T.to_string())
    print("\n   UMAFall test, falls caught at the threshold, by direction:")
    print(by_dir.round(3).to_string())
    print("\nB. Full stream, per recording (max score while replaying the recording)")
    print(St.round(3).T.to_string())

    S.round(4).to_csv(S3 / "step7_segments.csv")
    St.round(4).to_csv(S3 / "step7_stream.csv")
    by_dir.round(4).to_csv(S3 / "step7_umafall_by_direction.csv")
    _plot_roc(*_roc_data)
    import datetime
    flag.write_text(f"run {datetime.datetime.now():%Y-%m-%d %H:%M} with {model_path}\n")
    print(f"\nsaved step7_segments.csv, step7_stream.csv, step7_umafall_by_direction.csv")
    print(f"figure: {FIG / 'stage3_step7_roc.png'}")
    print("The test set is now spent. Any further change must be judged on the dev/select data.")


def _plot_roc(um, pu, fr, pf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for a, (m, p, title) in zip(ax, ((um, pu, "UMAFall test subjects"),
                                     (fr, pf, "FallAllD report half"))):
        g = m.gated.values
        y = m.label.values[g]
        for score, lab in ((p[g], "XGBoost"), (m.svm_max.values[g], "gate only (peak |a|)")):
            fpr, tpr, _ = roc_curve(y, score)
            a.plot(fpr, tpr, label=f"{lab}  AUC {roc_auc_score(y, score):.3f}")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_title(f"{title} (gated segments)")
        a.set_xlabel("false positive rate")
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("true positive rate")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage3_step7_roc.png", dpi=150)


# --------------------------------------------------------------------------
# step 0b: the gate after the bottleneck
# --------------------------------------------------------------------------

def gate(uma_root, target=None):
    """The 20 Hz filter rounds off impact peaks, so a fixed 2.5 g gate now lets
    fewer falls through. Pick the highest threshold that keeps the fall coverage
    the gate had before the bottleneck. UMAFall dev subjects only."""
    from data.loader import scan
    # Fixed reference: the 2.5 g design gate on the unfiltered signal. Do NOT
    # read the current GATE_SVM_G here -- once it is lowered, the "before"
    # coverage rises and every rerun would recommend a lower gate again.
    REF_GATE = 2.5
    split, _ = splits()
    rows = []
    for rec in scan(Path(uma_root)):
        if rec.subject not in split["dev_subjects"]:
            continue
        raw = float(np.linalg.norm(np.clip(rec.acc, -8, 8), axis=1).max())
        acc_h, _ = harmonise(rec.acc, None)
        rows.append(dict(fall=rec.is_fall, peak_before=raw,
                         peak_after=float(np.linalg.norm(acc_h, axis=1).max())))
    d = pd.DataFrame(rows)
    f, a = d[d.fall], d[~d.fall]
    before = float((f.peak_before > REF_GATE).mean())
    target = before if target is None else target
    print(f"{len(f)} falls, {len(a)} ADLs (UMAFall dev subjects)")
    print(f"reference: {REF_GATE} g gate before the bottleneck catches falls {before:.3f}, "
          f"ADLs {(a.peak_before > REF_GATE).mean():.3f}")
    print(f"median fall peak: before {f.peak_before.median():.2f} g, "
          f"after {f.peak_after.median():.2f} g\n")
    tab = []
    for thr in np.round(np.arange(1.6, 2.61, 0.05), 2):
        tab.append(dict(threshold_g=thr, falls_gated=(f.peak_after > thr).mean(),
                        adls_gated=(a.peak_after > thr).mean()))
    T = pd.DataFrame(tab)
    ok = T[T.falls_gated >= target - 1e-9]
    pick = float(ok.threshold_g.max()) if len(ok) else float(T.threshold_g.min())
    print("after the bottleneck:")
    print(T.round(3).to_string(index=False))
    print(f"\nrule: highest threshold that still gates >= {target:.3f} of falls "
          f"-> {pick:.2f} g")
    S3.mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(S3 / "step0b_gate.csv", index=False)
    (S3 / "step0b_gate_pick.txt").write_text(f"{pick:.2f}\n")
    print(f"\nset GATE_SVM_G = {pick:.2f} in streams/motion.py, then rerun: py -m testing.stage3 prep")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "gate", "sweep", "replot", "fine", "features", "calibrate", "build", "test"])
    ap.add_argument("--uma-root", default=UMA_ROOT)
    ap.add_argument("--fa-root", default=str(fa.DEFAULT_ROOT))
    a = ap.parse_args(argv)
    if a.step == "prep":
        prep(a.uma_root, a.fa_root)
    elif a.step == "gate":
        gate(a.uma_root)
    elif a.step == "sweep":
        sweep()
    elif a.step == "replot":
        replot()
    elif a.step == "fine":
        fine()
    elif a.step == "features":
        features()
    elif a.step == "calibrate":
        calibrate()
    elif a.step == "build":
        build()
    elif a.step == "test":
        test(a.uma_root, a.fa_root)


if __name__ == "__main__":
    main()
