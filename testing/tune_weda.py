"""
testing/tune_weda.py -- Stage 3.5b: full hyperparameter tuning of the WEDA-trained
XGBoost (the Stage 3.5 winner). Dev parts only; the test parts are spent.

Scored on every setting:
    train      WEDA-FALL dev, fitted and scored on itself (to show overfitting)
    weda       WEDA-FALL dev, leave-one-person-out (new people)
    uma, fa    UMAFall dev and FallAllD dev (datasets never seen in training)
    unseen     mean of uma and fa

Rule (fixed before running, also in the log):
    - WEDA LOSO must stay >= 0.893 (current 0.903 minus 0.01)
    - best 'unseen'; within 0.005 of the best = tie -> shallowest, fewest trees
    - switch from the current setting only if 'unseen' improves by >= 0.01;
      otherwise keep the current model

Steps (repo root):
    py -m testing.tune_weda sweep    depth x trees, underfit -> overfit (42 settings) + figure
    py -m testing.tune_weda fine     216 settings around the sweep pick (depth, trees,
                                  learning rate, min_child_weight, lambda, subsampling)
    py -m testing.tune_weda build    only if 'fine' says SWITCH: save the new model
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testing.stage35 import S35, FIG, _dev_all  # noqa: E402

CURRENT = dict(max_depth=4, n_estimators=100, learning_rate=0.1, min_child_weight=1.0,
               reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8)
WEDA_FLOOR = 0.893
TIE = 0.005
MIN_GAIN = 0.01

DEPTHS = [1, 2, 3, 4, 5, 6, 8]
TREES = [25, 50, 100, 200, 400, 800]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _fit(params, X, y):
    from xgboost import XGBClassifier
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    p = {**CURRENT, **params}
    return XGBClassifier(**p, gamma=0.0, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=0, n_jobs=1).fit(X, y)


def _gauc(meta, p):
    from sklearn.metrics import roc_auc_score
    y, g = meta.label.values, meta.gated.values
    return roc_auc_score(y[g], p[g]) if len(set(y[g])) == 2 else np.nan


def evaluate(params, dev):
    wm, wX = dev["wedafall"]
    y, grp = wm.label.values, wm.group.values
    oof = np.full(len(y), np.nan)
    for s in np.unique(grp):
        te = grp == s
        oof[te] = _fit(params, wX[~te], y[~te]).predict_proba(wX[te])[:, 1]
    full = _fit(params, wX, y)
    r = dict(train=_gauc(wm, full.predict_proba(wX)[:, 1]), weda=_gauc(wm, oof))
    for key, name in (("uma", "umafall"), ("fa", "fallalld")):
        m, X = dev[name]
        r[key] = _gauc(m, full.predict_proba(X)[:, 1])
    r["unseen"] = np.nanmean([r["uma"], r["fa"]])
    return {**params, **r}


def _run(grid, dev, label):
    """All settings in parallel (one per CPU core), progress printed in order."""
    from joblib import Parallel, delayed
    t0, rows = time.time(), []
    jobs = (delayed(evaluate)(p, dev) for p in grid)
    try:
        results = Parallel(n_jobs=-1, return_as="generator")(jobs)
    except TypeError:                                   # older joblib
        results = Parallel(n_jobs=-1)(jobs)
    for i, r in enumerate(results, 1):
        rows.append(r)
        print(f"[{i:3d}/{len(grid)}] {label(r)}  WEDA {r['weda']:.3f}  UMAFall {r['uma']:.3f}  "
              f"FallAllD {r['fa']:.3f}  unseen {r['unseen']:.3f}   ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)


def _pick(D):
    ok = D[D.weda >= WEDA_FLOOR]
    if not len(ok):
        return None
    best = ok.unseen.max()
    ties = ok[ok.unseen >= best - TIE]
    return ties.sort_values(["max_depth", "n_estimators", "unseen"],
                            ascending=[True, True, False]).iloc[0]


def _desc(r):
    return (f"depth {int(r['max_depth'])} trees {int(r['n_estimators'])} lr {r['learning_rate']} "
            f"mcw {r['min_child_weight']:g} lambda {r['reg_lambda']:g} "
            f"sub {r['subsample']:g}")


# --------------------------------------------------------------------------
# step 1: sweep
# --------------------------------------------------------------------------

def sweep():
    dev = _dev_all()
    base = evaluate(CURRENT, dev)
    print(f"current setting: WEDA {base['weda']:.3f}  UMAFall {base['uma']:.3f}  "
          f"FallAllD {base['fa']:.3f}  unseen {base['unseen']:.3f}\n")
    grid = [dict(CURRENT, max_depth=d, n_estimators=n) for d in DEPTHS for n in TREES]
    D = _run(grid, dev, lambda r: f"depth {r['max_depth']} trees {r['n_estimators']:3d}")
    D.round(4).to_csv(S35 / "tune_weda_sweep.csv", index=False)
    (S35 / "tune_weda_current.json").write_text(json.dumps(base, indent=2, default=float))
    _plot_sweep(D)
    p = _pick(D)
    print(f"\nsweep pick: {_desc(p)}  WEDA {p.weda:.3f}  unseen {p.unseen:.3f}" if p is not None
          else "\nno setting keeps WEDA >= floor")
    print(f"saved {S35 / 'tune_weda_sweep.csv'}\nfigure: {FIG / 'stage35b_fitting_curve.png'}")


def _plot_sweep(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("train", "Training data (WEDA dev)"), ("weda", "WEDA, new people (LOSO)"),
            ("uma", "UMAFall dev (unseen)"), ("fa", "FallAllD dev (unseen)")]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for a, (col, title) in zip(ax, cols):
        for d in DEPTHS:
            s = D[D.max_depth == d]
            a.plot(s.n_estimators, s[col], marker="o", ms=3, label=f"depth {d}")
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks(TREES)
        a.set_xticklabels([str(t) for t in TREES])
        a.set_title(title)
        a.set_xlabel("number of trees")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("gated AUC")
    ax[3].legend(fontsize=8)
    fig.suptitle("WEDA-trained XGBoost from underfit to overfit")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage35b_fitting_curve.png", dpi=150)


# --------------------------------------------------------------------------
# step 2: fine grid around the sweep pick
# --------------------------------------------------------------------------

def fine():
    S = pd.read_csv(S35 / "tune_weda_sweep.csv")
    p = _pick(S)
    d, n = int(p.max_depth), int(p.n_estimators)
    FINE = dict(max_depth=sorted({max(1, d - 1), d, d + 1}),
                n_estimators=sorted({max(25, n // 2), n, n * 2}),
                learning_rate=[0.05, 0.1, 0.2],
                min_child_weight=[1.0, 5.0],
                reg_lambda=[1.0, 10.0],
                subsample=[0.8, 1.0])                  # colsample_bytree set equal
    print(f"fine grid around depth {d}, trees {n}: "
          + ", ".join(f"{k} {v}" for k, v in FINE.items()) + "\n")
    grid = [dict(zip(FINE, v)) for v in itertools.product(*FINE.values())]
    for g in grid:
        g["colsample_bytree"] = g["subsample"]
    dev = _dev_all()
    D = _run(grid, dev, _desc)
    D.round(4).to_csv(S35 / "tune_weda_fine.csv", index=False)

    base = json.loads((S35 / "tune_weda_current.json").read_text())
    best = _pick(D)
    ok = D[D.weda >= WEDA_FLOOR]
    pd.set_option("display.width", 220)
    cols = ["max_depth", "n_estimators", "learning_rate", "min_child_weight", "reg_lambda",
            "subsample", "weda", "uma", "fa", "unseen"]
    print(f"\n{len(ok)} of {len(D)} settings keep WEDA >= {WEDA_FLOOR}. Top 10 on unseen:")
    print(ok.sort_values("unseen", ascending=False)[cols].head(10).round(3).to_string(index=False))
    gain = best.unseen - base["unseen"]
    switch = gain >= MIN_GAIN
    print(f"\ncurrent : WEDA {base['weda']:.3f}  unseen {base['unseen']:.3f}")
    print(f"pick    : {_desc(best)}  WEDA {best.weda:.3f}  unseen {best.unseen:.3f}  "
          f"(gain {gain:+.3f})")
    print(f"\n{'SWITCH: run  py -m testing.tune_weda build' if switch else 'KEEP the current model'}"
          f"  (rule: switch only if gain >= {MIN_GAIN})")
    pick = {k: (float(best[k]) if k not in ("max_depth", "n_estimators") else int(best[k]))
            for k in CURRENT}
    (S35 / "tune_weda_pick.json").write_text(json.dumps(
        dict(params=pick, switch=bool(switch), gain=float(gain),
             weda=float(best.weda), unseen=float(best.unseen)), indent=2))
    print(f"saved {S35 / 'tune_weda_fine.csv'}, {S35 / 'tune_weda_pick.json'}")


# --------------------------------------------------------------------------
# step 3: build (only if the rule says switch)
# --------------------------------------------------------------------------

def build(out="models/motion.joblib"):
    import shutil
    from streams.motion import MotionStream, GATE_SVM_G, GATE_GYRO_DPS, RATE_HZ, ACC_CLIP_G
    from testing.stage35 import _threshold
    P = json.loads((S35 / "tune_weda_pick.json").read_text())
    if not P["switch"]:
        sys.exit("the rule said KEEP; the current model stays")
    params = P["params"]
    dm, dX = _dev_all()["wedafall"]
    y, grp = dm.label.values, dm.group.values
    oof = np.full(len(y), np.nan)
    for s in np.unique(grp):
        te = grp == s
        oof[te] = _fit(params, dX[~te], y[~te]).predict_proba(dX[te])[:, 1]
    thr = _threshold(dm, oof)
    final = _fit(params, dX, y)

    ms = MotionStream(gate_svm=GATE_SVM_G, gate_gyro=GATE_GYRO_DPS)
    ms.model, ms.classes_ = final, [0, 1]
    ms.platt, ms.threshold, ms.feature_index = (1.0, 0.0), thr, list(range(dX.shape[1]))
    cfg = dict(model="xgboost", params=params, feature_set="all_18",
               feature_index=ms.feature_index, rate_hz=RATE_HZ, acc_clip_g=ACC_CLIP_G,
               gate_svm_g=GATE_SVM_G, platt=dict(a=1.0, b=0.0), platt_used=False,
               threshold=thr, threshold_rule="95% specificity on gated ADL segments, "
               "WEDA-FALL dev LOSO, raw scores (evaluation only)",
               trained_on="WEDA-FALL dev subjects (stage 3.5b tuned)",
               n_segments=int(len(y)), n_falls=int(y.sum()),
               note="tuned on dev data after the one-time test; no fresh test score")
    ms.config = cfg
    out = Path(out)
    backup = out.with_name("motion_weda_stage35.joblib")
    if out.exists() and not backup.exists():
        shutil.copy(out, backup)
        print(f"previous model kept as {backup}")
    ms.save(out)
    assert np.allclose(MotionStream.load(out).impact_probability(dX),
                       final.predict_proba(dX)[:, 1], atol=1e-6), "saved model does not reproduce"
    (S35 / "final_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {out}: threshold {thr:.4f}; config in {S35 / 'final_config.json'}")
    print("next: py -m streams.motion --selftest --model models/motion.joblib")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sweep", "fine", "build"])
    a = ap.parse_args(argv)
    {"sweep": sweep, "fine": fine, "build": build}[a.step]()


if __name__ == "__main__":
    main()
