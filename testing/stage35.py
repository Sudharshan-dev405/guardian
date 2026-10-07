"""
testing/stage35.py -- Stage 3.5: three datasets (UMAFall, FallAllD, WEDA-FALL).

Train on each dataset (and on pairs), test on the others. Same XGBoost, same
18 features, same signal path (+/-8 g clip, 20 Hz) and 2.25 g gate.

Rules fixed before any results (also in motion_module_log.txt):
- Every dataset has a locked test part, opened once in step 6.
- Choosing uses development parts only.
- Winner = highest average gated AUC on the other datasets' development parts,
  with at least 0.95 on its own dataset (LOSO).

Steps (repo root):
    py -m testing.stage35 split      step 1  lock the WEDA test subjects; one splits file
    py -m testing.stage35 prep       step 2  harmonised segments for all three datasets,
                                          dev and test (test files are LOCKED), sanity table
    py -m testing.stage35 tune       step 3  one dataset at a time: small XGBoost grid, own
                                          LOSO + the other two dev parts
    py -m testing.stage35 pooled     step 4  two datasets together, scored on the third
    py -m testing.stage35 matrix     step 5  dev results matrix and the winner by the rule
    py -m testing.stage35 test       step 6  ONE-TIME test on all three locked test parts:
                                          the step 5 winner vs the current Stage 3 model
    py -m testing.stage35 build      step 7  save the tested winner as models/motion.joblib
                                          (old model kept as motion_umafall_stage3.joblib)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testing.compare_motion import OUT  # noqa: E402

DATASETS = ("umafall", "fallalld", "wedafall")
ROOTS = {"umafall": r"data\raw\UMAFall",
         "fallalld": r"data\raw\FallAllD\FallAllD",
         "wedafall": r"data\raw\WEDA-FALL\dataset\50Hz"}

S35 = OUT / "stage35"
SEED = 0
WEDA_YOUNG = [f"U{i:02d}" for i in range(1, 15)]     # did falls and ADLs
WEDA_ELDER = [f"U{i:02d}" for i in range(21, 32)]    # ADLs only, ages 77-95
N_TEST_YOUNG, N_TEST_ELDER = 4, 3                    # about 25% of each group


def split():
    S35.mkdir(parents=True, exist_ok=True)
    path = S35 / "splits.json"
    if path.exists():
        print(f"{path} already exists, keeping it:\n{path.read_text()}")
        return
    uma = json.loads((OUT / "split.json").read_text())
    fa = json.loads((OUT / "fallalld_split.json").read_text())
    rng = np.random.default_rng(SEED)
    w_test = sorted(rng.choice(WEDA_YOUNG, N_TEST_YOUNG, replace=False).tolist()
                    + rng.choice(WEDA_ELDER, N_TEST_ELDER, replace=False).tolist())
    splits = {
        "umafall": {"dev": uma["dev_subjects"], "test": uma["test_subjects"],
                    "note": "test opened once before (stage 3 step 7, UMAFall-only model)"},
        "fallalld": {"dev": fa["select"], "test": fa["report"],
                     "note": "dev = select half, test = report half; test opened once "
                             "before (stage 3 step 7)"},
        "wedafall": {"dev": sorted(s for s in WEDA_YOUNG + WEDA_ELDER if s not in w_test),
                     "test": w_test,
                     "note": f"{N_TEST_YOUNG} young (falls + ADLs) and {N_TEST_ELDER} elder "
                             "(ADLs only) subjects, chosen at random with seed 0"},
        "seed": SEED,
        "rule": "test parts are opened once, in step 6; all choices use dev parts only",
    }
    path.write_text(json.dumps(splits, indent=2))
    for name in ("umafall", "fallalld", "wedafall"):
        d = splits[name]
        print(f"{name:9s} dev {len(d['dev']):2d} subjects | test {len(d['test'])}: {d['test']}")
    print(f"\nsaved {path}")


def _scanner(name):
    if name == "umafall":
        return None                                   # collect() default = UMAFall loader
    if name == "fallalld":
        from data import loader_fallalld
        return loader_fallalld.scan
    from data import loader_weda
    return loader_weda.scan


def load(name, part):
    """part = 'dev' or 'test'. Test files are read only by step 6."""
    if part == "test" and sys._getframe(1).f_code.co_name not in ("final_test",):
        raise RuntimeError("test parts are locked until step 6")
    tag = "dev" if part == "dev" else "test_LOCKED"
    return pd.read_pickle(S35 / f"{name}_{tag}.pkl")


def prep(roots=None):
    from testing.compare_motion import collect
    from streams.motion import harmonise, GATE_SVM_G
    roots = {**ROOTS, **(roots or {})}
    splits = json.loads((S35 / "splits.json").read_text())
    rows = []
    for name in DATASETS:
        print(f"{name}: building harmonised segments ...")
        meta, X, _ = collect(Path(roots[name]), scanner=_scanner(name), transform=harmonise)
        meta = meta.assign(dataset=name, group=name + ":" + meta.subject)
        for part in ("dev", "test"):
            k = meta.subject.isin(splits[name][part]).values
            m, x = meta[k].reset_index(drop=True), X[k]
            tag = "dev" if part == "dev" else "test_LOCKED"
            pd.to_pickle((m, x), S35 / f"{name}_{tag}.pkl")
            row = dict(dataset=name, part=part, subjects=m.subject.nunique(),
                       segments=len(m), falls=int(m.label.sum()),
                       adls=int((m.label == 0).sum()))
            if part == "dev":   # gate coverage only on dev; test stays unread
                f, a = m[m.label == 1], m[m.label == 0]
                row.update(gate_falls=f.gated.mean(), gate_adls=a.gated.mean())
                faint = f[f.cause == "faint"]
                if len(faint):
                    row.update(faint_falls=len(faint), gate_faint=faint.gated.mean())
                old = a[a.elder]
                if len(old):
                    row.update(elder_adls=len(old), gate_elder_adls=old.gated.mean())
            rows.append(row)
        missing = set(meta.subject) - set(splits[name]["dev"]) - set(splits[name]["test"])
        if missing:
            print(f"  note: subjects in neither part (ignored): {sorted(missing)}")
    T = pd.DataFrame(rows).set_index(["dataset", "part"])
    T.round(3).to_csv(S35 / "step2_sanity.csv")
    pd.set_option("display.width", 220)
    print(f"\ngate {GATE_SVM_G} g, +/-8 g clip, 20 Hz path for every dataset")
    print("(gate coverage shown for dev parts only; test parts stay unread)\n")
    print(T.round(3).to_string())
    print(f"\nsaved {S35}")


# --------------------------------------------------------------------------
# steps 3-5: tune, pooled, matrix (dev parts only)
# --------------------------------------------------------------------------

# Small grid around the Stage 3 setting (depth 2, 100 trees, lr 0.1, lambda 10).
GRID = dict(max_depth=[2, 3, 4], n_estimators=[50, 100, 300], reg_lambda=[1, 10])
FIXED = dict(learning_rate=0.1, min_child_weight=1.0)
OWN_FLOOR = 0.95      # own-data LOSO gated AUC must reach this
TIE = 0.005           # within 0.005 of the best = tie -> simplest wins
SHORT = {"umafall": "uma", "fallalld": "fa", "wedafall": "weda"}
SINGLE = {n: (n,) for n in DATASETS}
POOLED = {"uma+fa": ("umafall", "fallalld"),
          "uma+weda": ("umafall", "wedafall"),
          "fa+weda": ("fallalld", "wedafall")}


def _dev_all():
    return {n: load(n, "dev") for n in DATASETS}


def _pool(dev, names):
    metas, Xs = zip(*(dev[n] for n in names))
    return pd.concat(metas, ignore_index=True), np.vstack(Xs)


def _loso(params, X, meta):
    """Leave-one-person-out. Uses 'group' (dataset:subject) because subject
    IDs repeat across datasets (S02 is in UMAFall and FallAllD)."""
    from testing.stage3 import XGB
    y, g = meta.label.values, meta.group.values
    p = np.full(len(y), np.nan)
    for s in np.unique(g):
        te = g == s
        p[te] = XGB(**params).fit(X[~te], y[~te]).predict_proba(X[te])
    return p


def _score(params, names, dev):
    from testing.stage3 import XGB, gated_auc
    meta, X = _pool(dev, names)
    oof = _loso(params, X, meta)
    full = XGB(**params).fit(X, meta.label.values)
    r = dict(own_loso=gated_auc(meta, oof))
    if len(names) > 1:                       # own LOSO per dataset, for information
        for n in names:
            k = (meta.dataset == n).values
            r[f"own_{SHORT[n]}"] = gated_auc(meta[k], oof[k])
    others = [o for o in DATASETS if o not in names]
    for o in others:
        m, x = dev[o]
        r[o] = gated_auc(m, full.predict_proba(x))
    r["cross_mean"] = float(np.nanmean([r[o] for o in others]))
    return r


def _grid(train_sets, out_name):
    import itertools
    import time
    dev = _dev_all()
    keys = list(GRID)
    combos = [dict(zip(keys, v), **FIXED) for v in itertools.product(*GRID.values())]
    rows, t0, total, i = [], time.time(), len(train_sets) * len(combos), 0
    for tname, names in train_sets.items():
        for params in combos:
            i += 1
            r = dict(train=tname, **params, **_score(params, names, dev))
            rows.append(r)
            print(f"[{i:3d}/{total}] {tname:9s} depth {params['max_depth']} trees "
                  f"{params['n_estimators']:3d} lambda {params['reg_lambda']:2d}  own "
                  f"{r['own_loso']:.3f}  cross {r['cross_mean']:.3f}   ({time.time() - t0:.0f} s)")
    D = pd.DataFrame(rows)
    D.round(4).to_csv(S35 / out_name, index=False)
    print()
    for tname in train_sets:
        best, ok = _pick(D[D.train == tname])
        flag = "" if ok else f"   (no setting reaches own {OWN_FLOOR}; best shown)"
        print(f"{tname:9s} pick: depth {int(best.max_depth)} trees {int(best.n_estimators)} "
              f"lambda {int(best.reg_lambda)}  own {best.own_loso:.3f}  "
              f"cross {best.cross_mean:.3f}{flag}")
    print(f"\nsaved {S35 / out_name}")


def _pick(D):
    """Own LOSO >= floor; highest cross_mean; ties (0.005) -> shallowest, fewest trees."""
    D = D.assign(_c=D.cross_mean.fillna(-1.0))   # NaN = could not be scored
    ok = D[D.own_loso >= OWN_FLOOR]
    passed = len(ok) > 0
    if not passed:
        ok = D
    best = ok._c.max()
    ties = ok[ok._c >= best - TIE]
    ties = ties.sort_values(["max_depth", "n_estimators", "cross_mean"],
                            ascending=[True, True, False])
    return ties.iloc[0], passed


def tune():
    _grid(SINGLE, "step3_single.csv")


def pooled():
    _grid(POOLED, "step4_pooled.csv")


def matrix():
    """Step 5: one row per training set (its picked setting), one column per
    dataset (dev part). Own-data cells are LOSO; others are trained-on-all."""
    from testing.stage3 import XGB
    from sklearn.metrics import roc_auc_score
    D = pd.concat([pd.read_csv(S35 / "step3_single.csv"),
                   pd.read_csv(S35 / "step4_pooled.csv")], ignore_index=True)
    dev = _dev_all()
    wm, wX = dev["wedafall"]
    rows = []
    for tname, names in {**SINGLE, **POOLED}.items():
        best, ok = _pick(D[D.train == tname])
        row = dict(train=tname, depth=int(best.max_depth), trees=int(best.n_estimators),
                   reg_lambda=int(best.reg_lambda), own_loso=best.own_loso,
                   floor_ok=ok, cross_mean=best.cross_mean)
        for n in DATASETS:
            row[n] = best.own_loso if n in names and len(names) == 1 else best.get(n, np.nan)
            if n in names and len(names) > 1:
                row[n] = best[f"own_{SHORT[n]}"]
        if "wedafall" not in names:          # faint-type falls vs ADLs, gated, unseen data
            params = dict(max_depth=row["depth"], n_estimators=row["trees"],
                          reg_lambda=row["reg_lambda"], **FIXED)
            meta, X = _pool(dev, names)
            p = XGB(**params).fit(X, meta.label.values).predict_proba(wX)
            k = (wm.gated & ((wm.cause == "faint") | (wm.label == 0))).values
            y = wm.label.values[k]
            row["weda_faint_auc"] = roc_auc_score(y, p[k]) if len(set(y)) == 2 else np.nan
        rows.append(row)
    M = pd.DataFrame(rows).set_index("train")
    M.round(4).to_csv(S35 / "step5_matrix.csv")
    pd.set_option("display.width", 220)
    print("gated AUC on dev parts. Cells for a training dataset are LOSO (unseen people);")
    print("other cells are unseen datasets. cross_mean = mean over unseen datasets.\n")
    print(M.round(3).to_string())

    # original rule (fixed before results)
    elig = M[M.floor_ok]
    orig = elig.cross_mean.idxmax() if len(elig) else None
    print(f"\nORIGINAL rule (own >= {OWN_FLOOR}, best cross_mean): {orig}")

    # revised rule (declared after steps 3-4, before step 5; see log).
    # Fair test of pooling: for each unseen dataset, every candidate gets its
    # own best setting for that dataset (equally optimistic for all).
    print("\nREVISED rule. Does training on two datasets beat each single one on the third?")
    helps = 0
    for n in DATASETS:
        pair = [t for t, names in POOLED.items() if n not in names][0]
        singles = [t for t, names in SINGLE.items() if n not in names]
        best = {t: D.loc[D.train == t, n].max() for t in singles + [pair]}
        ok = best[pair] > max(best[t] for t in singles)
        helps += ok
        print(f"  unseen {n:9s} " + "  ".join(f"{t} {best[t]:.3f}" for t in singles)
              + f"  | {pair} {best[pair]:.3f}  -> pooling {'helps' if ok else 'does not help'}")
    if helps >= 2:
        P = D[D.train.isin(list(POOLED))]
        G = (P.groupby(["max_depth", "n_estimators", "reg_lambda"]).cross_mean.mean()
             .rename("lodo").reset_index())
        top = G.lodo.max()
        g = G[G.lodo >= top - TIE].sort_values(["max_depth", "n_estimators", "lodo"],
                                               ascending=[True, True, False]).iloc[0]
        final = (f"train on ALL THREE dev parts, depth {int(g.max_depth)} trees "
                 f"{int(g.n_estimators)} lambda {int(g.reg_lambda)} "
                 f"(mean unseen-dataset AUC {g.lodo:.3f})")
        G.round(4).to_csv(S35 / "step5_lodo_settings.csv", index=False)
    else:
        final = f"single dataset: {M.loc[list(SINGLE)].cross_mean.idxmax()}"
    print(f"\nWINNER (revised rule, pooling helped on {helps} of 3): {final}\n")
    _plot_matrix(M)
    print(f"saved {S35 / 'step5_matrix.csv'}\nfigure: {FIG / 'stage35_step5_matrix.png'}")


FIG = OUT / "figures"


def _plot_matrix(M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = M[list(DATASETS)].astype(float).values
    trained = [set({**SINGLE, **POOLED}[t]) for t in M.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(V, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(V.shape[0]):
        for j, n in enumerate(DATASETS):
            own = n in trained[i]
            txt = "" if np.isnan(V[i, j]) else f"{V[i, j]:.3f}" + ("\n(LOSO)" if own else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="normal" if own else "bold")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(["UMAFall", "FallAllD", "WEDA-FALL"])
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{t}  (mean unseen {c:.3f})" for t, c in zip(M.index, M.cross_mean)])
    ax.set_xlabel("scored on (dev part)")
    ax.set_ylabel("trained on")
    ax.set_title("Gated AUC: bold = dataset never seen in training")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage35_step5_matrix.png", dpi=150)


# --------------------------------------------------------------------------
# step 6: the one-time test
# --------------------------------------------------------------------------

TARGET_SPEC = 0.95
NAMES = {"umafall": "UMAFall", "fallalld": "FallAllD", "wedafall": "WEDA-FALL"}


def _models():
    """Fixed before the test: the step 5 winner (revised rule) and the current
    model (Stage 3, UMAFall dev, settings from final_config.json)."""
    M = pd.read_csv(S35 / "step5_matrix.csv").set_index("train")
    w = M.loc["wedafall"]
    cfg = json.loads((OUT / "stage3" / "final_config.json").read_text())
    return {
        "WEDA-trained (winner)": ("wedafall", dict(max_depth=int(w.depth), n_estimators=int(w.trees),
                                                   reg_lambda=float(w.reg_lambda), **FIXED)),
        "UMAFall-trained (current)": ("umafall", cfg["params"]),
    }


def _threshold(meta, p):
    """Lowest score that keeps 95% of gated ADLs below it (dev LOSO)."""
    g = meta.gated.values
    adl = np.sort(p[g & (meta.label.values == 0)])
    k = int(np.ceil(TARGET_SPEC * len(adl)))
    return float(np.nextafter(adl[min(k, len(adl)) - 1], 1.0))


def _test_metrics(m, p, thr):
    from testing.stage3 import gated_auc
    from sklearn.metrics import roc_auc_score
    y, g = m.label.values, m.gated.values
    alarm = g & (p >= thr)
    adl_h = m.loc[m.label == 0, "seconds"].sum() / 3600
    r = dict(falls=int(y.sum()), adls=int((y == 0).sum()),
             auc_gated=gated_auc(m, p), auc_all=roc_auc_score(y, p),
             falls_caught=alarm[y == 1].mean(),
             false_alarms_per_hour=alarm[y == 0].sum() / adl_h)
    faint = (m.cause == "faint").values
    if faint.any():
        r["faint_caught"] = alarm[faint].mean()
    old = (m.elder & (m.label == 0)).values
    if old.any():
        r["elder_adl_false_alarms"] = f"{int(alarm[old].sum())} of {int(old.sum())}"
    return r


def final_test():
    import datetime
    from testing.stage3 import XGB
    flag = S35 / "step6_DONE.txt"
    if flag.exists():
        sys.exit(f"step 6 already ran ({flag.read_text().strip()}). Re-running would turn "
                 "the test parts into tuning data.")
    dev = _dev_all()
    test = {n: load(n, "test") for n in DATASETS}     # the only place test parts are read

    rows, rocs = [], {}
    for label, (train, params) in _models().items():
        dm, dX = dev[train]
        thr = _threshold(dm, _loso(params, dX, dm))
        model = XGB(**params).fit(dX, dm.label.values)
        print(f"{label}: trained on {NAMES[train]} dev, threshold {thr:.4f} "
              f"({TARGET_SPEC:.0%} specificity on gated dev ADLs, LOSO)")
        for n in DATASETS:
            m, X = test[n]
            p = model.predict_proba(X)
            rows.append(dict(model=label, test=NAMES[n],
                             seen=("same dataset, new people" if n == train else "unseen dataset"),
                             threshold=thr, **_test_metrics(m, p, thr)))
            rocs[(label, n)] = (m, p)
    T = pd.DataFrame(rows)
    T.round(4).to_csv(S35 / "step6_test.csv", index=False)
    pd.set_option("display.width", 240)
    print("\nONE-TIME TEST (locked test parts). Alarm = gate fired AND score >= threshold.\n")
    print(T.drop(columns="threshold").round(3).to_string(index=False))
    _plot_test_roc(rocs)
    flag.write_text(f"run {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    print(f"\nsaved {S35 / 'step6_test.csv'}\nfigure: {FIG / 'stage35_step6_roc.png'}")
    print("The test parts are now spent. Further changes are judged on dev data only.")


def _plot_test_roc(rocs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for a, n in zip(ax, DATASETS):
        for (label, tn), (m, p) in rocs.items():
            if tn != n:
                continue
            g = m.gated.values
            y = m.label.values[g]
            fpr, tpr, _ = roc_curve(y, p[g])
            a.plot(fpr, tpr, label=f"{label}  AUC {roc_auc_score(y, p[g]):.3f}")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_title(f"{NAMES[n]} test (gated segments)")
        a.set_xlabel("false positive rate")
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("true positive rate")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage35_step6_roc.png", dpi=150)


# --------------------------------------------------------------------------
# step 7: save the tested winner (exactly the model step 6 tested)
# --------------------------------------------------------------------------

def build(out="models/motion.joblib"):
    import shutil
    from testing.stage3 import XGB
    from streams.motion import MotionStream, GATE_SVM_G, GATE_GYRO_DPS, RATE_HZ, ACC_CLIP_G
    if not (S35 / "step6_DONE.txt").exists():
        sys.exit("run step 6 first")
    label, (train, params) = next(iter(_models().items()))
    dm, dX = _dev_all()[train]
    y = dm.label.values
    thr = _threshold(dm, _loso(params, dX, dm))
    final = XGB(**params).fit(dX, y)
    final.m.set_params(n_jobs=1)

    ms = MotionStream(gate_svm=GATE_SVM_G, gate_gyro=GATE_GYRO_DPS)
    ms.model, ms.classes_ = final.m, [0, 1]
    ms.platt, ms.threshold, ms.feature_index = (1.0, 0.0), thr, list(range(dX.shape[1]))
    cfg = dict(model="xgboost", params=params, feature_set="all_18",
               feature_index=ms.feature_index, rate_hz=RATE_HZ, acc_clip_g=ACC_CLIP_G,
               gate_svm_g=GATE_SVM_G, platt=dict(a=1.0, b=0.0), platt_used=False,
               threshold=thr, threshold_rule="95% specificity on gated ADL segments, "
               "WEDA-FALL dev LOSO, raw scores (evaluation only; fusion uses the probability)",
               trained_on=f"{NAMES[train]} dev subjects (stage 3.5 winner)",
               n_segments=int(len(y)), n_falls=int(y.sum()))
    ms.config = cfg

    out = Path(out)
    if out.exists():
        backup = out.with_name("motion_umafall_stage3.joblib")
        if not backup.exists():
            shutil.copy(out, backup)
            print(f"old model kept as {backup}")
    ms.save(out)
    p_check = MotionStream.load(out).impact_probability(dX)
    assert np.allclose(p_check, final.predict_proba(dX), atol=1e-6), "saved model does not reproduce"
    (S35 / "final_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {out}: {label}, {len(y)} segments, {int(y.sum())} falls, threshold {thr:.4f}")
    print(f"reload check passed; config in {S35 / 'final_config.json'}")
    print("\nnext: py -m streams.motion --selftest --model models/motion.joblib")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "prep", "tune", "pooled", "matrix", "test", "build"])
    a = ap.parse_args(argv)
    {"split": split, "prep": prep, "tune": tune, "pooled": pooled, "matrix": matrix,
     "test": final_test, "build": build}[a.step]()


if __name__ == "__main__":
    main()
