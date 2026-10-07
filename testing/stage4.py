"""
testing/stage4.py -- Stage 4 Part B: try improvements on the WEDA-trained motion
model. Dev parts only (the test parts are spent).

    WEDA-FALL dev   training data; 'weda' = leave-one-person-out (new people)
    UMAFall dev     unseen device
    FallAllD dev    unseen device

On a new device we pretend we only have its normal daily activity to
calibrate with (on the real band: the first days of wear). To keep that
honest, each unseen dataset is split into two halves BY PERSON: the
calibration comes from one half and is scored on the other, then swapped.

Steps (repo root):
    py -m testing.stage4 e1    device threshold   (operating rule: +5 points falls caught)
    py -m testing.stage4 e2    device scaling     (model rule: +0.01 unseen AUC)
    py -m testing.stage4 prep  re-cut segments for E3/E4 (dev people only, reads raw data once)
    py -m testing.stage4 e3    longer look after impact: 1.5 vs 3 vs 5 s   (model rule)
    py -m testing.stage4 e4    lower gate: 2.25 vs 2.0 vs 1.8 g            (operating rule)
    py -m testing.stage4 build save the adopted model (3 s after impact) as models/motion.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testing.compare_motion import OUT  # noqa: E402
from testing.stage35 import _dev_all, _threshold  # noqa: E402
from testing.tune_weda import CURRENT, WEDA_FLOOR, _fit, _gauc  # noqa: E402

S4 = OUT / "stage4"
UNSEEN = ("umafall", "fallalld")
MODEL_GAIN = 0.01      # E2, E3
OPER_GAIN = 0.05       # E1, E4 (5 points of falls caught)
SPEC_HELD = 0.93       # "held at 95%": within 2 points on the people it was not set on


def _halves(meta, seed=0):
    """Two groups of people, so calibration never sees the people it is scored on."""
    subs = np.array(sorted(meta.subject.unique()))
    np.random.default_rng(seed).shuffle(subs)
    return [meta.subject.isin(subs[0::2]).values, meta.subject.isin(subs[1::2]).values]


def _operating(meta, alarm):
    y, g = meta.label.values, meta.gated.values
    adl_h = meta.loc[meta.label == 0, "seconds"].sum() / 3600
    gadl = g & (y == 0)
    return dict(falls=int(y.sum()), falls_caught=alarm[y == 1].mean(),
                specificity_gated=1 - alarm[gadl].sum() / max(gadl.sum(), 1),
                false_alarms_per_hour=alarm[y == 0].sum() / adl_h)


def _weda_loso(X, meta, scale=None):
    """LOSO on WEDA dev. scale(train_mask) -> (center, spread) or None."""
    y, grp = meta.label.values, meta.group.values
    oof = np.full(len(y), np.nan)
    for s in np.unique(grp):
        te = grp == s
        Xs = X
        if scale is not None:
            c, sp = scale(~te)
            Xs = (X - c) / sp
        oof[te] = _fit(CURRENT, Xs[~te], y[~te]).predict_proba(Xs[te])[:, 1]
    return oof


# --------------------------------------------------------------------------
# E1: device threshold
# --------------------------------------------------------------------------

def e1():
    S4.mkdir(parents=True, exist_ok=True)
    dev = _dev_all()
    wm, wX = dev["wedafall"]
    thr_fixed = _threshold(wm, _weda_loso(wX, wm))
    model = _fit(CURRENT, wX, wm.label.values)
    print(f"fixed threshold (WEDA dev, 95% specificity on gated ADLs): {thr_fixed:.4f}\n")

    rows = []
    for name in UNSEEN:
        m, X = dev[name]
        p = model.predict_proba(X)[:, 1]
        g = m.gated.values
        rows.append(dict(dataset=name, method="fixed (WEDA threshold)", threshold=thr_fixed,
                         **_operating(m, g & (p >= thr_fixed))))
        alarm, thrs = np.zeros(len(m), bool), []
        A, B = _halves(m)
        for cal, ev in ((A, B), (B, A)):
            t = _threshold(m[cal].reset_index(drop=True), p[cal])
            thrs.append(t)
            alarm[ev] = g[ev] & (p[ev] >= t)
        rows.append(dict(dataset=name, method="device (own daily activity)",
                         threshold=float(np.mean(thrs)), **_operating(m, alarm)))
    T = pd.DataFrame(rows)
    T.round(4).to_csv(S4 / "e1_threshold.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))

    mean = T.groupby("method").falls_caught.mean()
    gain = mean["device (own daily activity)"] - mean["fixed (WEDA threshold)"]
    dev_spec = T[T.method.str.startswith("device")].specificity_gated
    held = bool((dev_spec >= SPEC_HELD).all())
    print(f"\nmean falls caught: fixed {mean['fixed (WEDA threshold)']:.3f}, "
          f"device {mean['device (own daily activity)']:.3f}  (gain {gain:+.3f})")
    print(f"device specificity on the scored halves: {', '.join(f'{v:.3f}' for v in dev_spec)}"
          f"  -> {'held' if held else 'NOT held'} (needs >= {SPEC_HELD} everywhere)")
    print(f"E1 verdict: {'ADOPT' if gain >= OPER_GAIN and held else 'REJECT'}  "
          f"(rule: +{OPER_GAIN:.2f} with false alarms held at 95% specificity)")
    print(f"saved {S4 / 'e1_threshold.csv'}")


# --------------------------------------------------------------------------
# E2: device scaling
# --------------------------------------------------------------------------

def _stats(X, meta, mask, kind):
    """Centre and spread of each feature over the device's DAILY ACTIVITY only."""
    A = X[mask & (meta.label.values == 0)]
    if kind == "robust":
        c = np.median(A, axis=0)
        s = np.percentile(A, 75, axis=0) - np.percentile(A, 25, axis=0)
    else:
        c, s = A.mean(axis=0), A.std(axis=0)
    return c, np.where(s > 1e-9, s, 1.0)


def e2():
    S4.mkdir(parents=True, exist_ok=True)
    dev = _dev_all()
    wm, wX = dev["wedafall"]
    y = wm.label.values
    rows = []
    for kind in ("none", "robust", "zscore"):
        if kind == "none":
            oof = _weda_loso(wX, wm)
            model = _fit(CURRENT, wX, y)
        else:
            oof = _weda_loso(wX, wm, scale=lambda tr, k=kind: _stats(wX, wm, tr, k))
            c, s = _stats(wX, wm, np.ones(len(y), bool), kind)
            model = _fit(CURRENT, (wX - c) / s, y)
        r = dict(scaling=kind, weda=_gauc(wm, oof))
        for name in UNSEEN:
            m, X = dev[name]
            if kind == "none":
                p = model.predict_proba(X)[:, 1]
            else:
                p = np.full(len(m), np.nan)
                A, B = _halves(m)
                for cal, ev in ((A, B), (B, A)):
                    c, s = _stats(X, m, cal, kind)
                    p[ev] = model.predict_proba((X[ev] - c) / s)[:, 1]
            r[name] = _gauc(m, p)
        r["unseen"] = float(np.mean([r[n] for n in UNSEEN]))
        rows.append(r)
        print(f"{kind:7s} WEDA {r['weda']:.3f}  UMAFall {r['umafall']:.3f}  "
              f"FallAllD {r['fallalld']:.3f}  unseen {r['unseen']:.3f}")
    T = pd.DataFrame(rows)
    T.round(4).to_csv(S4 / "e2_scaling.csv", index=False)

    base = T.set_index("scaling").loc["none"]
    ok = T[(T.scaling != "none") & (T.weda >= WEDA_FLOOR)]
    best = ok.loc[ok.unseen.idxmax()] if len(ok) else None
    if best is None:
        print("\nE2 verdict: REJECT (no scaling keeps WEDA >= floor)")
    else:
        gain = best.unseen - base.unseen
        print(f"\nbest scaling: {best.scaling}  unseen {best.unseen:.3f} vs none "
              f"{base.unseen:.3f}  (gain {gain:+.3f})")
        print(f"E2 verdict: {'ADOPT ' + best.scaling if gain >= MODEL_GAIN else 'REJECT'}  "
              f"(rule: +{MODEL_GAIN} unseen AUC, WEDA >= {WEDA_FLOOR})")
    print(f"saved {S4 / 'e2_scaling.csv'}")


# --------------------------------------------------------------------------
# E3 / E4: re-cut segments (dev people only), several variants in one pass
# --------------------------------------------------------------------------

VARIANTS = {                     # name: (gate g, seconds after the trigger)
    "g2.25_p1.5": (2.25, 1.5),   # current
    "g2.25_p3.0": (2.25, 3.0),
    "g2.25_p5.0": (2.25, 5.0),
    "g2.0_p1.5": (2.0, 1.5),
    "g1.8_p1.5": (1.8, 1.5),
}
PRE = 2.0
FS = 50.0


def prep():
    import json
    from testing.stage35 import DATASETS, ROOTS, S35, _scanner
    from data.loader import scan as uma_scan
    from streams.motion import GATE_GYRO_DPS, harmonise, segment_features
    S4.mkdir(parents=True, exist_ok=True)
    splits = json.loads((S35 / "splits.json").read_text())
    rows = {v: [] for v in VARIANTS}
    feats = {v: [] for v in VARIANTS}
    pre_n = int(PRE * FS)
    for name in DATASETS:
        dev_people = set(splits[name]["dev"])
        print(f"{name}: reading and cutting {len(VARIANTS)} variants ...")
        for rec in (_scanner(name) or uma_scan)(Path(ROOTS[name])):
            if rec.subject not in dev_people:
                continue                                   # test people are never cut
            acc = np.asarray(rec.acc, dtype=float)
            gyro = np.asarray(rec.gyro, dtype=float) if rec.gyro is not None else None
            acc, gyro = harmonise(acc, gyro)
            n, svm = len(acc), np.linalg.norm(acc, axis=1)
            for v, (gate, post) in VARIANTS.items():
                post_n = int(post * FS)
                if n < pre_n + post_n + 2:
                    continue
                hit = svm > gate
                if gyro is not None:
                    hit = hit | (np.linalg.norm(gyro, axis=1) > GATE_GYRO_DPS)
                idx = int(np.flatnonzero(hit)[0]) if hit.any() else int(np.argmax(svm))
                i = int(np.clip(idx, pre_n, n - post_n - 1))
                sa = acc[i - pre_n:i + post_n + 1]
                sg = gyro[i - pre_n:i + post_n + 1] if gyro is not None else None
                feats[v].append(segment_features(sa, sg, FS, pre_n))
                rows[v].append(dict(dataset=name, subject=rec.subject,
                                    group=f"{name}:{rec.subject}", label=int(rec.is_fall),
                                    gated=bool(hit.any()), seconds=rec.duration,
                                    file=f"{rec.path.parent.name}/{rec.path.name}",
                                    shifted=bool(i != idx),
                                    cause=getattr(rec, "cause", ""),
                                    elder=bool(getattr(rec, "elder", False))))
    out = {v: (pd.DataFrame(rows[v]), np.asarray(feats[v])) for v in VARIANTS}
    for v, (m, _) in out.items():
        key = m.dataset + "/" + m.file
        assert key.is_unique, f"{v}: recording names are not unique, matching would be wrong"
    pd.to_pickle(out, S4 / "variants_dev.pkl")
    print()
    for v, (m, _) in out.items():
        sh = m[m.label == 1].shifted.mean()
        print(f"{v:11s} segments {len(m):5d}  falls past gate {m[m.label == 1].gated.mean():.3f}  "
              f"ADLs past gate {m[m.label == 0].gated.mean():.3f}  "
              f"falls with window pushed back (recording ends too soon) {sh:.3f}")
    print(f"\nsaved {S4 / 'variants_dev.pkl'}")


def _load_variant(v, keep=None):
    m, X = pd.read_pickle(S4 / "variants_dev.pkl")[v]
    if not (m.dataset + "/" + m.file).is_unique:
        sys.exit("variants_dev.pkl is from the old prep (WEDA names repeat across "
                 "activity folders). Run  py -m testing.stage4 prep  again.")
    if keep is not None:
        k = (m.dataset + "/" + m.file).isin(keep).values
        m, X = m[k].reset_index(drop=True), X[k]
    return {n: (m[m.dataset == n].reset_index(drop=True), X[(m.dataset == n).values])
            for n in ("umafall", "fallalld", "wedafall")}


def _scores(d):
    """WEDA LOSO scores and unseen-dataset scores for one variant."""
    wm, wX = d["wedafall"]
    out = {"wedafall": (wm, _weda_loso(wX, wm))}
    model = _fit(CURRENT, wX, wm.label.values)
    for n in UNSEEN:
        m, X = d[n]
        out[n] = (m, model.predict_proba(X)[:, 1])
    return out


def e3():
    allv = pd.read_pickle(S4 / "variants_dev.pkl")
    names = ["g2.25_p1.5", "g2.25_p3.0", "g2.25_p5.0"]
    keys = [set(allv[v][0].dataset + "/" + allv[v][0].file) for v in names]
    keep = set.intersection(*keys)                 # same recordings for every variant
    print(f"comparing on the {len(keep)} recordings long enough for all three\n")
    rows = []
    for v in names:
        sc = _scores(_load_variant(v, keep))
        r = dict(variant=v, after_s=VARIANTS[v][1], weda=_gauc(*sc["wedafall"]),
                 **{n: _gauc(*sc[n]) for n in UNSEEN})
        r["unseen"] = float(np.mean([r[n] for n in UNSEEN]))
        rows.append(r)
        print(f"{VARIANTS[v][1]:.1f} s after  WEDA {r['weda']:.3f}  UMAFall {r['umafall']:.3f}  "
              f"FallAllD {r['fallalld']:.3f}  unseen {r['unseen']:.3f}")
    T = pd.DataFrame(rows)
    T.round(4).to_csv(S4 / "e3_post_window.csv", index=False)
    base = T.iloc[0]
    ok = T.iloc[1:][T.iloc[1:].weda >= WEDA_FLOOR]
    if not len(ok):
        print("\nE3 verdict: REJECT (no longer window keeps WEDA >= floor)")
    else:
        best = ok.loc[ok.unseen.idxmax()]
        gain = best.unseen - base.unseen
        print(f"\nbest: {best.after_s:.1f} s, unseen {best.unseen:.3f} vs 1.5 s "
              f"{base.unseen:.3f}  (gain {gain:+.3f})")
        print(f"E3 verdict: {'ADOPT ' + str(best.after_s) + ' s' if gain >= MODEL_GAIN else 'REJECT'}"
              f"  (rule: +{MODEL_GAIN} unseen AUC, WEDA >= {WEDA_FLOOR})")
    print(f"saved {S4 / 'e3_post_window.csv'}")


def _caught_at_spec(m, p, spec=0.95):
    """Whole system (gate AND model): falls caught when at most 5% of ALL daily
    activities raise an alarm. Threshold read off this dataset's own ROC, so it
    measures what the gate + model can do, not how well a threshold travels."""
    y, g = m.label.values, m.gated.values
    adl = np.sort(p[g & (y == 0)])[::-1]
    k = int(np.floor((1 - spec) * (y == 0).sum()))   # alarms allowed
    t = -np.inf if len(adl) <= k else np.nextafter(adl[k], np.inf)
    return float((g & (p >= t))[y == 1].mean())


def e4():
    rows = []
    for v in ["g2.25_p1.5", "g2.0_p1.5", "g1.8_p1.5"]:
        sc = _scores(_load_variant(v))
        r = dict(variant=v, gate_g=VARIANTS[v][0])
        for n in ("wedafall",) + UNSEEN:
            m, p = sc[n]
            r[f"{n}_gate_falls"] = m[m.label == 1].gated.mean()
            r[f"{n}_gate_adls"] = m[m.label == 0].gated.mean()
            r[f"{n}_caught"] = _caught_at_spec(m, p)
        r["unseen_caught"] = float(np.mean([r[f"{n}_caught"] for n in UNSEEN]))
        rows.append(r)
        print(f"gate {r['gate_g']:.2f} g  falls caught at 95% specificity:  "
              f"WEDA {r['wedafall_caught']:.3f}  UMAFall {r['umafall_caught']:.3f}  "
              f"FallAllD {r['fallalld_caught']:.3f}  unseen mean {r['unseen_caught']:.3f}")
        print(f"            falls past gate: WEDA {r['wedafall_gate_falls']:.3f}  "
              f"UMAFall {r['umafall_gate_falls']:.3f}  FallAllD {r['fallalld_gate_falls']:.3f}")
    T = pd.DataFrame(rows)
    T.round(4).to_csv(S4 / "e4_gate.csv", index=False)
    base = T.iloc[0]
    best = T.iloc[1:].loc[T.iloc[1:].unseen_caught.idxmax()]
    gain = best.unseen_caught - base.unseen_caught
    print(f"\nbest: gate {best.gate_g:.2f} g, unseen {best.unseen_caught:.3f} vs 2.25 g "
          f"{base.unseen_caught:.3f}  (gain {gain:+.3f})")
    print(f"E4 verdict: {'ADOPT ' + str(best.gate_g) + ' g' if gain >= OPER_GAIN else 'REJECT'}"
          f"  (rule: +{OPER_GAIN:.2f} falls caught at 95% specificity)")
    print(f"saved {S4 / 'e4_gate.csv'}")


# --------------------------------------------------------------------------
# build: the adopted change (E3, 3 s after impact)
# --------------------------------------------------------------------------

ADOPTED = "g2.25_p3.0"


def build(out="models/motion.joblib"):
    import json
    import shutil
    from streams.motion import (MotionStream, GATE_SVM_G, GATE_GYRO_DPS, RATE_HZ,
                                ACC_CLIP_G, POST_SEC)
    gate, post = VARIANTS[ADOPTED]
    if abs(POST_SEC - post) > 1e-9:
        sys.exit(f"streams/motion.py still has POST_SEC = {POST_SEC}; set it to {post} first")

    # Check only (nothing is chosen here): 1.5 s vs 3 s on every recording
    # long enough for 3 s, not just the ones long enough for 5 s.
    allv = pd.read_pickle(S4 / "variants_dev.pkl")
    keep = set.intersection(*[set(allv[v][0].dataset + "/" + allv[v][0].file)
                              for v in ("g2.25_p1.5", ADOPTED)])
    print(f"check on {len(keep)} recordings (all long enough for 3 s):")
    for v in ("g2.25_p1.5", ADOPTED):
        sc = _scores(_load_variant(v, keep))
        r = {n: _gauc(*sc[n]) for n in ("wedafall",) + UNSEEN}
        print(f"  {VARIANTS[v][1]:.1f} s after  WEDA {r['wedafall']:.3f}  UMAFall {r['umafall']:.3f}"
              f"  FallAllD {r['fallalld']:.3f}  unseen {np.mean([r[n] for n in UNSEEN]):.3f}")

    wm, wX = _load_variant(ADOPTED)["wedafall"]
    y = wm.label.values
    thr = _threshold(wm, _weda_loso(wX, wm))
    final = _fit(CURRENT, wX, y)
    ms = MotionStream(gate_svm=GATE_SVM_G, gate_gyro=GATE_GYRO_DPS)
    ms.model, ms.classes_ = final, [0, 1]
    ms.platt, ms.threshold, ms.feature_index = (1.0, 0.0), thr, list(range(wX.shape[1]))
    cfg = dict(model="xgboost", params=CURRENT, feature_set="all_18",
               feature_index=ms.feature_index, rate_hz=RATE_HZ, acc_clip_g=ACC_CLIP_G,
               gate_svm_g=GATE_SVM_G, pre_sec=PRE, post_sec=post,
               platt=dict(a=1.0, b=0.0), platt_used=False, threshold=thr,
               threshold_rule="95% specificity on gated ADL segments, WEDA-FALL dev LOSO, "
               "raw scores (evaluation only; fusion uses the probability)",
               trained_on="WEDA-FALL dev subjects, 3 s after impact (stage 4, E3)",
               n_segments=int(len(y)), n_falls=int(y.sum()),
               note="changed after the one-time test; dev numbers only")
    ms.config = cfg
    out = Path(out)
    backup = out.with_name("motion_weda_post1p5.joblib")
    if out.exists() and not backup.exists():
        shutil.copy(out, backup)
        print(f"\nprevious model kept as {backup}")
    ms.save(out)
    assert np.allclose(MotionStream.load(out).impact_probability(wX),
                       final.predict_proba(wX)[:, 1], atol=1e-6), "saved model does not reproduce"
    (S4 / "final_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {out}: {len(y)} segments, {int(y.sum())} falls, threshold {thr:.4f}")
    print("next: py -m streams.motion --selftest --model models/motion.joblib")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["e1", "e2", "prep", "e3", "e4", "build"])
    a = ap.parse_args(argv)
    {"e1": e1, "e2": e2, "prep": prep, "e3": e3, "e4": e4, "build": build}[a.step]()


if __name__ == "__main__":
    main()
