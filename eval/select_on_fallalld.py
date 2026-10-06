"""
eval/select_on_fallalld.py -- Stage 2b: choose between the top UMAFall models
using FallAllD, without spending FallAllD's honesty.

  * Each candidate is trained ONCE on the UMAFall dev subjects
    (eval/outputs/split.json). The UMAFall test subjects are never used.
  * FallAllD subjects are split in two (eval/outputs/fallalld_split.json):
      select  -> the model is chosen on these
      report  -> untouched by the choice; this is the cross-dataset number
                 that goes in the report
    If the model were chosen on all of FallAllD, the FallAllD result would no
    longer be a fair test of anything.
  * Both FallAllD signal paths are scored:
      native   238 -> 50 Hz (what a faster real sensor looks like)
      matched  238 -> 20 -> 50 Hz (UMAFall's own path; isolates people and
               protocol from sampling rate)
    The choice is made on the native path.
  * Nothing is retrained, refitted or re-thresholded on FallAllD.

Metrics, per model / path / half:
  auc_gated        model AUC on segments the 2.5 g gate passed
  auc_all          model AUC on every segment
  sys_sensitivity  falls that pass the gate AND score >= 0.5
  sys_fa_per_hour  ADLs that pass the gate AND score >= 0.5, per hour of ADL
  gate_falls / gate_adls   share of falls / ADLs the gate passes

Commands (repo root):
    py -m eval.select_on_fallalld                  # rf xgb svm
    py -m eval.select_on_fallalld --models rf xgb svm
    py -m eval.select_on_fallalld --limit 300      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loader_fallalld as fa  # noqa: E402
from eval.compare_motion import MODELS, OUT, SEED, collect  # noqa: E402

TIE = 0.01


def fallalld_segments(root, matched, limit):
    tag = "matched" if matched else "native"
    cache = OUT / f"fallalld_{tag}_segments.pkl"
    if cache.exists() and not limit:
        return pd.read_pickle(cache)
    meta, X, _ = collect(Path(root), scanner=fa.scan, limit=limit,
                         match_umafall_path=matched)
    out = (meta, X)
    if not limit:
        pd.to_pickle(out, cache)
    return out


def split_fallalld(meta):
    path = OUT / "fallalld_split.json"
    if path.exists():
        return json.loads(path.read_text())
    per = meta.groupby("subject").label.sum().sort_values(ascending=False)
    rng = np.random.default_rng(SEED)
    halves = {"select": [], "report": []}
    falls = {"select": 0, "report": 0}
    cap = int(np.ceil(len(per) / 2))
    # greedy: biggest fall counts first, each to the half with fewer falls so far
    order = list(per.index)
    for s in order:
        options = [h for h in halves if len(halves[h]) < cap]
        options.sort(key=lambda h: (falls[h], len(halves[h]), rng.random()))
        h = options[0]
        halves[h].append(s)
        falls[h] += int(per[s])
    split = {h: sorted(v) for h, v in halves.items()}
    split["falls"] = falls
    split["rule"] = "model chosen on 'select' only; 'report' is the cross-dataset result"
    path.write_text(json.dumps(split, indent=2))
    return split


def score(meta, p):
    y, g = meta.label.values, meta.gated.values
    adl_h = meta.loc[meta.label == 0, "seconds"].sum() / 3600
    hit = g & (p >= 0.5)
    return dict(
        n_falls=int(y.sum()), n_adls=int((y == 0).sum()),
        auc_gated=roc_auc_score(y[g], p[g]) if len(set(y[g])) == 2 else np.nan,
        auc_all=roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
        sys_sensitivity=hit[y == 1].mean() if y.sum() else np.nan,
        sys_fa_per_hour=hit[y == 0].sum() / adl_h if adl_h else np.nan,
        gate_falls=g[y == 1].mean() if y.sum() else np.nan,
        gate_adls=g[y == 0].mean() if (y == 0).sum() else np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--uma-root", default=r"data\raw\UMAFall")
    ap.add_argument("--fa-root", default=str(fa.DEFAULT_ROOT))
    ap.add_argument("--models", nargs="*", default=["rf", "xgb", "svm"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    for m in a.models:
        if MODELS[m][1] != "features":
            sys.exit(f"{m}: only feature models are supported here")

    split = json.loads((OUT / "split.json").read_text())
    um, UX, _ = collect(Path(a.uma_root))
    dev = um.subject.isin(split["dev_subjects"]).values
    assert not set(split["test_subjects"]) & set(um.subject[dev])
    print(f"UMAFall training: {dev.sum()} segments from dev subjects "
          f"(test subjects {split['test_subjects']} not used)")

    models = {m: MODELS[m][0]().fit(UX[dev], um.label.values[dev], um.subject.values[dev])
              for m in ["threshold"] + a.models}

    rows = []
    for matched in (False, True):
        tag = "matched" if matched else "native"
        meta, X = fallalld_segments(a.fa_root, matched, a.limit)
        if a.limit:   # smoke test: do not write the real split
            sel = sorted(meta.subject.unique())[::2]
            fsplit = dict(select=sel, report=[s for s in meta.subject.unique() if s not in sel])
        else:
            fsplit = split_fallalld(meta)
        print(f"\nFallAllD {tag}: {len(meta)} segments, {meta.subject.nunique()} subjects  "
              f"select={fsplit['select']}  report={fsplit['report']}")
        for name, m in models.items():
            p = m.predict_proba(X)
            for half in ("select", "report"):
                k = meta.subject.isin(fsplit[half]).values
                rows.append(dict(model=name, path=tag, half=half,
                                 **score(meta[k].reset_index(drop=True), p[k])))

    R = pd.DataFrame(rows)
    R.round(4).to_csv(OUT / "fallalld_selection.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["auc_gated", "auc_all", "sys_sensitivity", "sys_fa_per_hour",
            "gate_falls", "gate_adls"]

    sel = R[(R.path == "native") & (R.half == "select") & (R.model != "threshold")]
    sel = sel.set_index("model")
    print("\nSELECTION half, native path (the choice is made on this table only):")
    print(sel[cols].round(3).to_string())
    rank = sel.auc_gated.fillna(sel.auc_all)   # all-records AUC only if no ADL passed the gate
    close = sel[rank >= rank.max() - TIE]
    pick = close.sys_fa_per_hour.idxmin()
    print(f"\nwithin {TIE} AUC of the best: {list(close.index)}  -> pick: {pick} "
          f"(fewest false alarms among those)")
    (OUT / "fallalld_pick.txt").write_text(pick + "\n")

    rep = R[R.half == "report"].set_index(["path", "model"])
    print("\nREPORT half (not used for the choice) -- the cross-dataset result:")
    print(rep[cols].round(3).to_string())
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
