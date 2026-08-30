"""
tools/context_on_falls.py -- what does the context classifier output on fall
records, which it never saw in training?

The context classifier is trained on ADL records only, because a fall is a
TRANSITION between states, not a state: a 2.5 s window spanning a fall holds
some standing, an impact, and some lying, and there is no correct single
label for it. Falls are the motion stream's job.

But the classifier still RUNS on fall windows in the live pipeline, and it
has to output something. This measures what.

Why it matters: core/fusion.py halves the motion contribution when context
reports "seated hand activity" with confidence above 0.7. If that fires
during a real fall, the suppression rule built for demo scenario 1 works
against demo scenario 2. This quantifies how often, across every fall record,
so the decision is made on a number instead of one trace.

Impact time is located as the peak SVM, then windows are split into
pre-impact and post-impact.

    py -m tools.context_on_falls
    py -m tools.context_on_falls --csv falls_context.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import scan, iter_windows, TARGET_FS      # noqa: E402
from streams.context import ContextStream                   # noqa: E402

SUPPRESS_STATE = "seated hand activity"
SUPPRESS_CONF = 0.7


def impact_time(rec):
    """Seconds to the peak resultant acceleration."""
    svm = np.linalg.norm(np.asarray(rec.acc, dtype=float), axis=1)
    return float(int(np.argmax(svm)) / TARGET_FS)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Context classifier behaviour on fall records")
    ap.add_argument("--root", default=r"data\raw\UMAFall")
    ap.add_argument("--model", default="models/context.joblib")
    ap.add_argument("--csv", help="write per-record rows to this file")
    a = ap.parse_args(argv)

    ctx = ContextStream.load(a.model)

    pre = Counter()
    post = Counter()
    conf_by_state = defaultdict(list)
    suppress_windows = 0
    total_post = 0
    records_with_suppress = 0
    by_fall_type = defaultdict(lambda: [0, 0])   # type -> [suppress, records]
    rows = []

    n_rec = 0
    for rec in scan(a.root):
        if not rec.is_fall:
            continue
        n_rec += 1
        t_imp = impact_time(rec)
        rec_suppress = 0
        for w, m in iter_windows(rec):
            ctx.score(w)                       # score FIRST, then read state
            state, conf = ctx.last_state, ctx.last_confidence
            is_post = w["t"] >= t_imp
            (post if is_post else pre)[state] += 1
            conf_by_state[state].append(conf)
            if is_post:
                total_post += 1
                if state == SUPPRESS_STATE and conf > SUPPRESS_CONF:
                    suppress_windows += 1
                    rec_suppress += 1
        ftype = rec.activity
        by_fall_type[ftype][1] += 1
        if rec_suppress:
            records_with_suppress += 1
            by_fall_type[ftype][0] += 1
        rows.append((rec.path.name, rec.subject, ftype, round(t_imp, 2),
                     rec_suppress))

    if n_rec == 0:
        print("no fall records found")
        return

    print(f"{n_rec} fall records, context classifier trained on ADLs only\n")

    def show(title, counter):
        tot = sum(counter.values())
        print(f"  {title}  ({tot} windows)")
        for state, k in counter.most_common():
            c = conf_by_state[state]
            print(f"    {state:<22} {k:>5}  {k/tot*100:>5.1f}%   "
                  f"median conf {np.median(c):.2f}")

    show("BEFORE impact", pre)
    print()
    show("AFTER impact", post)

    print(f"\n  suppression rule (state == '{SUPPRESS_STATE}' and "
          f"confidence > {SUPPRESS_CONF}):")
    print(f"    post-impact windows affected: {suppress_windows} / {total_post} "
          f"({suppress_windows/max(total_post,1)*100:.1f}%)")
    print(f"    fall records with at least one: {records_with_suppress} / {n_rec} "
          f"({records_with_suppress/n_rec*100:.1f}%)")

    print(f"\n  by fall type:")
    for ftype in sorted(by_fall_type):
        s, tot = by_fall_type[ftype]
        print(f"    {ftype:<16} {s:>3} / {tot:<3} records affected "
              f"({s/max(tot,1)*100:>5.1f}%)")

    print(f"\n  Reading: the classifier has no 'fallen person' class, so it")
    print(f"  extrapolates post-impact windows to the nearest thing it knows.")
    print(f"  Where that lands on '{SUPPRESS_STATE}' above {SUPPRESS_CONF}")
    print(f"  confidence, the scenario-1 suppression rule would damp a real")
    print(f"  fall. Guard: skip the halving while MotionStream.time_since_impact")
    print(f"  is set.")

    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["file", "subject", "fall_type", "impact_t_s",
                         "suppress_windows"])
            wr.writerows(rows)
        print(f"\n  per-record rows -> {a.csv}")


if __name__ == "__main__":
    main()
