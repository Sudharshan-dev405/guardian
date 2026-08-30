"""
tools/trace.py -- replay one dataset record through the trained context and
motion streams and print what each produces per window.

This is a debugging and demo-selection tool, not part of the pipeline.

    py -m tools.trace data\\raw\\UMAFall\\<file>.csv
    py -m tools.trace --glob "*Fall_forwardFall*"
    py -m tools.trace --glob "*ADL_Aplausing*" --root data\\raw\\UMAFall

The SUPPRESS column is the important one. core/fusion.py halves the motion
contribution when context reports "seated hand activity" with confidence
above 0.7. A window marked SUPPRESS while the motion score is high is a
window where a real fall would be damped -- that is a conflict between demo
scenario 1 and demo scenario 2, and it is worth knowing about before the
rule is wired.
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import read_umafall, iter_windows          # noqa: E402
from streams.context import ContextStream                    # noqa: E402
from streams.motion import MotionStream                      # noqa: E402

SUPPRESS_STATE = "seated hand activity"
SUPPRESS_CONF = 0.7


def trace(path, ctx_model="models/context.joblib", mot_model="models/motion.joblib"):
    path = Path(path)
    ctx = ContextStream.load(ctx_model)
    mot = MotionStream.load(mot_model)
    rec = read_umafall(path)

    print(f"\n{path.name}")
    print(f"  activity={rec.activity}  is_fall={rec.is_fall}  "
          f"true_state={rec.state}  subject={rec.subject}")
    print(f"\n{'t':>6} {'ctx state':<22} {'conf':>5} {'cscore':>6} "
          f"{'motion':>6} {'imp':>5} {'still':>5} {'mq':>5}  flag")
    print("-" * 78)

    conflicts = 0
    for w, _meta in iter_windows(rec):
        # Score FIRST, then read the attributes. Reading last_state before
        # calling score() gives the previous window's value.
        cs = ctx.score(w)
        state, conf = ctx.last_state, ctx.last_confidence
        ms = mot.score(w)
        imp, still, mq = mot.last_impact, mot.last_stillness, mot.last_quality

        suppress = (state == SUPPRESS_STATE and conf > SUPPRESS_CONF)
        flag = ""
        if suppress and ms > 0.3:
            flag = "SUPPRESS + high motion  <-- CONFLICT"
            conflicts += 1
        elif suppress:
            flag = "SUPPRESS"
        elif state == "unknown":
            flag = "open-set reject"

        print(f"{w['t']:>6.2f} {state:<22} {conf:>5.2f} {cs:>6.2f} "
              f"{ms:>6.2f} {imp:>5.2f} {still:>5.2f} {mq:>5.2f}  {flag}")

    print("-" * 78)
    if conflicts:
        print(f"{conflicts} window(s) where the fusion suppression rule would damp "
              f"an active motion event.")
        if rec.is_fall:
            print("This record IS a fall. The scenario-1 rule would work against "
                  "scenario 2 here.")
    else:
        print("no suppression/motion conflicts in this record")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trace one record through both streams")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--glob", help="pattern to pick the first matching file")
    ap.add_argument("--root", default="data/raw/UMAFall")
    ap.add_argument("--context", default="models/context.joblib")
    ap.add_argument("--motion", default="models/motion.joblib")
    ap.add_argument("--all", action="store_true",
                    help="trace every file matching --glob, summary only")
    a = ap.parse_args(argv)

    if a.path:
        files = [a.path]
    elif a.glob:
        files = sorted(globmod.glob(str(Path(a.root) / a.glob)))
        if not files:
            print(f"nothing matched {a.glob} under {a.root}")
            return
        if not a.all:
            files = files[:1]
    else:
        ap.print_help()
        return

    if a.all:
        ctx = ContextStream.load(a.context)
        mot = MotionStream.load(a.motion)
        print(f"{'file':<58} {'peak':>5} {'conflicts':>9}")
        for f in files:
            try:
                rec = read_umafall(Path(f))
            except Exception:                                # noqa: BLE001
                continue
            mot.reset()
            peak, conf_n = 0.0, 0
            for w, _ in iter_windows(rec):
                ctx.score(w)
                st, cf = ctx.last_state, ctx.last_confidence
                ms = mot.score(w)
                peak = max(peak, ms)
                if st == SUPPRESS_STATE and cf > SUPPRESS_CONF and ms > 0.3:
                    conf_n += 1
            print(f"{Path(f).name:<58} {peak:>5.2f} {conf_n:>9}")
        return

    for f in files:
        trace(f, a.context, a.motion)


if __name__ == "__main__":
    main()
