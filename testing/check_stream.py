"""
testing/check_stream.py -- acceptance check for any stream module (Stage 5.7).

Run it on a module before it joins the pipeline (Stage 6: Mithuna's
physiological module, Sudharshan's activity module). It checks the rules in
contract.py with made-up windows, so it needs no dataset.

    py -m testing.check_stream physiological            (a name from core/pipeline.py STREAMS)
    py -m testing.check_stream motion
    py -m testing.check_stream streams.physiological:PhysiologicalStream --model models/x.joblib
    py -m testing.check_stream all

Checks
    1 loads        imports, builds (and loads its model, if any)
    2 contract     is a contract.Stream
    3 own module   does not import another stream (contract rule)
    4 windows      on 10 normal and awkward windows: no crash, score and
                   last_quality are numbers in [0, 1]
    4b damaged in  empty, too-short or NaN windows must not get full quality
    5 long run     5 minutes of windows in a row: still within the rules
    6 speed        mean time per window (budget: windows arrive every 1.24 s,
                   shared by all streams)
    7 pipeline     plugs into core/pipeline.py and its extras save as JSON

Result: PASS / WARN / FAIL per check, saved to
testing/outputs/stage5/check_<name>.txt. Exit code 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contract import Stream  # noqa: E402
from core.pipeline import STREAMS, Pipeline  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage5"
FS, N, HOP = 50, 125, 62
SPEED_PASS_MS, SPEED_WARN_MS = 50.0, 200.0


def _window(t=0.0, kind="normal", rng=np.random.default_rng(0)):
    acc = np.column_stack([rng.normal(0, .05, N), rng.normal(0, .05, N),
                           1 + rng.normal(0, .05, N)])
    gyro = rng.normal(0, 10, (N, 3))
    hr, temp = 75.0, 33.2
    if kind == "no_hr":
        hr = temp = None
    elif kind == "no_gyro":
        gyro = None
    elif kind == "empty":
        acc, gyro = np.zeros((0, 3)), np.zeros((0, 3))
    elif kind == "short":
        acc, gyro = acc[:10], gyro[:10]
    elif kind == "nan":
        acc[40:50] = np.nan
    elif kind == "huge":
        acc[60:65] = 16.0
    elif kind == "hr_high":
        hr = 220.0
    elif kind == "hr_low":
        hr = 25.0
    elif kind == "impact":
        acc[60:66] += np.array([4.0, 3.0, 5.0])
    return {"t": float(t), "acc": acc, "gyro": gyro, "hr": hr, "temp": temp, "fs": FS}


DAMAGED = ("empty", "short", "nan")
CASES = ("normal", "no_hr", "no_gyro", "empty", "short", "nan", "huge",
         "hr_high", "hr_low", "impact")


def _resolve(target, model):
    if target in STREAMS:
        mod, cls, m = STREAMS[target]
        return target, mod, cls, model or m
    mod, cls = target.split(":")
    return cls, mod, cls, model


def _ok01(x):
    return isinstance(x, (int, float, np.floating)) and math.isfinite(x) and 0.0 <= x <= 1.0


def check(target, model=None):
    name, mod, cls, model = _resolve(target, model)
    res, lines = [], []

    def add(check_name, status, detail):
        res.append(status)
        lines.append(f"[{status:4s}] {check_name:12s} {detail}")

    # 1 loads
    try:
        C = getattr(importlib.import_module(mod), cls)
        path = ROOT / model if model else None
        s = C.load(str(path)) if path is not None and path.exists() else C()
        how = f"model {model}" if path is not None and path.exists() else \
            ("no model file found, untrained" if model else "no model")
        add("loads", "PASS" if not (model and how.startswith("no model file")) else "WARN",
            f"{mod}.{cls} ({how})")
    except Exception as e:                                       # noqa: BLE001
        add("loads", "FAIL", f"{type(e).__name__}: {e}")
        return name, res, lines

    # 2 contract
    add("contract", "PASS" if isinstance(s, Stream) else "FAIL",
        "subclass of contract.Stream" if isinstance(s, Stream) else "not a contract.Stream")

    # 3 imports another stream?
    src = inspect.getsource(importlib.import_module(mod))
    own = mod.split(".")[-1]
    other = sorted({m for m in re.findall(r"^\s*(?:from|import)\s+streams\.(\w+)", src, re.M)
                    if m != own})
    add("own module", "FAIL" if other else "PASS",
        f"imports other streams: {other}" if other else "imports no other stream")

    # 4 awkward windows (fresh instance each, so one case cannot hide another)
    bad, table = [], []
    for k, case in enumerate(CASES):
        try:
            si = C.load(str(ROOT / model)) if model and (ROOT / model).exists() else C()
            sc = si.score(_window(t=0.0, kind=case))
            q = getattr(si, "last_quality", None)
            fine = _ok01(sc) and _ok01(q)
            table.append(f"    {case:9s} score {sc!s:>8.8s}  quality {q!s:>8.8s}"
                         + ("" if fine else "   <- breaks the rules"))
            if not fine:
                bad.append(case)
        except Exception as e:                                   # noqa: BLE001
            table.append(f"    {case:9s} CRASH {type(e).__name__}: {e}")
            bad.append(case)
    add("windows", "FAIL" if bad else "PASS",
        f"problems in: {bad}" if bad else f"all {len(CASES)} cases within the rules")
    lines.extend(table)
    # damaged input should not be reported at full quality ("gaps as gaps")
    full = [c for c, row in zip(CASES, table)
            if c in DAMAGED and re.search(r"quality\s+1\.0\b", row)]
    add("damaged in", "WARN" if full else "PASS",
        f"full quality despite damaged input: {full}" if full
        else "quality drops for empty, short or NaN windows")

    # 5 long run + 6 speed
    try:
        sl = C.load(str(ROOT / model)) if model and (ROOT / model).exists() else C()
        times, broke = [], 0
        n_win = int(300 / (HOP / FS))
        for i in range(n_win):
            w = _window(t=i * HOP / FS, kind="impact" if i == n_win // 2 else "normal")
            t0 = time.perf_counter()
            sc = sl.score(w)
            times.append((time.perf_counter() - t0) * 1000)
            broke += not (_ok01(sc) and _ok01(getattr(sl, "last_quality", None)))
        add("long run", "FAIL" if broke else "PASS",
            f"{n_win} windows (5 min), {broke} broke the rules")
        ms = float(np.mean(times))
        st = "PASS" if ms < SPEED_PASS_MS else "WARN" if ms < SPEED_WARN_MS else "FAIL"
        add("speed", st, f"{ms:.2f} ms per window on average, worst {max(times):.1f} ms "
                         f"(pass < {SPEED_PASS_MS:.0f} ms)")
    except Exception as e:                                       # noqa: BLE001
        add("long run", "FAIL", f"crashed: {type(e).__name__}: {e}")

    # 7 pipeline
    try:
        p = Pipeline(streams={name: (mod, cls, model)})
        rec = p.step(_window())[0]
        json.dumps(rec)
        keys = sorted(rec["extras"])
        add("pipeline", "PASS" if "error" not in rec["extras"] else "FAIL",
            f"extras stored: {keys if keys else 'none'}")
    except Exception as e:                                       # noqa: BLE001
        add("pipeline", "FAIL", f"{type(e).__name__}: {e}")
    return name, res, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="STREAMS name, module:Class, or all")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    targets = list(STREAMS) if a.target == "all" else [a.target]
    failed = False
    OUT.mkdir(parents=True, exist_ok=True)
    for tg in targets:
        name, res, lines = check(tg, a.model)
        verdict = "FAIL" if "FAIL" in res else "WARN" if "WARN" in res else "PASS"
        text = f"=== {name}: {verdict} ===\n" + "\n".join(lines)
        print(text + "\n")
        (OUT / f"check_{name}.txt").write_text(text + "\n")
        failed |= verdict == "FAIL"
    print(f"saved in {OUT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
