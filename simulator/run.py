"""
simulator/run.py -- replay SIMULATED scenarios through every stream and show
what each one says, window by window (Stage 5.2 / 5.5).

    py -m simulator.run --list
    py -m simulator.run hard_fall_long_lie
    py -m simulator.run all
    py -m simulator.run all --db      also save every run in Postgres (Stage 5.5)

For each scenario it writes
    testing/outputs/stage5/<name>_outputs.csv   one row per stream per window
    testing/outputs/stage5/<name>_truth.json    what really happened (never shown to streams)
    testing/outputs/figures/stage5_<name>.png   timeline: signal, each stream, truth

Times in outputs are window END times: the moment a stream's answer exists.
There is no fusion and no alert yet (Stage 7); "flagged" below only means
the motion score crossed its evaluation threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.pipeline import Pipeline  # noqa: E402
from data.loader import WINDOW_SEC  # noqa: E402
from simulator.scenarios import SCENARIOS, build  # noqa: E402

OUT = ROOT / "testing" / "outputs" / "stage5"

# which demo wearer (backend/db/init/03_seed.sql) each scenario belongs to
WEARER = {"normal_day": 2, "hard_fall_long_lie": 2, "fall_then_get_up": 3,
          "faint_collapse": 3, "false_alarm": 4, "hr_spike_no_fall": 4}
FIG = ROOT / "testing" / "outputs" / "figures"


def run(name, store=None):
    sc = build(name)
    pipe = Pipeline()
    rows = []
    for w in sc.windows():
        for r in pipe.step(w):
            r["t"] = round(r["t"] + WINDOW_SEC, 2)
            rows.append(r)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    flat = D.assign(extras=D.extras.apply(json.dumps))
    flat.to_csv(OUT / f"{name}_outputs.csv", index=False)
    (OUT / f"{name}_truth.json").write_text(json.dumps(
        dict(name=sc.name, title=sc.title, description=sc.description,
             simulated=True, seconds=round(sc.seconds, 1), truth=sc.truth,
             events=sc.events), indent=2))
    thr = getattr(pipe.streams.get("motion"), "threshold", None)
    summary = _summary(sc, D, thr)
    print(f"\n=== {sc.title}  [{name}, SIMULATED, {sc.seconds:.0f} s] ===")
    print("streams:\n" + pipe.describe())
    print(summary)
    _plot(sc, D, thr)
    if store is not None:
        wid = WEARER.get(name, 2)
        run_id = store.start_run(wid, name, truth=json.loads(
            (OUT / f"{name}_truth.json").read_text()), simulated=True)
        store.write(run_id, wid, rows)
        print(f"database: run {run_id} for wearer {wid}, {len(rows)} stream outputs")
    return D


def _ex(D, stream, key):
    s = D[D.stream == stream]
    return s.t.values, np.array([e.get(key) for e in s.extras], dtype=object)


def _summary(sc, D, thr):
    lines = []
    for e in sc.events:
        lines.append(f"truth: {e['what']} at {e['t']:.1f} s")
    m = D[D.stream == "motion"]
    if len(m):
        t_gate, gate = _ex(D, "motion", "gate_open")
        opened = t_gate[gate == True]                           # noqa: E712
        lines.append("motion: gate " + (f"first opened at {opened[0]:.1f} s" if len(opened)
                                        else "never opened"))
        i = int(m.score.values.argmax())
        lines.append(f"motion: peak score {m.score.values[i]:.2f} at {m.t.values[i]:.1f} s")
        if thr is not None:
            hit = m[m.score >= thr]
            lines.append(f"motion: score >= {thr:.3f} (evaluation threshold) "
                         + (f"first at {hit.t.values[0]:.1f} s" if len(hit) else "never"))
        _, still = _ex(D, "motion", "stillness")
        lines.append(f"motion: stillness at the end {float(still[-1] or 0):.2f}")
        _, st_s = _ex(D, "motion", "still_s")
        _, rec = _ex(D, "motion", "recovered")
        _, mv = _ex(D, "motion", "moved_after_impact_s")
        if len(st_s) and st_s[-1] is not None:
            moved = {True: "yes", False: "no", None: "no impact"}[rec[-1]]
            lines.append(f"movement: no real movement for the last {float(st_s[-1]):.0f} s; "
                         f"moved again after the impact: {moved} ({float(mv[-1] or 0):.0f} s of movement)")
    a = D[D.stream == "activity"]
    if len(a):
        t, st = _ex(D, "activity", "state")
        runs, prev = [], None
        for ti, s in zip(t, st):
            if s != prev:
                runs.append(f"{ti:.0f}s {s}")
                prev = s
        lines.append("activity: " + " -> ".join(runs[:12]) + (" ..." if len(runs) > 12 else ""))
    p = D[D.stream == "physiological"]
    if len(p):
        lines.append(f"physiological: mean quality {p.quality.mean():.2f}"
                     + ("  (placeholder until Stage 6)" if p.quality.max() == 0 else ""))
    return "\n".join(lines)


def _plot(sc, D, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_sig = np.arange(len(sc.acc)) / 50.0
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1.3, 1.3]))
    colours = {"fall": "#f4c7c3", "daily activity": "#e8e8e8", "synthetic": "#cfe2f3"}
    for a in ax:
        for seg in sc.truth:
            a.axvspan(seg["start"], seg["end"], color=colours.get(seg["kind"], "#eee"),
                      alpha=0.6, lw=0)
        for e in sc.events:
            a.axvline(e["t"], color="#c0392b", lw=1, ls=":")

    ax[0].plot(t_sig, np.linalg.norm(sc.acc, axis=1), lw=0.6, color="#333")
    ax[0].axhline(2.25, color="#c0392b", lw=0.8, ls="--")
    ax[0].set_ylabel("|a| (g)")
    ax[0].set_title(f"{sc.title}  (SIMULATED: real WEDA-FALL motion from unseen people, "
                    "synthetic still periods and heart rate)", fontsize=10)
    for seg in sc.truth:
        ax[0].text(seg["start"] + 0.3, ax[0].get_ylim()[1] * 0.92,
                   seg["label"].split(" (")[0][:28], fontsize=7, va="top")

    m = D[D.stream == "motion"]
    if len(m):
        _, still = _ex(D, "motion", "stillness")
        ax[1].step(m.t, m.score, where="post", label="motion score (impact)", color="#1f4e79")
        ax[1].step(m.t, np.asarray(still, float), where="post", ls="--", lw=1,
                   label="stillness (for fusion)", color="#7f8c8d")
        if thr is not None:
            ax[1].axhline(thr, color="#c0392b", lw=0.8, ls="--",
                          label=f"evaluation threshold {thr:.2f}")
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].legend(fontsize=7, loc="upper left")
        _, st_s = _ex(D, "motion", "still_s")
        if len(st_s) and st_s[0] is not None:
            tw = ax[1].twinx()
            tw.step(m.t, np.asarray(st_s, float), where="post", color="#e67e22", lw=1, ls=":")
            tw.set_ylabel("seconds without\nreal movement", color="#e67e22", fontsize=8)
            tw.tick_params(axis="y", colors="#e67e22", labelsize=7)
    ax[1].set_ylabel("motion")

    a = D[D.stream == "activity"]
    if len(a):
        _, st = _ex(D, "activity", "state")
        names = ["stationary", "ambulating", "seated hand activity", "lying/immobile", "unknown"]
        y = [names.index(s) if s in names else len(names) - 1 for s in st]
        ax[2].scatter(a.t, y, s=10, c=a.quality, cmap="Greens", vmin=0, vmax=1,
                      edgecolors="#888", linewidths=0.3)
        ax[2].set_yticks(range(len(names)))
        ax[2].set_yticklabels(names, fontsize=7)
    ax[2].set_ylabel("activity")

    ax[3].plot(t_sig, sc.hr, lw=0.8, color="#8e44ad")
    ax[3].set_ylabel("HR (bpm)\nsynthetic")
    ax[3].set_xlabel("time (s)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"stage5_{sc.name}.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--db", action="store_true", help="also save runs in Postgres")
    a = ap.parse_args(argv)
    if a.list:
        for k, f in SCENARIOS.items():
            print(f"{k:20s} {(f.__doc__ or '').strip()}")
        return
    store = None
    if a.db:
        from core.store import Store
        store = Store()
    for name in (SCENARIOS if a.name == "all" else [a.name]):
        run(name, store)
    if store is not None:
        store.close()
    print(f"\nsaved in {OUT} and {FIG}")


if __name__ == "__main__":
    main()
