"""
testing/module_report.py -- one report on where each module stands
(Stage 5.6), built only from saved results. Nothing is retrained or re-tested.

    py -m simulator.run all          (first, so the scenario results exist)
    py -m testing.check_stream all   (optional, adds the acceptance check verdicts)
    py -m testing.module_report

Writes
    testing/outputs/stage5/module_report.txt
    testing/outputs/figures/stage5_scenarios_overview.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTS = ROOT / "testing" / "outputs"
S5, FIG = OUTS / "stage5", OUTS / "figures"
SCEN = ["normal_day", "hard_fall_long_lie", "fall_then_get_up", "faint_collapse",
        "false_alarm", "hr_spike_no_fall"]

# Activity numbers come from the activity module's own evaluation (UMAFall,
# subject-disjoint holdout) and the FallAllD transfer check. Kept here as text
# because that module is being replaced in Stage 6.
ACTIVITY = [
    "Four states + 'unknown' (open-set reject), 15 features, random forest,",
    "temperature-scaled confidence.",
    "UMAFall, people not seen in training: macro F1 0.71, accuracy 0.73",
    "  per state: ambulating 0.90, seated hand activity 0.75, lying/immobile 0.70,",
    "  stationary 0.51 (sitting vs standing is mostly forearm tilt at the wrist)",
    "FallAllD, unchanged model: macro F1 0.22, calibration error 0.377 (confidently wrong)",
    "Stage 5 simulation: flips state every 1-3 s on WEDA-FALL data.",
    "STATUS: being replaced by Sudharshan's activity module in Stage 6.",
]


def _scenario(name, thr):
    f, tj = S5 / f"{name}_outputs.csv", S5 / f"{name}_truth.json"
    if not (f.exists() and tj.exists()):
        return None
    D = pd.read_csv(f)
    truth = json.loads(tj.read_text())
    m = D[D.stream == "motion"].copy()
    ex = m.extras.apply(json.loads)
    impact = truth["events"][0]["t"] if truth["events"] else None
    gate = m.t[ex.apply(lambda e: bool(e.get("gate_open")))]
    flag = m.t[m.score >= thr] if thr is not None else pd.Series(dtype=float)
    last = ex.iloc[-1]
    return dict(scenario=name, title=truth["title"], seconds=truth["seconds"],
                impact=impact,
                gate=float(gate.iloc[0]) if len(gate) else None,
                flagged=float(flag.iloc[0]) if len(flag) else None,
                peak=float(m.score.max()),
                still_end=float(last.get("still_s") or 0),
                recovered=last.get("recovered"))


def _fmt(x, spec=".1f", none="-"):
    return none if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def main():
    cfg_p = OUTS / "stage4" / "final_config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    thr = cfg.get("threshold")
    L = ["GUARDIAN - MODULE REPORT (Stage 5)", "=" * 78, ""]

    # ---------------- motion
    p = cfg.get("params", {})
    L += ["MOTION MODULE (Pranavah)", "-" * 78,
          "What it does: gate (|a| > 2.25 g), then XGBoost scores the 2 s before and",
          "3 s after the impact (0 to 1). Also reports stillness and a movement tracker",
          "(seconds without real movement, moved again after the impact).",
          f"Model now: trained on {cfg.get('trained_on', '?')}",
          f"  depth {p.get('max_depth')}, {p.get('n_estimators')} trees, lambda "
          f"{p.get('reg_lambda')}, {cfg.get('n_segments')} segments, {cfg.get('n_falls')} falls, "
          f"evaluation threshold {_fmt(thr, '.3f')}", ""]
    t6 = OUTS / "stage35" / "step6_test.csv"
    if t6.exists():
        T = pd.read_csv(t6)
        T = T[T.model.str.startswith("WEDA")]
        L += ["One-time test (Stage 3.5, locked test people; this was the 1.5 s version):",
              f"  {'test set':12s} {'seen?':26s} {'AUC':>6s} {'falls caught':>13s} {'false alarms/h':>15s}"]
        for _, r in T.iterrows():
            L.append(f"  {r.test:12s} {r.seen:26s} {r.auc_gated:6.3f} {r.falls_caught:13.0%} "
                     f"{r.false_alarms_per_hour:15.1f}")
        L += ["Stage 4: looking 3 s after the impact instead of 1.5 s was kept; on all",
              "  recordings it is a tie (0.809 vs 0.811), so no gain is claimed. The 3 s",
              "  version has dev numbers only: the test parts were already used.",
              "Limits: thresholds tuned on one device do not carry over to another; the",
              "  real band needs its own. Scores swing up to 0.05 with 18 training people.", ""]

    # ---------------- activity
    L += ["ACTIVITY MODULE (current version)", "-" * 78] + ACTIVITY + [""]

    # ---------------- physiological
    L += ["PHYSIOLOGICAL MODULE (Mithuna)", "-" * 78,
          "Placeholder: returns 0 with quality 0 until Stage 6. Heart rate in the",
          "scenarios is synthetic.", ""]

    # ---------------- acceptance checks
    checks = sorted(S5.glob("check_*.txt"))
    if checks:
        L += ["ACCEPTANCE CHECK (testing/check_stream.py)", "-" * 78]
        for c in checks:
            lines = c.read_text().splitlines()
            L.append("  " + lines[0].strip("= "))
            L += ["    " + x for x in lines[1:] if x.startswith("[WARN]") or x.startswith("[FAIL]")]
        L.append("")

    # ---------------- scenarios
    rows = [r for r in (_scenario(n, thr) for n in SCEN) if r is not None]
    if rows:
        L += ["SIMULATED SCENARIOS (real WEDA-FALL motion from unseen people,",
              "synthetic still periods and heart rate; a demonstration, not an evaluation)",
              "-" * 78,
              f"  {'scenario':20s} {'fall at':>8s} {'flagged':>8s} {'delay':>6s} {'peak':>5s} "
              f"{'still at end':>12s} {'moved again':>12s}"]
        for r in rows:
            delay = (r["flagged"] - r["impact"]) if r["flagged"] is not None and r["impact"] is not None else None
            mv = {True: "yes", False: "no", None: "no impact"}[r["recovered"]]
            L.append(f"  {r['scenario']:20s} {_fmt(r['impact']):>8s} {_fmt(r['flagged']):>8s} "
                     f"{_fmt(delay):>6s} {r['peak']:5.2f} {r['still_end']:10.0f} s {mv:>12s}")
        L += ["  (times in seconds; 'flagged' = motion score above the evaluation threshold;",
              "   alerts come from fusion in Stage 7, not from this)", ""]
        _plot(rows)

    S5.mkdir(parents=True, exist_ok=True)
    (S5 / "module_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved {S5 / 'module_report.txt'}\nfigure: {FIG / 'stage5_scenarios_overview.png'}")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(rows) + 1.4))
    for i, r in enumerate(rows[::-1]):
        y = i
        ax.barh(y, r["seconds"], color="#ececec", height=0.5)
        if r["still_end"] > 0:
            ax.barh(y, r["still_end"], left=r["seconds"] - r["still_end"], color="#f5cba7",
                    height=0.5)
        if r["impact"] is not None:
            ax.plot(r["impact"], y, "v", color="#c0392b", ms=9)
        if r["flagged"] is not None:
            ax.plot(r["flagged"], y, "o", color="#1f4e79", ms=7)
            note = f"flagged {r['flagged'] - r['impact']:.1f} s after the fall" if r["impact"] else "flagged"
        else:
            note = "not flagged"
        mv = {True: ", moved again", False: ", did not move again", None: ""}[r["recovered"]]
        ax.text(r["seconds"] + 1, y, note + mv, va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["title"] for r in rows[::-1]], fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, max(r["seconds"] for r in rows) * 1.45)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], marker="v", ls="", color="#c0392b", label="fall (truth)"),
                       Line2D([], [], marker="o", ls="", color="#1f4e79", label="motion flags it"),
                       Patch(color="#f5cba7", label="no real movement until the end")],
              fontsize=7, loc="lower right")
    ax.set_title("Simulated scenarios: what the motion stream saw (SIMULATED data)", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stage5_scenarios_overview.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
