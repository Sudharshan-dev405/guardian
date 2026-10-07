"""
core/watch.py -- follow live runs from the database AS a given user
(Stage 5.9). A terminal stand-in for the caregiver / admin live screen:
it only shows what that user is allowed to see.

    py -m core.watch --user 5            Caregiver 1 (sees Wearer 1 only)
    py -m core.watch --user 1            Admin (sees everyone)
    py -m core.watch --user 4            Wearer 3 (sees only themselves)

Stop with Ctrl+C. Start a live replay in another terminal:
    py -m simulator.live hard_fall_long_lie
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.store import as_user, connect  # noqa: E402

LATEST = """
SELECT DISTINCT ON (o.stream) o.stream, o.t_s, o.score, o.extras
FROM stream_outputs o WHERE o.run_id = %s
ORDER BY o.stream, o.t_s DESC
"""


def snapshot(uid, threshold):
    out = []
    with as_user(uid) as c:
        runs = c.execute(
            "SELECT r.id, u.full_name, r.scenario FROM runs r JOIN users u ON u.id = r.wearer_id "
            "WHERE r.status = 'live' ORDER BY r.id").fetchall()
        for run_id, who, scen in runs:
            rows = {s: (t, sc, ex) for s, t, sc, ex in c.execute(LATEST, (run_id,)).fetchall()}
            parts = [f"{who} [{scen}]"]
            if "motion" in rows:
                t, sc, ex = rows["motion"]
                rec = {True: "yes", False: "no", None: "-"}[ex.get("recovered")]
                flag = "  << FALL-LIKE IMPACT" if threshold and sc >= threshold else ""
                parts.append(f"t={t:5.1f}s motion {sc:.2f} no movement {ex.get('still_s', 0):4.0f}s "
                             f"moved again {rec}{flag}")
            if "activity" in rows:
                parts.append(f"activity {rows['activity'][2].get('state', '?')}")
            out.append("  |  ".join(parts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True, help="users.id (see 03_seed.sql)")
    ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args(argv)
    with connect() as c:
        row = c.execute("SELECT full_name, role FROM users WHERE id = %s", (a.user,)).fetchone()
    if row is None:
        sys.exit(f"no user {a.user}")
    try:
        from streams.motion import MotionStream
        thr = MotionStream.load("models/motion.joblib").threshold
    except Exception:                                         # noqa: BLE001
        thr = None
    print(f"watching as {row[0]} [{row[1]}]  (Ctrl+C to stop)")
    last = None
    try:
        while True:
            lines = snapshot(a.user, thr) or ["no live runs visible to this user"]
            if lines != last:
                print(time.strftime("%H:%M:%S"), "\n  " + "\n  ".join(lines), flush=True)
                last = lines
            time.sleep(a.every)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
