"""
simulator/live.py -- play a SIMULATED scenario in real time, as if the band
were sending it now, and save each window to the database the moment it is
scored (Stage 5.9). This is what the website will show live in Stage 8.

    py -m simulator.live hard_fall_long_lie              real time, saved under its demo wearer
    py -m simulator.live faint_collapse --speed 4        four times faster
    py -m simulator.live normal_day --no-db              print only
    py -m simulator.live fall_then_get_up --wearer 3     save under another demo wearer

While it runs, watch it from another terminal as a given user:
    py -m core.watch --user 5        (Caregiver 1)

Each saved window also sends a 'guardian_live' notification (ids only),
which a backend can use to push updates to the browser.
No fusion and no alerts yet (Stage 7).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.pipeline import Pipeline  # noqa: E402
from data.loader import WINDOW_SEC  # noqa: E402
from simulator.run import WEARER  # noqa: E402
from simulator.scenarios import SCENARIOS, build  # noqa: E402


def _line(rows, thr):
    by = {r["stream"]: r for r in rows}
    m, a = by.get("motion"), by.get("activity")
    parts = [f"t={rows[0]['t']:6.1f}s"]
    if m:
        ex = m["extras"]
        flag = "  << FALL-LIKE IMPACT" if thr is not None and m["score"] >= thr else ""
        rec = {True: "yes", False: "no", None: "-"}[ex.get("recovered")]
        parts.append(f"motion {m['score']:.2f}  no movement {ex.get('still_s', 0):5.1f}s  "
                     f"moved again {rec:3s}{flag}")
    if a:
        parts.append(f"activity {a['extras'].get('state', '?')}")
    return "  |  ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(SCENARIOS))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wearer", type=int, default=None)
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args(argv)

    sc = build(a.name)
    pipe = Pipeline()
    thr = getattr(pipe.streams.get("motion"), "threshold", None)
    wid = a.wearer or WEARER.get(a.name, 2)
    store = run_id = None
    if not a.no_db:
        from core.store import Store
        store = Store()
        truth = dict(name=sc.name, title=sc.title, simulated=True, truth=sc.truth,
                     events=sc.events, seconds=round(sc.seconds, 1))
        run_id = store.start_run(wid, sc.name, truth=truth, status="live",
                                 notes=f"live replay x{a.speed:g}")
    print(f"LIVE (SIMULATED): {sc.title}, {sc.seconds:.0f} s at x{a.speed:g}"
          + (f", run {run_id} for wearer {wid}" if run_id else ", not saved"))
    for e in sc.events:
        print(f"  (truth, hidden from the streams: {e['what']} at {e['t']:.1f} s)")

    start, status = time.monotonic(), "done"
    try:
        for w in sc.windows():
            t_end = w["t"] + WINDOW_SEC
            wait = start + t_end / a.speed - time.monotonic()
            if wait > 0:
                time.sleep(wait)                       # the window "arrives" now
            rows = pipe.step(w)
            for r in rows:
                r["t"] = round(r["t"] + WINDOW_SEC, 2)
            if store is not None:
                store.write(run_id, wid, rows)
                store.notify(run_id, wid, rows[0]["t"])
            print(_line(rows, thr), flush=True)
    except KeyboardInterrupt:
        status = "stopped"
        print("\nstopped")
    finally:
        if store is not None:
            store.end_run(run_id, status)
            store.close()
            print(f"run {run_id} marked {status}")


if __name__ == "__main__":
    main()
