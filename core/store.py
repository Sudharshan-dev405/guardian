"""
core/store.py -- save pipeline outputs in Postgres and read them back as a
given person, with the database enforcing who may see what (Stage 5.3).

Needs the database running (docker compose up -d) and:  pip install "psycopg[binary]"

    py -m core.store --ping       can we connect, and what is in there
    py -m core.store --check      what each demo user is allowed to see

Two logins (see backend/db/init/02_access.sql):
    GUARDIAN_DB_URL      owner, used by the pipeline to write
    GUARDIAN_APP_DB_URL  the website's login, limited by role
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager

DB_URL = os.environ.get("GUARDIAN_DB_URL",
                        "postgresql://guardian:guardian_dev@localhost:5433/guardian")
APP_URL = os.environ.get("GUARDIAN_APP_DB_URL",
                         "postgresql://guardian_app:guardian_app_dev@localhost:5433/guardian")


def connect(url=None):
    import psycopg
    return psycopg.connect(url or DB_URL)


class Store:
    """Writer used by the pipeline / simulator."""

    def __init__(self, url=None):
        self.conn = connect(url)

    def start_run(self, wearer_id, scenario, truth=None, simulated=True, notes=None,
                  status="done") -> int:
        from psycopg.types.json import Jsonb
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM devices WHERE wearer_id = %s ORDER BY id LIMIT 1",
                        (wearer_id,))
            dev = cur.fetchone()
            cur.execute("INSERT INTO runs (wearer_id, device_id, scenario, simulated, truth, "
                        "notes, status) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (wearer_id, dev[0] if dev else None, scenario, simulated,
                         Jsonb(truth) if truth is not None else None, notes, status))
            run_id = cur.fetchone()[0]
        self.conn.commit()
        return run_id

    def write(self, run_id, wearer_id, rows):
        """rows: pipeline records {t, stream, score, quality, extras}."""
        from psycopg.types.json import Jsonb
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO stream_outputs (run_id, wearer_id, t_s, stream, score, quality, extras) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(run_id, wearer_id, r["t"], r["stream"], r["score"], r["quality"],
                  Jsonb(r["extras"])) for r in rows])
        self.conn.commit()

    def notify(self, run_id, wearer_id, t):
        """Tell listeners (the future backend) that a live run has new rows.
        Only ids go in the message; the data itself stays behind the access rules."""
        self.conn.execute("SELECT pg_notify('guardian_live', %s)",
                          (f'{{"run_id": {run_id}, "wearer_id": {wearer_id}, "t": {t}}}',))
        self.conn.commit()

    def end_run(self, run_id, status="done"):
        self.conn.execute("UPDATE runs SET status = %s, ended_at = now() WHERE id = %s",
                          (status, run_id))
        self.conn.commit()

    def close(self):
        self.conn.close()


@contextmanager
def as_user(user_id, url=None):
    """Connection that sees only what user_id is allowed to see."""
    conn = connect(url or APP_URL)
    try:
        with conn.transaction():
            conn.execute("SELECT set_config('app.user_id', %s, true)", (str(user_id),))
            yield conn
    finally:
        conn.close()


def ping():
    with connect() as c:
        for t in ("users", "caregiver_links", "devices", "runs", "stream_outputs", "alerts"):
            n = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"  {t:16s} {n}")


def check():
    with connect() as c:
        users = c.execute("SELECT id, full_name, role FROM users ORDER BY id").fetchall()
    print(f"{'user':40s} {'people':>6s} {'runs':>5s} {'outputs':>8s}  wearers visible")
    for uid, name, role in users:
        with as_user(uid) as c:
            people = c.execute("SELECT count(*) FROM users").fetchone()[0]
            runs = c.execute("SELECT count(*) FROM runs").fetchone()[0]
            outs = c.execute("SELECT count(*) FROM stream_outputs").fetchone()[0]
            seen = [r[0] for r in c.execute(
                "SELECT DISTINCT wearer_id FROM runs ORDER BY 1").fetchall()]
        print(f"{name + ' [' + role + ']':40s} {people:6d} {runs:5d} {outs:8d}  {seen}")
    with as_user(0) as c:                                    # nobody logged in
        n = c.execute("SELECT count(*) FROM stream_outputs").fetchone()[0]
    print(f"{'(no user set)':40s} {'':6s} {'':5s} {n:8d}  <- must be 0")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.ping:
        ping()
    if a.check:
        check()
    if not (a.ping or a.check):
        ap.print_help()


if __name__ == "__main__":
    main()
