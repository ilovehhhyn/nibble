"""Single-runner durable jobs on shared scratch; SQLite rollback journal, not WAL."""

import json
from pathlib import Path
import sqlite3
import time
import uuid

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "jobs.sqlite"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, spec TEXT NOT NULL,
                created REAL NOT NULL, started REAL, ended REAL, error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0)""")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, spec):
        job_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs(id,status,spec,created) VALUES(?,?,?,?)",
                (job_id, "queued", json.dumps(spec), time.time()),
            )
        return self.get(job_id)

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        result = dict(row)
        result["spec"] = json.loads(result["spec"])
        result["cancel_requested"] = bool(result["cancel_requested"])
        return result

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='running',started=? WHERE id=?",
                (time.time(), row["id"]),
            )
        return self.get(row["id"])

    def finish(self, job_id, status, error=None):
        if status not in TERMINAL:
            raise ValueError("invalid terminal status")
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,ended=?,error=? WHERE id=? AND status='running'",
                (status, time.time(), error, job_id),
            )

    def cancel(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE id=? AND status IN ('queued','running')",
                (job_id,),
            )
            db.execute(
                "UPDATE jobs SET status='cancelled',ended=? WHERE id=? AND status='queued'",
                (time.time(), job_id),
            )
        return self.get(job_id)

    def recover(self):
        # Never replay an optimizer step automatically after an unknown process failure.
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='interrupted',ended=?,error=? WHERE status='running'",
                (
                    time.time(),
                    "runner restarted; inspect artifacts before submitting a new run",
                ),
            )
