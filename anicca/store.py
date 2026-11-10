"""SQLite applicant store shared by the Streamlit app and the webhook server.

Set ANICCA_DB to choose the file. On Streamlit Community Cloud the disk is
reset on redeploy; move to Postgres/Supabase before real pilots.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS applicants (
    id TEXT PRIMARY KEY,
    fingerprint TEXT,
    name TEXT, email TEXT, phone TEXT, linkedin_url TEXT,
    job TEXT,
    source TEXT, source_ref TEXT,
    received_at TEXT,
    raw_text TEXT,
    parsed_json TEXT,
    integrity_json TEXT,
    status TEXT DEFAULT 'New',
    duplicate_sources TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_fp ON applicants(fingerprint);
CREATE INDEX IF NOT EXISTS ix_email ON applicants(email);
CREATE TABLE IF NOT EXISTS audit_log (
    ts TEXT, actor TEXT, applicant_id TEXT, action TEXT, detail TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("ANICCA_DB", os.path.join("data", "anicca.db"))
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with _LOCK:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------ writes
    def find_duplicate(self, fingerprint: str, email: str, job: str) -> Optional[str]:
        q = "SELECT id FROM applicants WHERE fingerprint = ?"
        args: list = [fingerprint]
        if email:
            q += " OR (lower(email) = lower(?) AND job = ?)"
            args += [email, job]
        row = self.conn.execute(q, args).fetchone()
        return row["id"] if row else None

    def add_duplicate_source(self, applicant_id: str, source: str) -> None:
        with _LOCK:
            row = self.conn.execute("SELECT duplicate_sources FROM applicants WHERE id=?", [applicant_id]).fetchone()
            srcs = json.loads(row["duplicate_sources"] or "[]") if row else []
            srcs.append({"source": source, "at": now_iso()})
            self.conn.execute("UPDATE applicants SET duplicate_sources=? WHERE id=?", [json.dumps(srcs), applicant_id])
            self.conn.commit()

    def insert(self, rec: dict) -> str:
        aid = rec.get("id") or uuid.uuid4().hex[:12]
        parsed = rec.get("parsed", {})
        with _LOCK:
            self.conn.execute(
                """INSERT INTO applicants (id, fingerprint, name, email, phone, linkedin_url, job, source,
                   source_ref, received_at, raw_text, parsed_json, integrity_json, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [aid, rec.get("fingerprint"), rec.get("name") or parsed.get("name", ""),
                 rec.get("email") or parsed.get("email", ""), rec.get("phone") or parsed.get("phone", ""),
                 rec.get("linkedin_url") or parsed.get("linkedin_url", ""), rec.get("job", ""),
                 rec.get("source", "upload"), rec.get("source_ref", ""), rec.get("received_at") or now_iso(),
                 rec.get("raw_text", ""), json.dumps(parsed), json.dumps(rec.get("integrity", {})), "New"],
            )
            self.conn.commit()
        self.log("system", aid, "ingested", rec.get("source", "upload"))
        return aid

    def set_status(self, applicant_id: str, status: str, actor: str = "recruiter", note: str = "") -> None:
        with _LOCK:
            self.conn.execute("UPDATE applicants SET status=? WHERE id=?", [status, applicant_id])
            self.conn.commit()
        self.log(actor, applicant_id, f"status:{status}", note)

    def log(self, actor: str, applicant_id: str, action: str, detail: str = "") -> None:
        with _LOCK:
            self.conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?)", [now_iso(), actor, applicant_id, action, detail])
            self.conn.commit()

    def delete(self, applicant_id: str) -> None:
        """Erase an applicant's data (data-subject deletion request). The audit row keeps only the id."""
        with _LOCK:
            self.conn.execute("DELETE FROM applicants WHERE id=?", [applicant_id])
            self.conn.commit()
        self.log("recruiter", applicant_id, "erased", "candidate data deleted on request")

    def clear(self) -> None:
        with _LOCK:
            self.conn.execute("DELETE FROM applicants")
            self.conn.commit()
        self.log("recruiter", "*", "cleared", "all applicants removed")

    # ------------------------------------------------------------ reads
    def jobs(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT job FROM applicants WHERE job<>'' ORDER BY job")]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]

    def to_frame(self, job: Optional[str] = None) -> pd.DataFrame:
        q, args = "SELECT * FROM applicants", []
        if job:
            q += " WHERE job = ?"
            args.append(job)
        rows = [dict(r) for r in self.conn.execute(q + " ORDER BY received_at DESC", args)]
        for r in rows:
            r["parsed"] = json.loads(r.pop("parsed_json") or "{}")
            r["integrity"] = json.loads(r.pop("integrity_json") or "{}")
            r["duplicate_sources"] = json.loads(r.get("duplicate_sources") or "[]")
        return pd.DataFrame(rows)

    def audit_frame(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM audit_log ORDER BY ts DESC", self.conn)
