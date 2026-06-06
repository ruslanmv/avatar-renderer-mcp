"""
users_db.py – Minimal SQLite store for users, sessions, and render jobs.

Uses only the Python standard library (``sqlite3``) so it adds no dependencies.
A fresh connection is opened per operation (WAL mode) which is simple and safe
for the demo's request volume. For higher scale, swap this module for
Postgres/Supabase/Turso behind the same function signatures.

Schema:
    users        – one row per Hugging Face account that has signed in
    sessions     – opaque app session tokens issued to the browser
    render_jobs  – ownership + lifecycle of render jobs (one per /render-upload)

Author:
    Ruslan Magana Vsevolodovna
License:
    Apache-2.0
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hf_user_id      TEXT UNIQUE NOT NULL,
    hf_username     TEXT,
    hf_email        TEXT,
    hf_access_token TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token TEXT UNIQUE NOT NULL,
    expires_at    TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS render_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id       TEXT UNIQUE NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON render_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON render_jobs(user_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create tables if they don't exist (idempotent)."""
    with _connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(_SCHEMA)


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #

def get_user_by_hf_id(db_path: str, hf_user_id: str) -> Optional[dict]:
    """Return the user row for a Hugging Face account id, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE hf_user_id = ?", (hf_user_id,)
        ).fetchone()
        return dict(row) if row else None


def upsert_user(
    db_path: str,
    *,
    hf_user_id: str,
    hf_username: Optional[str],
    hf_email: Optional[str],
    hf_access_token: Optional[str],
) -> dict:
    """Insert or update a user keyed by their Hugging Face account id."""
    now = _utcnow()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (hf_user_id, hf_username, hf_email, hf_access_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(hf_user_id) DO UPDATE SET
                hf_username     = excluded.hf_username,
                hf_email        = excluded.hf_email,
                hf_access_token = excluded.hf_access_token,
                updated_at      = excluded.updated_at
            """,
            (hf_user_id, hf_username, hf_email, hf_access_token, now, now),
        )
        row = conn.execute(
            "SELECT * FROM users WHERE hf_user_id = ?", (hf_user_id,)
        ).fetchone()
        return dict(row)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def create_session(db_path: str, *, user_id: int, session_token: str, ttl_hours: int) -> None:
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=ttl_hours)).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (user_id, session_token, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (user_id, session_token, expires, now.isoformat()),
        )


def get_user_by_session(db_path: str, session_token: str) -> Optional[dict]:
    """Return the user row for a valid, unexpired session token, else None.

    The returned dict includes ``hf_access_token`` for server-side use; callers
    that return data to the browser must strip it (see /me in api.py).
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT u.* , s.expires_at AS _session_expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.session_token = ?
            """,
            (session_token,),
        ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["_session_expires_at"])
        if expires_at < datetime.now(timezone.utc):
            # Expired — clean it up and deny.
            conn.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
            return None
        user = dict(row)
        user.pop("_session_expires_at", None)
        return user


def delete_session(db_path: str, session_token: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))


# --------------------------------------------------------------------------- #
# Render jobs
# --------------------------------------------------------------------------- #

def record_job(db_path: str, *, user_id: int, job_id: str, status: str = "processing") -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO render_jobs (user_id, job_id, status, created_at) VALUES (?, ?, ?, ?)",
            (user_id, job_id, status, _utcnow()),
        )


def update_job_status(db_path: str, *, job_id: str, status: str, completed: bool = False) -> None:
    completed_at = _utcnow() if completed else None
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE render_jobs SET status = ?, completed_at = COALESCE(?, completed_at) WHERE job_id = ?",
            (status, completed_at, job_id),
        )


def get_job_owner(db_path: str, job_id: str) -> Optional[int]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT user_id FROM render_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return int(row["user_id"]) if row else None


def list_user_jobs(db_path: str, user_id: int, limit: int = 50) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT job_id, status, created_at, completed_at FROM render_jobs "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
