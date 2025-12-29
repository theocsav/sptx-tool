import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .settings import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                run_dir TEXT,
                output_dir TEXT,
                config_path TEXT,
                job_id TEXT,
                message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_queue (
                run_id INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                submit INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_queue)").fetchall()}
        if "submit" not in columns:
            conn.execute("ALTER TABLE run_queue ADD COLUMN submit INTEGER NOT NULL DEFAULT 1")
        conn.commit()


def create_run(run_name: str, status: str = "created") -> int:
    created = utc_now()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (run_name, status, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_name, status, created, created),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_run(run_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    columns = ", ".join([f"{key} = ?" for key in fields.keys()])
    values = list(fields.values()) + [run_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE runs SET {columns} WHERE id = ?", values)
        conn.commit()


def fetch_run(run_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def list_runs() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM runs ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def enqueue_run(run_id: int, submit: bool = True) -> None:
    created = utc_now()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO run_queue (run_id, state, submit, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, "queued", 1 if submit else 0, created, created),
        )
        conn.commit()


def next_queued_run() -> Optional[Dict[str, int]]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT run_id, submit FROM run_queue WHERE state = ? ORDER BY created_at ASC LIMIT 1",
            ("queued",),
        )
        row = cur.fetchone()
    return {"run_id": int(row["run_id"]), "submit": int(row["submit"])} if row else None


def mark_run_running(run_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE run_queue SET state = ?, updated_at = ? WHERE run_id = ?",
            ("running", utc_now(), run_id),
        )
        conn.commit()


def remove_from_queue(run_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM run_queue WHERE run_id = ?", (run_id,))
        conn.commit()
