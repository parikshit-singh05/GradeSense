"""
GradeSense — SQLite logging helpers
===================================
A tiny wrapper around the standard library `sqlite3` for persisting every
prediction the Streamlit app makes. Kept separate from main.py so it's
easy to test and to swap out later (e.g. for a real database).

DB schema (single table: predictions):
    id              INTEGER PRIMARY KEY
    created_at      TEXT    (ISO 8601 UTC)
    predicted_g3    REAL
    risk_tier       TEXT    ("Low" / "Medium" / "High")
    payload_json    TEXT    (full input row as JSON — guarantees the
                             log is self-describing even if feature
                             names change later)

Streamlit reruns the whole script on every input change, so a row is
written each time the user adjusts a widget. That gives a complete
audit trail; if it becomes noisy you can add debouncing here (e.g.
skip writes if the last row's payload is identical).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "app" / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    predicted_g3 REAL    NOT NULL,
    risk_tier    TEXT    NOT NULL,
    payload_json TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions(created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    """Open a connection with row-factory set so reads return dicts."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the table + index if they don't exist. Idempotent."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def log_prediction(predicted_g3: float, risk_tier: str, payload: dict[str, Any]) -> int:
    """Insert one prediction row. Returns the new row id."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, predicted_g3, risk_tier, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (created_at, predicted_g3, risk_tier, payload_json),
        )
        conn.commit()
        return cur.lastrowid


def read_history(limit: int = 20) -> list[dict[str, Any]]:
    """
    Return the most recent `limit` rows, newest first. Each row dict has
    keys: id, created_at, predicted_g3, risk_tier, payload (parsed dict).
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, predicted_g3, risk_tier, payload_json "
            "FROM predictions "
            "ORDER BY id DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        out.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "predicted_g3": r["predicted_g3"],
            "risk_tier": r["risk_tier"],
            "payload": payload,
        })
    return out


def count_rows() -> int:
    """Diagnostic — how many rows are in the table."""
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
