import sqlite3
import json
import os
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).parent.parent / "data" / "risk.db"))


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scored_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            user_id         TEXT,
            date            TEXT,
            anomaly_score   REAL,
            sequence_score  REAL,
            rule_score      REAL,
            final_risk      REAL,
            risk_level      TEXT,
            confidence      REAL,
            triggered_rules TEXT,
            explanation     TEXT,
            session_data    TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def save_session(result: dict, session_data: dict = None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO scored_sessions
            (session_id, user_id, date, anomaly_score, sequence_score,
             rule_score, final_risk, risk_level, confidence,
             triggered_rules, explanation, session_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("session_id", session_data.get("session_id", "") if session_data else ""),
        result.get("user_id", session_data.get("user_id", "") if session_data else ""),
        result.get("date", session_data.get("date", "") if session_data else ""),
        result.get("anomaly_score", 0),
        result.get("sequence_score", 0),
        result.get("rule_score", 0),
        result.get("final_risk", 0),
        result.get("risk_level", ""),
        result.get("confidence", 0),
        json.dumps(result.get("triggered_rules", [])),
        json.dumps(result.get("explanation", {})),
        json.dumps(session_data or {}),
    ))
    conn.commit()
    conn.close()


def save_batch(rows: list):
    conn = get_conn()
    for row in rows:
        triggered = row.get("triggered_rules", [])
        if not isinstance(triggered, str):
            triggered = json.dumps(triggered)

        conn.execute("""
            INSERT INTO scored_sessions
                (session_id, user_id, date, anomaly_score, sequence_score,
                 rule_score, final_risk, risk_level, confidence,
                 triggered_rules, explanation, session_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("session_id", ""),
            row.get("user_id", ""),
            row.get("date", ""),
            row.get("anomaly_score", 0),
            row.get("sequence_score", 0),
            row.get("rule_score", 0),
            row.get("final_risk", 0),
            row.get("risk_level", ""),
            row.get("confidence", 0),
            triggered,
            row.get("explanation", ""),
            json.dumps({k: v for k, v in row.items()
                        if k not in ("anomaly_score", "sequence_score", "rule_score",
                                     "final_risk", "risk_level", "confidence",
                                     "triggered_rules", "explanation")}),
        ))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM scored_sessions").fetchone()[0]
    counts = {}
    for level in ("low", "medium", "high", "critical"):
        counts[level] = conn.execute(
            "SELECT COUNT(*) FROM scored_sessions WHERE risk_level = ?", (level,)
        ).fetchone()[0]
    conn.close()
    return {
        "total_sessions": total,
        "low_count":      counts["low"],
        "medium_count":   counts["medium"],
        "high_count":     counts["high"],
        "critical_count": counts["critical"],
    }


def get_top_risks(n: int = 10) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM scored_sessions ORDER BY final_risk DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_all_sessions() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM scored_sessions ORDER BY final_risk DESC"
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_session_by_id(session_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM scored_sessions WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def _row_to_dict(row) -> dict:
    d = dict(row)
    for field in ("triggered_rules", "explanation", "session_data"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
