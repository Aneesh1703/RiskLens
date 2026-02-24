import json
import os
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, inspect, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

# Fallback to local SQLite if DATABASE_URL is not set
DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).parent.parent / "data" / "risk.db"))

db_url = os.getenv("DATABASE_URL")
if db_url:
    # SQLAlchemy requires `postgresql://` instead of `postgres://`
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
else:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        # check_same_thread is needed only for SQLite
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ScoredSession(Base):
    __tablename__ = "scored_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    date = Column(String)
    anomaly_score = Column(Float, default=0.0)
    sequence_score = Column(Float, default=0.0)
    rule_score = Column(Float, default=0.0)
    final_risk = Column(Float, default=0.0)
    risk_level = Column(String, index=True)
    confidence = Column(Float, default=0.0)
    triggered_rules = Column(Text)
    explanation = Column(Text)
    session_data = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Initialize DB tables
Base.metadata.create_all(bind=engine)


def save_session(result: dict, session_data: dict = None):
    db = SessionLocal()
    try:
        db_session = ScoredSession(
            session_id=result.get("session_id", session_data.get("session_id", "") if session_data else ""),
            user_id=result.get("user_id", session_data.get("user_id", "") if session_data else ""),
            date=result.get("date", session_data.get("date", "") if session_data else ""),
            anomaly_score=result.get("anomaly_score", 0),
            sequence_score=result.get("sequence_score", 0),
            rule_score=result.get("rule_score", 0),
            final_risk=result.get("final_risk", 0),
            risk_level=result.get("risk_level", ""),
            confidence=result.get("confidence", 0),
            triggered_rules=json.dumps(result.get("triggered_rules", [])),
            explanation=json.dumps(result.get("explanation", {})),
            session_data=json.dumps(session_data or {})
        )
        db.add(db_session)
        db.commit()
    finally:
        db.close()


def save_batch(rows: list):
    db = SessionLocal()
    try:
        db_sessions = []
        for row in rows:
            triggered = row.get("triggered_rules", [])
            if not isinstance(triggered, str):
                triggered = json.dumps(triggered)

            explanation = row.get("explanation", {})
            if not isinstance(explanation, str):
                explanation = json.dumps(explanation)

            meta_data = {k: v for k, v in row.items()
                         if k not in ("anomaly_score", "sequence_score", "rule_score",
                                      "final_risk", "risk_level", "confidence",
                                      "triggered_rules", "explanation")}

            db_sessions.append(ScoredSession(
                session_id=row.get("session_id", ""),
                user_id=row.get("user_id", ""),
                date=row.get("date", ""),
                anomaly_score=row.get("anomaly_score", 0),
                sequence_score=row.get("sequence_score", 0),
                rule_score=row.get("rule_score", 0),
                final_risk=row.get("final_risk", 0),
                risk_level=row.get("risk_level", ""),
                confidence=row.get("confidence", 0),
                triggered_rules=triggered,
                explanation=explanation,
                session_data=json.dumps(meta_data)
            ))
        db.add_all(db_sessions)
        db.commit()
    finally:
        db.close()


def get_stats() -> dict:
    db = SessionLocal()
    try:
        total = db.query(ScoredSession).count()
        low_count = db.query(ScoredSession).filter(ScoredSession.risk_level == "low").count()
        medium_count = db.query(ScoredSession).filter(ScoredSession.risk_level == "medium").count()
        high_count = db.query(ScoredSession).filter(ScoredSession.risk_level == "high").count()
        critical_count = db.query(ScoredSession).filter(ScoredSession.risk_level == "critical").count()

        return {
            "total_sessions": total,
            "low_count": low_count,
            "medium_count": medium_count,
            "high_count": high_count,
            "critical_count": critical_count,
        }
    finally:
        db.close()


def get_top_risks(n: int = 10) -> list:
    db = SessionLocal()
    try:
        rows = db.query(ScoredSession).order_by(ScoredSession.final_risk.desc()).limit(n).all()
        return [_row_to_dict(r) for r in rows]
    finally:
        db.close()


def get_all_sessions() -> list:
    db = SessionLocal()
    try:
        rows = db.query(ScoredSession).order_by(ScoredSession.final_risk.desc()).all()
        return [_row_to_dict(r) for r in rows]
    finally:
        db.close()


def get_session_by_id(session_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        row = db.query(ScoredSession).filter(ScoredSession.session_id == session_id).order_by(ScoredSession.created_at.desc()).first()
        return _row_to_dict(row) if row else None
    finally:
        db.close()


def _row_to_dict(obj) -> dict:
    d = {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}
    
    if 'created_at' in d and d['created_at']:
        d['created_at'] = str(d['created_at'])

    for field in ("triggered_rules", "explanation", "session_data"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d

