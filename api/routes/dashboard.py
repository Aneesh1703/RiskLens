"""
Dashboard Routes — real data from SQLite
=========================================
GET /dashboard/stats       → overview counts from DB
GET /dashboard/top-risks   → top-N riskiest sessions from DB
GET /dashboard/sessions    → all scored sessions
GET /dashboard/session/:id → single session by ID
GET /dashboard/budget      → GenAI API budget usage
"""

from fastapi import APIRouter, Request
from api.middleware.rate_limiter import limiter, get_budget_status
from api.db import get_stats, get_top_risks, get_all_sessions, get_session_by_id

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats")
@limiter.limit("60/minute")
def stats(request: Request):
    """Returns overall risk distribution stats from DB."""
    return get_stats()


@router.get("/top-risks")
@limiter.limit("60/minute")
def top_risks(request: Request, n: int = 10):
    """Returns top-N riskiest sessions from DB."""
    return get_top_risks(n)


@router.get("/sessions")
@limiter.limit("60/minute")
def all_sessions(request: Request):
    """Returns all scored sessions from DB."""
    return get_all_sessions()


@router.get("/session/{session_id}")
@limiter.limit("60/minute")
def session_detail(request: Request, session_id: str):
    """Returns a single scored session by ID."""
    session = get_session_by_id(session_id)
    if session is None:
        return {"error": "Session not found"}
    return session


@router.get("/budget")
@limiter.limit("60/minute")
def budget(request: Request):
    """Returns current GenAI API budget usage."""
    return get_budget_status()
