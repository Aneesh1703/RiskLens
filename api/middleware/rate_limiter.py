import os
import time
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

DAILY_GEMINI_LIMIT = int(os.getenv("DAILY_GEMINI_LIMIT", "70"))
_daily_counter = 0
_day_start = time.time()


def check_gemini_budget():
    """Raises 429 if daily Gemini budget is exhausted."""
    global _daily_counter, _day_start

    now = time.time()
    if now - _day_start > 86400:
        _daily_counter = 0
        _day_start = now

    if _daily_counter >= DAILY_GEMINI_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily GenAI budget exhausted ({DAILY_GEMINI_LIMIT} calls). Resets in {int(86400 - (now - _day_start))}s."
        )
    _daily_counter += 1


def get_budget_status() -> dict:
    return {
        "used": _daily_counter,
        "limit": DAILY_GEMINI_LIMIT,
        "remaining": max(0, DAILY_GEMINI_LIMIT - _daily_counter),
    }


API_SECRET = os.getenv("API_SECRET_KEY")


async def verify_api_key(request: Request, call_next):
    """Reject requests without valid API key (if API_SECRET_KEY is set)."""
    if API_SECRET is None:
        return await call_next(request)

    if request.url.path in ("/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    # Check for X-API-Key header or generic Authorization Bearer token header
    api_key = request.headers.get("X-API-Key")
    
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header.replace("Bearer ", "")

    if api_key != API_SECRET:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"}
        )
    return await call_next(request)


def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )
