"""
Scoring Routes — ML model inference (no Gemini calls)
======================================================
POST /score  → score single session (saves to DB)
POST /batch  → score entire CSV/DataFrame (saves to DB)
"""

import sys
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File
import pandas as pd
import io

# Add src/ to path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.models.schemas import SessionInput, ScoreResponse
from api.middleware.rate_limiter import limiter
from api.db import save_session, save_batch
from genai.inference_pipeline import score_session, score_dataframe

router = APIRouter(tags=["Scoring"])


@router.post("/score", response_model=ScoreResponse)
@limiter.limit("30/minute")
def score_single(request: Request, session: SessionInput):
    """Score a single session through the full pipeline and save to DB."""
    session_dict = session.model_dump()
    result = score_session(session_dict, explain=True)

    # Persist to SQLite
    save_session(result, session_dict)

    return result


@router.post("/batch")
@limiter.limit("5/minute")
async def score_batch(request: Request, file: UploadFile = File(...)):
    """Upload a CSV, score all rows, save to DB, return results."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    scored = score_dataframe(df, top_n_explain=0)   # ML only, explain on-demand

    # Persist all scored rows to SQLite
    records = scored.to_dict(orient="records")
    save_batch(records)

    return records
