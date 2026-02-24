"""
Risk Detection API entry point.
Run: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import scoring, genai, dashboard
from api.middleware.rate_limiter import limiter, verify_api_key, rate_limit_handler

app = FastAPI(
    title="Insider Threat Risk Detection API",
    description="ML pipeline + GenAI explainability for insider threat detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(verify_api_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

app.include_router(scoring.router)
app.include_router(genai.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def preload_models():
    from genai.inference_pipeline import load_models
    print("[STARTUP] Preloading ML models...")
    load_models()
    print("[STARTUP] Models ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
