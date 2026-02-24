import sys
from pathlib import Path
from fastapi import APIRouter, Request
from dataclasses import asdict

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.models.schemas import ExplainRequest, ExplainResponse, ReportRequest, ReportResponse, AskRequest
from api.middleware.rate_limiter import limiter, check_gemini_budget
from genai.explainer import explain_risk
from genai.report import generate_report
from genai.copilot import create_session, ask, CopilotSession, ChatMessage

router = APIRouter(tags=["GenAI"])


@router.post("/explain", response_model=ExplainResponse)
@limiter.limit("10/minute")
def explain(request: Request, body: ExplainRequest):
    check_gemini_budget()
    result = explain_risk(body.fused_risk, body.session_context)
    return asdict(result)


@router.post("/report", response_model=ReportResponse)
@limiter.limit("5/minute")
def report(request: Request, body: ReportRequest):
    check_gemini_budget()
    result = generate_report(body.fused_risk, body.session_context)
    return asdict(result)


@router.post("/ask")
@limiter.limit("10/minute")
def ask_copilot(request: Request, body: AskRequest):
    check_gemini_budget()

    session = CopilotSession(
        risk_context=body.risk_context,
        history=[ChatMessage(role=m["role"], content=m["content"]) for m in body.history]
    )

    response = ask(session, body.question)

    return {
        "response": response,
        "history": [{"role": m.role, "content": m.content} for m in session.history]
    }
