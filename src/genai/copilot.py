"""
Analyst Copilot — multi-turn conversational Q&A about flagged sessions.
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
app = FastAPI()

from google import genai
from genai.explainer import get_model, build_risk_context, DEFAULT_MODEL


@dataclass
class ChatMessage:
    role:    str      # "user" or "assistant"
    content: str


@dataclass
class CopilotSession:
    risk_context:  dict
    history:       List[ChatMessage] = field(default_factory=list)


COPILOT_SYSTEM_PROMPT = """
ROLE
You are a senior SOC analyst copilot in an insider-threat detection platform.
You assist human analysts in investigating flagged user sessions.

You are given a structured JSON risk context describing a session scored by:
- Behavioral anomaly detection model
- Command-sequence model
- Rule-based detection engine
- Risk fusion layer

You must NEVER invent signals not present in this context.

RISK CONTEXT
{context}

OBJECTIVE
Help the analyst understand:
- Why the session was flagged
- How risky it is
- What the user might be trying to do
- What to do next

Your responses must always be grounded ONLY in the risk context above.

RESPONSE MODES (adapt to question intent)

1. WHY-FLAGGED — Reference exact scores and triggered rules
2. SIMULATION — Numbered list of likely next actions, map to MITRE ATT&CK
3. SUMMARY/REPORT — Executive summary + technical breakdown
4. RECOMMENDATIONS — Prioritized: containment, investigation, monitoring, escalation
5. GENERAL — Conversational but evidence-grounded

RULES
- Never fabricate logs, commands, or data
- Never assume facts not in context
- If data is missing say: "Not enough evidence in current session context."
- Distinguish between: observed signals, model inference, hypothetical simulation
- Professional SOC tone, concise (2-4 sentences default), expand only when asked
"""


def create_session(fused_risk, session_context: dict = None) -> CopilotSession:
    context = build_risk_context(fused_risk, session_context)
    return CopilotSession(risk_context=context, history=[])

def ask(session: CopilotSession,
        question: str,
        api_key: str = None,
        model_name: str = DEFAULT_MODEL) -> str:
    """Send a question to the copilot, returns response string."""
    session.history.append(ChatMessage(role="user", content=question))

    prompt = COPILOT_SYSTEM_PROMPT.format(context=session.risk_context)
    for msg in session.history:
        prompt += f"\n{msg.role}: {msg.content}"

    client, model_name = get_model(api_key, model_name)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    session.history.append(ChatMessage(role="assistant", content=response.text))
    return response.text


@app.post("/ask")
def ask_copilot(session: CopilotSession, question: str):
    return ask(session, question)


if __name__ == "__main__":
    mock_fused = {
        "anomaly_score": 0.92, "sequence_score": 0.87,
        "rule_score": 0.80, "final_risk": 0.95,
        "risk_level": "critical", "confidence": 0.95,
        "triggered_rules": ["exfiltration_pattern", "privilege_escalation"],
    }
    mock_session = {
        "user_id": "AJF0370", "date": "2010-06-15",
        "after_hours": True, "login_count": 12, "command_count": 34,
    }

    session = create_session(mock_fused, mock_session)
    print("Copilot ready. Type 'quit' to exit.\n")

    while True:
        q = input("Analyst > ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        response = ask(session, q)
        print(f"\nCopilot > {response}\n")
