"""
Structured Risk Explanation Engine
===================================
Sends structured risk context (not raw logs) to Gemini
and returns a parsed, actionable explanation.

This is the foundation module — report.py, copilot.py,
and simulator.py all build on top of this.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

app = FastAPI()

from google import genai


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.5-flash"


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

@dataclass
class RiskExplanation:
    """Structured output from the LLM."""
    threat_category:       str
    severity:              str              # Low / Medium / High / Critical
    likely_intent:         str
    recommended_actions:   List[str]
    confidence_reasoning:  str


# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────

def get_model(api_key: str = None, model_name: str = DEFAULT_MODEL):
    """Configure and return a Gemini client."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    client = genai.Client(api_key=api_key)
    return client, model_name


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────

def build_risk_context(fused_risk, session_context: dict = None) -> dict:
    """
    Build a structured JSON context dict from a FusedRisk
    object + optional session metadata.

    This is what gets sent to the LLM — never raw logs.
    """
    # Handle both dict and dataclass inputs
    if isinstance(fused_risk, dict):
        context = {k: fused_risk[k] for k in [
            "anomaly_score", "sequence_score", "rule_score",
            "final_risk", "risk_level", "confidence", "triggered_rules"
        ] if k in fused_risk}
    else:
        context = {
            "anomaly_score":   fused_risk.anomaly_score,
            "sequence_score":  fused_risk.sequence_score,
            "rule_score":      fused_risk.rule_score,
            "final_risk":      fused_risk.final_risk,
            "risk_level":      fused_risk.risk_level,
            "confidence":      fused_risk.confidence,
            "triggered_rules": fused_risk.triggered_rules,
        }
    if session_context:
        context.update(session_context)
    return context


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert cybersecurity analyst at a SOC (Security Operations Center).
You will receive a structured JSON object describing a flagged user session.
Your job is to analyze the risk signals and provide a structured assessment.

Always respond in valid JSON with exactly these fields:
- threat_category (string): e.g. "Data Exfiltration", "Privilege Escalation", "Reconnaissance"
- severity (string): one of "Low", "Medium", "High", "Critical"
- likely_intent (string): one-sentence assessment of the user's probable intent
- recommended_actions (list of strings): 2-4 specific SOC actions
- confidence_reasoning (string): why you are confident (or not) in this assessment
"""


# ─────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────

def explain_risk(fused_risk,
                 session_context: dict = None,
                 api_key: str = None,
                 model_name: str = DEFAULT_MODEL) -> RiskExplanation:
    """
    Main entry point. Takes a FusedRisk + optional session metadata,
    sends structured context to Gemini, returns a parsed RiskExplanation.
    """
    client, model_name = get_model(api_key, model_name)
    context = build_risk_context(fused_risk, session_context)
    response = client.models.generate_content(
        model=model_name,
        contents=f"{SYSTEM_PROMPT}\n\n{json.dumps(context, indent=2)}",
    )
    return parse_response(response.text)


def parse_response(raw_text: str) -> RiskExplanation:
    """Parse Gemini's JSON response into a RiskExplanation dataclass."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Gemini returned an empty response")

    # Strip markdown code fences if present (```json ... ```)
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return RiskExplanation(
            threat_category=data.get("threat_category", "Unknown"),
            severity=data.get("severity", "Unknown"),
            likely_intent=data.get("likely_intent", "Unknown"),
            recommended_actions=data.get("recommended_actions", []),
            confidence_reasoning=data.get("confidence_reasoning", ""),
        )
    except json.JSONDecodeError as e:
        print(f"\n[DEBUG] Raw Gemini response:\n{raw_text}\n")
        raise ValueError(f"Failed to parse Gemini response as JSON: {e}")


#POST /explain api
@app.post("/explain")
def explain_risk_api(fused_risk: dict, session_context: dict = None):
    """API endpoint to explain risk."""
    response = explain_risk(fused_risk, session_context)
    return asdict(response)


# ─────────────────────────────────────────────
# CLI SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated FusedRisk-like dict for testing
    mock_fused = {
        "anomaly_score":   0.92,
        "sequence_score":  0.87,
        "rule_score":      0.80,
        "final_risk":      0.95,
        "risk_level":      "critical",
        "confidence":      0.95,
        "triggered_rules": ["exfiltration_pattern", "privilege_escalation"],
    }

    mock_session = {
        "user_id":       "AJF0370",
        "date":          "2010-06-15",
        "after_hours":   True,
        "login_count":   12,
        "command_count":  34,
    }

    context = build_risk_context(mock_fused, mock_session)
    response = explain_risk(context)
    print("Risk context that would be sent to Gemini:")
    print(json.dumps(context, indent=2))
    print("\nResponse from Gemini:")
    print(json.dumps(asdict(response), indent=2))
