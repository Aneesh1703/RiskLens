"""
Natural Language Threat Report Generator
=========================================
Generates structured executive + technical threat reports
by building on the explainer module's Gemini integration.

Produces three sections per high-risk event:
  1. Executive Summary  — non-technical, one paragraph
  2. Technical Summary  — z-scores, model probabilities, evidence
  3. Recommended Actions — prioritized SOC response steps
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
load_dotenv()

from google import genai

# Import the shared explainer utilities
from genai.explainer import get_model, build_risk_context, RiskExplanation, DEFAULT_MODEL


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

@dataclass
class ThreatReport:
    """Full threat report output."""
    user_id:              str
    date:                 str
    risk_level:           str
    executive_summary:    str
    technical_summary:    str
    recommended_actions:  List[str]
    mitre_techniques:     List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

REPORT_SYSTEM_PROMPT = """You are a senior cybersecurity analyst writing a formal incident report.
You will receive a structured JSON object with risk scores, triggered rules, and session metadata.

Generate a threat report in valid JSON with exactly these fields:
- executive_summary (string): 2-3 sentence non-technical summary for management
- technical_summary (string): detailed paragraph with specific scores, z-scores, and behavioral evidence
- recommended_actions (list of strings): 3-5 prioritized SOC response steps
- mitre_techniques (list of strings): relevant MITRE ATT&CK technique IDs (e.g. "T1048", "T1078")
"""


# ─────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────

def generate_report(fused_risk,
                    session_context: dict = None,
                    api_key: str = None,
                    model_name: str = DEFAULT_MODEL) -> ThreatReport:
    """
    Generate a full threat report from a FusedRisk + session context.
    """

    client, model_name = get_model(api_key, model_name)
    context = build_risk_context(fused_risk, session_context)
    response = client.models.generate_content(
        model=model_name,
        contents=f"{REPORT_SYSTEM_PROMPT}\n\n{json.dumps(context, indent=2)}",
    )
    return parse_report_response(response.text, session_context["user_id"], session_context["date"], fused_risk["risk_level"])
    pass


def parse_report_response(raw_text: str, user_id: str, date: str,
                          risk_level: str) -> ThreatReport:
    """Parse Gemini's response into a ThreatReport dataclass."""
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
        return ThreatReport(
            user_id=user_id,
            date=date,
            risk_level=risk_level,
            executive_summary=data.get("executive_summary", ""),
            technical_summary=data.get("technical_summary", ""),
            recommended_actions=data.get("recommended_actions", []),
            mitre_techniques=data.get("mitre_techniques", []),
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini response as JSON: {e}")
    pass


# ─────────────────────────────────────────────
# CLI SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
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

    report = generate_report(mock_fused, mock_session)
    print(json.dumps(asdict(report), indent=2))
