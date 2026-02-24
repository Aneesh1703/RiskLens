from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class SessionInput(BaseModel):
    session_id:               str = "unknown"
    user_id:                  str = "unknown"
    date:                     str = ""
    after_hours:              int = 0
    login_count:              int = 0
    unique_devices_used:      int = 0
    total_bytes_transferred:  float = 0.0
    command_sequence:         str = ""
    command_count:            int = 0
    session_duration_s:       float = 0.0
    after_hours_activity:     int = 0

    class Config:
        extra = "allow"


class ExplainRequest(BaseModel):
    fused_risk:       dict
    session_context:  Optional[dict] = None


class ReportRequest(BaseModel):
    fused_risk:       dict
    session_context:  Optional[dict] = None


class AskRequest(BaseModel):
    risk_context:     dict
    question:         str
    history:          Optional[List[dict]] = []


class ScoreResponse(BaseModel):
    anomaly_score:    float
    sequence_score:   float
    rule_score:       float
    final_risk:       float
    risk_level:       str
    confidence:       float
    triggered_rules:  List[str]
    explanation:      Optional[dict] = None


class ExplainResponse(BaseModel):
    threat_category:       str
    severity:              str
    likely_intent:         str
    recommended_actions:   List[str]
    confidence_reasoning:  str


class ReportResponse(BaseModel):
    user_id:               str
    date:                  str
    risk_level:            str
    executive_summary:     str
    technical_summary:     str
    recommended_actions:   List[str]
    mitre_techniques:      List[str] = []


class DashboardStats(BaseModel):
    total_sessions:   int
    low_count:        int
    medium_count:     int
    high_count:       int
    critical_count:   int
