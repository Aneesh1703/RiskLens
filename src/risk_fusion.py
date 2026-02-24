"""
Risk Fusion Layer
=================
Pure-function score combiner. Receives pre-computed scores from
the Isolation Forest, LSTM, and Rule Engine, and produces a
unified risk profile per session.

No model loading — takes pre-computed scores only.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd


# Weights (ML-heavy: 80% ML, 20% rules)
W_ANOMALY  = 0.40
W_SEQUENCE = 0.40
W_RULES    = 0.20

# Agreement threshold (above this = "flagged")
FLAG_THRESHOLD = 0.50


@dataclass
class FusedRisk:
    """Unified risk profile for a single session."""
    anomaly_score:    float
    sequence_score:   float
    rule_score:       float
    final_risk:       float
    risk_level:       str            # low / medium / high / critical
    confidence:       float          # 0.10 – 0.95
    triggered_rules:  List[str] = field(default_factory=list)


def fuse_scores(anomaly_score: float,
                sequence_score: float,
                rule_score: float,
                triggered_rules: List[str] = None) -> FusedRisk:
    """
    Combine three pre-computed scores into a single FusedRisk.

    Steps:
      1. Weighted average (0.40 / 0.40 / 0.20)
      2. Tiered escalation override
      3. Risk level bucketing
      4. Agreement-based confidence
    """
    score = [anomaly_score, sequence_score, rule_score]
    final_risk = float(np.average(score, weights=[W_ANOMALY, W_SEQUENCE, W_RULES]))


    if rule_score >= 0.9:
        final_risk = max(final_risk, 0.95)
    if anomaly_score > 0.85 and sequence_score > 0.85:
        final_risk = max(final_risk, 0.95)
    elif rule_score >= 0.8:
        final_risk = max(final_risk, 0.75)
    elif anomaly_score >= 0.8 or sequence_score >= 0.8:
        final_risk = max(final_risk, 0.70)

    risk_level = classify_risk(final_risk)
    confidence = compute_confidence(anomaly_score, sequence_score, rule_score)
    return FusedRisk(
        anomaly_score=anomaly_score,
        sequence_score=sequence_score,
        rule_score=rule_score,
        final_risk=round(final_risk, 4),
        risk_level=risk_level,
        confidence=confidence,
        triggered_rules=triggered_rules or [],
    )


def classify_risk(score: float) -> str:
    """Map a 0–1 score to a risk level string."""
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.8:
        return "high"
    else:
        return "critical"


def compute_confidence(anomaly: float, sequence: float, rule: float) -> float:
    """Agreement-based confidence: how many signals flag above threshold."""
    flagged = sum(1 for s in [anomaly, sequence, rule] if s >= FLAG_THRESHOLD)
    return flagged / 3.0


def fuse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch fusion across a DataFrame.

    Expects columns: anomaly_score, sequence_score, rule_score, rule_names.
    Adds columns: final_risk, risk_level, confidence.
    """
    df["final_risk"] = df.apply(
        lambda row: fuse_scores(
            row["anomaly_score"],
            row["sequence_score"],
            row["rule_score"],
            row["rule_names"],
        ).final_risk,
        axis=1,
    )
    df["risk_level"] = df["final_risk"].apply(classify_risk)
    df["confidence"] = df.apply(
        lambda row: compute_confidence(
            row["anomaly_score"],
            row["sequence_score"],
            row["rule_score"],
        ),
        axis=1,
    )
    return df


def print_summary(df: pd.DataFrame, top_n: int = 10) -> None:
    """Pretty-print the top-N riskiest sessions."""
    print("=" * 80)
    print("Risk Fusion — Summary")
    print("=" * 80)
    


if __name__ == "__main__":
    # Smoke test
    result = fuse_scores(
        anomaly_score=0.85,
        sequence_score=0.72,
        rule_score=0.60,
        triggered_rules=["exfiltration_pattern", "after_hours_usb"],
    )
    print(f"Final Risk : {result.final_risk:.3f}")
    print(f"Level      : {result.risk_level}")
    print(f"Confidence : {result.confidence}")
    print(f"Rules      : {result.triggered_rules}")
