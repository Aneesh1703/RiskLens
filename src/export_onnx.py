#!/usr/bin/env python3
"""
ONNX Export Script
==================
Converts the trained PyTorch `lstm_risk_model.pt` into a lightweight,
CPU-optimized `lstm_risk_model.onnx` file for fast Cloud Run inference.
"""

import sys
import torch
from pathlib import Path

# Add src to the path so we can import the model class
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_sequence import LSTMRiskModel

PROJECT_ROOT = SRC_DIR.parent
PT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_risk_model.pt"
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_risk_model.onnx"


def export_model():
    print(f"Loading checkpoint from: {PT_MODEL_PATH}")
    if not PT_MODEL_PATH.exists():
        print("ERROR: Checkpoint not found. Run train_sequence.py first.")
        sys.exit(1)

    checkpoint = torch.load(PT_MODEL_PATH, map_location="cpu", weights_only=False)

    print("Initializing PyTorch model...")
    model = LSTMRiskModel(
        vocab_size=len(checkpoint["vocab"]),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_meta=checkpoint["num_meta"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Generating dummy inputs for shape inference...")
    max_len = checkpoint["max_len"]
    num_meta = checkpoint["num_meta"]

    # Dummy inputs: Batch size = 1
    dummy_seq = torch.randint(0, len(checkpoint["vocab"]), (1, max_len), dtype=torch.long)
    dummy_meta = torch.randn(1, num_meta, dtype=torch.float32)

    print(f"Exporting to ONNX at: {ONNX_MODEL_PATH}")
    torch.onnx.export(
        model,
        (dummy_seq, dummy_meta),
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=14,                      # Compatible opset
        do_constant_folding=True,              # Optimize graph
        input_names=["sequence", "metadata"],
        output_names=["logits"],
        dynamic_axes={                         # Allow variable batch sizes (essential for production)
            "sequence": {0: "batch_size"},
            "metadata": {0: "batch_size"},
            "logits":   {0: "batch_size"},
        }
    )
    
    # Save the original metadata alongside the ONNX file as JSON
    # so we don't need torch to read the vocab or scaler in production
    import json
    metadata_path = ONNX_MODEL_PATH.with_suffix(".meta.json")
    
    onnx_meta = {
        "vocab": checkpoint["vocab"],
        "max_len": max_len,
        "num_meta": num_meta,
        "scaler_mean": checkpoint["scaler_mean"].tolist() if hasattr(checkpoint["scaler_mean"], "tolist") else checkpoint["scaler_mean"],
        "scaler_scale": checkpoint["scaler_scale"].tolist() if hasattr(checkpoint["scaler_scale"], "tolist") else checkpoint["scaler_scale"],
        "mal_threshold": checkpoint.get("mal_threshold", 0.5)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(onnx_meta, f, indent=2)

    print(f"✅ Export complete! Saved to {ONNX_MODEL_PATH}")
    print(f"✅ Metadata dictionary saved to {metadata_path}")


if __name__ == "__main__":
    export_model()
