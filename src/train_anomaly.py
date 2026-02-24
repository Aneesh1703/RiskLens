import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import joblib

DATA_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/isolation_forest.joblib")


# 1. LOAD + PREPROCESS
def load_and_preprocess_data():
    """
    Loads features.parquet and prepares scaled features
    including user-level z-score deviation features.
    """
    print("\n[1] Loading features dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Sort for stability
    df = df.sort_values(["user_id", "date"])

    # Log transform bytes FIRST
    df["total_bytes_transferred"] = np.log1p(df["total_bytes_transferred"])

   
    # # Compute per-user z-scores
  
    # epsilon = 1e-6

    # for col in ["login_count", "total_bytes_transferred", "unique_devices_used"]:
    #     mean = df.groupby("user_id")[col].transform("mean")
    #     std = df.groupby("user_id")[col].transform("std")
    #     df[f"{col}_zscore"] = (df[col] - mean) / (std + epsilon)


    #rolling base line zscore
    window_size = 30
    epsilon = 1e-6

    for col in ["login_count", "total_bytes_transferred", "unique_devices_used"]:
        rolling_mean = df.groupby("user_id")[col].transform(lambda x:x.shift(1).rolling(window_size, min_periods=1).mean())
        rolling_std  = df.groupby("user_id")[col].transform(lambda x:x.shift(1).rolling(window_size, min_periods=1).std())
        df[f"{col}_rolling_zscore"] = (df[col] - rolling_mean) / (rolling_std + epsilon)
        df[f"{col}_rolling_zscore"] = df[f"{col}_rolling_zscore"].fillna(0)

   
    # Feature selection
   
    feature_cols = [
        "login_count",
        "after_hours_activity",
        "total_bytes_transferred",
        "unique_devices_used",
        "login_count_rolling_zscore",
        "total_bytes_transferred_rolling_zscore",
        "unique_devices_used_rolling_zscore"
    ]

    X = df[feature_cols].copy()

    # Fill missing
    X = X.fillna(0)

    # Scaling
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   Loaded {len(df):,} rows")
    print(f"   Using {len(feature_cols)} features (with behavioral z-scores)")

    return df, X_scaled, scaler, feature_cols


# 2. TRAIN MODEL

def train_model(X_scaled):
    """
    Train Isolation Forest.
    """
    print("\n[2] Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.02,   # 2% anomalies expected
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    print("   Model training complete.")
    return model


# 3. SCORE DATASET

def score_dataset(model, df, X_scaled):
    """
    Compute anomaly scores and probabilities.
    """
    print("\n[3] Scoring anomalies...")


    # Raw scores (higher = normal)
    scores = model.decision_function(X_scaled)

    # Convert → anomaly severity
    anomaly_raw = -scores

    # Normalize to 0–1 probability
    prob_scaler = MinMaxScaler()
    prob_scores = prob_scaler.fit_transform(anomaly_raw.reshape(-1, 1)).flatten()

    df = df.copy()
    df["anomaly_score_raw"] = scores
    df["risk_probability"] = prob_scores

    preds = model.predict(X_scaled)
    df["is_anomaly"] = (preds == -1).astype(int)

    anomaly_count = df["is_anomaly"].sum()

    print(f"   Flagged {anomaly_count:,} anomalies "
          f"({anomaly_count/len(df)*100:.2f}% of data)")

    print("\nTop 5 anomalous user-days:")
    print(
        df.sort_values("risk_probability", ascending=False)
        .head(5)[[
            "user_id",
            "date",
            "risk_probability",
            "login_count",
            "after_hours_activity",
            "total_bytes_transferred",
            "unique_devices_used"
        ]]
    )

    return df


# 4. SAVE MODEL

def save_artifacts(model, scaler, feature_cols):
    """
    Save trained model + scaler.
    """
    print("\n[4] Saving model artifacts...")

    MODEL_PATH.parent.mkdir(exist_ok=True)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "features": feature_cols
    }

    joblib.dump(artifacts, MODEL_PATH)

    print(f"   Saved → {MODEL_PATH}")


# MAIN

def main():
    print("=" * 55)
    print("   CERT Insider Threat — Isolation Forest Pipeline")
    print("=" * 55)

    # Load
    df, X_scaled, scaler, feature_cols = load_and_preprocess_data()

    # Train
    model = train_model(X_scaled)

    # Score
    scored_df = score_dataset(model, df, X_scaled)

    # Save
    save_artifacts(model, scaler, feature_cols)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()