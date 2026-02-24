import os
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_FLAG = OUT_DIR / "processing_done.flag"

CHUNK_SIZE = 100_000



# CSV → PARQUET

def process_file(file_path, out_name):
    output_file = OUT_DIR / f"{out_name}.parquet"

    if output_file.exists():
        print(f"Skipping {file_path.name} already processed")
        return

    print(f"Processing {file_path.name}")
    first_chunk = True

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        for col in chunk.columns:
            if "date" in col.lower():
                chunk[col] = pd.to_datetime(chunk[col], errors="coerce")

        chunk = chunk.dropna()

        for col in chunk.select_dtypes("object").columns:
            chunk[col] = chunk[col].astype("category")

        if first_chunk:
            chunk.to_parquet(output_file, index=False)
            first_chunk = False
        else:
            chunk.to_parquet(output_file, index=False, append=True)

        del chunk

    print(f"Saved → {output_file}")



def features():
    print("Building features using POLARS...")

    # -------- DEVICE --------
    device = (
        pl.scan_parquet(str(OUT_DIR / "device.parquet"))
        .with_columns(pl.col("date").cast(pl.Date))
        .group_by(["user","date"])
        .agg(pl.col("pc").n_unique().alias("unique_devices_used"))
        .with_columns(pl.col("user").alias("user_id"))
        .select(["user_id","date","unique_devices_used"])
    )

    device.collect().write_parquet(OUT_DIR / "device_features.parquet")
    print("device done")

    # -------- FILE --------
    file = (
        pl.scan_parquet(str(OUT_DIR / "file.parquet"))
        .with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("content").cast(pl.String).str.len_chars().alias("bytes")
        ])
        .group_by(["user","date"])
        .agg(pl.col("bytes").sum().alias("total_bytes_file"))
        .with_columns(pl.col("user").alias("user_id"))
        .select(["user_id","date","total_bytes_file"])
    )

    file.collect().write_parquet(OUT_DIR / "file_features.parquet")
    print("file done")

    # -------- HTTP --------
    http = (
        pl.scan_parquet(str(OUT_DIR / "http.parquet"))
        .with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("content").cast(pl.String).str.len_chars().alias("bytes")
        ])
        .group_by(["user","date"])
        .agg(pl.col("bytes").sum().alias("total_bytes_http"))
        .with_columns(pl.col("user").alias("user_id"))
        .select(["user_id","date","total_bytes_http"])
    )

    http.collect().write_parquet(OUT_DIR / "http_features.parquet")
    print("http done")

    # -------- LOGON --------
    logon = (
        pl.scan_parquet(str(OUT_DIR / "logon.parquet"))
        .with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("date").dt.hour().alias("hour")
        ])
        .with_columns(
            ((pl.col("hour")>=18) | (pl.col("hour")<=6))
            .cast(pl.Int8)
            .alias("after_hours")
        )
        .group_by(["user","date"])
        .agg([
            (pl.col("activity").str.to_lowercase()=="logon").sum().alias("login_count"),
            pl.col("after_hours").max().alias("after_hours_activity")
        ])
        .with_columns(pl.col("user").alias("user_id"))
        .select(["user_id","date","login_count","after_hours_activity"])
    )

    logon.collect().write_parquet(OUT_DIR / "logon_features.parquet")
    print("logon done")

    print("All feature tables created using polar.")


# MERGE

def merge_features():
    print("Merging feature tables...")

    device_features = pd.read_parquet(OUT_DIR / "device_features.parquet")
    file_features   = pd.read_parquet(OUT_DIR / "file_features.parquet")
    http_features   = pd.read_parquet(OUT_DIR / "http_features.parquet")
    logon_features  = pd.read_parquet(OUT_DIR / "logon_features.parquet")

    df = pd.merge(logon_features, device_features, on=["user_id","date"], how="left")
    df = pd.merge(df, file_features, on=["user_id","date"], how="left")
    df = pd.merge(df, http_features, on=["user_id","date"], how="left")

    df["total_bytes_transferred"] = (
        df["total_bytes_file"].fillna(0) +
        df["total_bytes_http"].fillna(0)
    )

    df["unique_devices_used"] = df["unique_devices_used"].fillna(0)

    df = df[[
        "user_id",
        "date",
        "login_count",
        "after_hours_activity",
        "total_bytes_transferred",
        "unique_devices_used"
    ]]

    df.to_parquet(OUT_DIR / "features.parquet")
    df.head(100).to_csv(OUT_DIR / "sample.csv", index=False)

    print("FINAL features.parquet created")



def run():
    features()
    merge_features()


if __name__ == "__main__":
    run()