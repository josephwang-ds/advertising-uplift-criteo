from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "criteo_uplift_sample.csv"
LOCAL_FULL_PATH = RAW_DIR / "criteo-research-uplift-v2.1.csv.gz"

CRITEO_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
EXPECTED_COLUMNS = [f"f{i}" for i in range(12)] + [
    "treatment",
    "conversion",
    "visit",
    "exposure",
]


def read_random_sample(source: Path | str, rows: int, chunksize: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None

    for chunk in pd.read_csv(source, compression="gzip", chunksize=chunksize):
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        if reservoir is None:
            reservoir = chunk
        else:
            reservoir = pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > rows * 3:
            reservoir = reservoir.nsmallest(rows, "_sample_key")

    if reservoir is None:
        raise ValueError("No rows were read from the source file.")
    return reservoir.nsmallest(rows, "_sample_key").drop(columns="_sample_key")


def download_sample(rows: int, chunksize: int, seed: int) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source = LOCAL_FULL_PATH if LOCAL_FULL_PATH.exists() else CRITEO_URL
    df = read_random_sample(source, rows=rows, chunksize=chunksize, seed=seed)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Source: {source}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = download_sample(args.rows, args.chunksize, args.seed)
    print(f"Saved {len(df):,} rows to {SAMPLE_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
