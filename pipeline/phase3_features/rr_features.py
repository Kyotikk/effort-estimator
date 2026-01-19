#!/usr/bin/env python3
"""
RR (respiratory rate) feature extraction.
Computes statistical features from RR interval time series.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_rr_features_window(rr_window: np.ndarray) -> dict:
    """
    Extract statistical features from RR window.
    RR is given in beats per minute (typical range 60-120).
    """
    features = {}

    if len(rr_window) < 2 or np.all(np.isnan(rr_window)):
        # Return NaN for all features if insufficient data
        return {
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_range": np.nan,
            "rr_median": np.nan,
            "rr_mad": np.nan,
            "rr_cv": np.nan,
        }

    # Remove NaN values
    valid = rr_window[~np.isnan(rr_window)]
    if len(valid) < 2:
        return {
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_range": np.nan,
            "rr_median": np.nan,
            "rr_mad": np.nan,
            "rr_cv": np.nan,
        }

    features["rr_mean"] = float(np.mean(valid))
    features["rr_std"] = float(np.std(valid))
    features["rr_min"] = float(np.min(valid))
    features["rr_max"] = float(np.max(valid))
    features["rr_range"] = features["rr_max"] - features["rr_min"]
    features["rr_median"] = float(np.median(valid))

    # Median absolute deviation
    mad = np.median(np.abs(valid - np.median(valid)))
    features["rr_mad"] = float(mad)

    # Coefficient of variation
    if features["rr_mean"] != 0:
        features["rr_cv"] = features["rr_std"] / features["rr_mean"]
    else:
        features["rr_cv"] = np.nan

    return features


def extract_rr_features(
    rr_path: str,
    windows_path: str,
    out_path: str,
    time_col: str = "t_sec",
    signal_col: str = "value",
) -> None:
    """
    Extract RR features for each window.

    Args:
        rr_path: Path to preprocessed RR CSV
        windows_path: Path to windows CSV (defines windows by t_start, t_end)
        out_path: Output path for features
        time_col: Name of time column in RR data
        signal_col: Name of signal column in RR data
    """
    # Load RR data
    rr_df = pd.read_csv(rr_path)
    if time_col not in rr_df.columns or signal_col not in rr_df.columns:
        raise ValueError(f"RR file missing '{time_col}' or '{signal_col}'. Columns: {list(rr_df.columns)}")

    t_rr = rr_df[time_col].to_numpy(dtype=float)
    rr_vals = rr_df[signal_col].to_numpy(dtype=float)

    # Load windows
    windows_df = pd.read_csv(windows_path)
    required_cols = ["t_start", "t_end"]
    if not all(c in windows_df.columns for c in required_cols):
        raise ValueError(f"Windows file missing {required_cols}. Columns: {list(windows_df.columns)}")

    t_starts = windows_df["t_start"].to_numpy(dtype=float)
    t_ends = windows_df["t_end"].to_numpy(dtype=float)

    # Extract features for each window
    all_features = []
    for t_start, t_end in zip(t_starts, t_ends):
        mask = (t_rr >= t_start) & (t_rr <= t_end)
        rr_window = rr_vals[mask]
        window_features = extract_rr_features_window(rr_window)
        all_features.append(window_features)

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(out_path, index=False)

    print(f"✓ RR features extracted → {out_path} | windows={len(features_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RR features from preprocessed data")
    parser.add_argument("--rr", required=True, help="Path to preprocessed RR CSV")
    parser.add_argument("--windows", required=True, help="Path to windows CSV")
    parser.add_argument("--out", required=True, help="Output path for features")
    parser.add_argument("--time_col", default="t_sec", help="Time column name")
    parser.add_argument("--signal_col", default="value", help="Signal column name")

    args = parser.parse_args()

    extract_rr_features(
        rr_path=args.rr,
        windows_path=args.windows,
        out_path=args.out,
        time_col=args.time_col,
        signal_col=args.signal_col,
    )
