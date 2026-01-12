#!/usr/bin/env python3
"""
Run TIFEX feature extraction per precomputed window and write one row per window.
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from tifex_py.feature_extraction import settings, extraction


# ---- Helper functions -----
def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def flatten_axis_feature_df(feat_df: pd.DataFrame, prefix_sep: str = "__") -> Dict[str, float]:
    """
    TIFEX returns a DataFrame indexed by axis (acc_x, acc_y, ...) and columns as features.
    We flatten into { "acc_x__mean": value, ... } so we can store a single row per window.
    """
    flat: Dict[str, float] = {}
    # index contains axis names; columns are feature names
    for signal_name, row in feat_df.iterrows():
        for feat_name, val in row.items():
            flat[f"{axis_name}{prefix_sep}{feat_name}"] = val
    return flat

# ---- Safe calculator selection -----
def choose_safe_stat_calculators(fs: int) -> settings.StatisticalFeatureParams:
    """
    Conservative "safe" statistical feature subset to avoid the common error-prone ones
    (HFD, moving_average, cardinality, hurst_exponent, etc.).
    """
    # This list is intentionally conservative.
    # If TIFEX doesn't recognize a name, we'll filter it below after we inspect defaults.
    safe_candidates = [
        "mean",
        "mean_of_abs",
        "std",
        "var",
        "median",
        "min",
        "max",
        "range",
        "iqr",
        "quantile_0.1",
        "quantile_0.25",
        "quantile_0.75",
        "quantile_0.9",
        "skewness",
        "kurtosis",
        "mean_abs_deviation",
        "cid_ce",
        "lempel_ziv_complexity",
        "number_cwt_peaks_1",
        "number_cwt_peaks_5",
        "large_std",
    ]

    # Build a default params object so we can discover which calculators are available
    default = settings.StatisticalFeatureParams(fs)
    available = getattr(default, "calculators", None)

    if isinstance(available, list):
        safe = [c for c in safe_candidates if c in available]
        # If none matched (unexpected), fall back to defaults (but we'll get NaNs for some features)
        return settings.StatisticalFeatureParams(fs, calculators=safe)

    # If the installed version doesn't expose calculators as a list, just return defaults.
    return default


def choose_safe_spectral_calculators(fs: int) -> settings.SpectralFeatureParams:
    """
    Conservative safe spectral subset.
    """
    safe_candidates = [
        "spectral_variance",
        "spectral_skewness",
        "spectral_kurtosis",
        "spectral_flatness",
        "spectral_centroid_order_1",
        "spectral_centroid_order_2",
        "median_frequency",
    ]

    default = settings.SpectralFeatureParams(fs)
    available = getattr(default, "calculators", None)

    if isinstance(available, list):
        safe = [c for c in safe if c in available]
        return settings.SpectralFeatureParams(fs, calculators=safe)

    return default


def choose_safe_timefreq_calculators(fs: int) -> settings.TimeFrequencyFeatureParams:
    """
    Conservative time-frequency subset: only TKEO representation (usually robust),
    and use safe statistical params for the representation.
    """
    stat_params = choose_safe_stat_calculators(fs)
    default = settings.TimeFrequencyFeatureParams(fs)

    available = getattr(default, "calculators", None)
    if isinstance(available, list) and "tkeo_features" in available:
        return settings.TimeFrequencyFeatureParams(
            fs,
            calculators=["tkeo_features"],
            tkeo_sf_params=stat_params,
        )

    return default

# ---- CLI ----
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TIFEX on pre-windowed signals.")
    ap.add_argument("--input", required=True, help="Preprocessed CSV with t_sec and signal columns")
    ap.add_argument("--windows", required=True, help="Window CSV with start_idx, end_idx")
    ap.add_argument("--out", required=True, help="Output feature CSV")
    ap.add_argument("--fs", required=True, type=float, help="Sampling frequency (Hz)")
    ap.add_argument("--signals", required=True, help="Comma-separated list of signal columns")
    ap.add_argument("--features", choices=["stat", "spec", "tf", "all"], default="stat")
    ap.add_argument("--safe", action="store_true", help="Use conservative feature subsets")
    ap.add_argument("--njobs", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

# ---- Main ----
def main() -> None:
    args = parse_args()
    fs = int(round(float(args.fs)))

    if args.quiet:
        warnings.filterwarnings("ignore")

    # Load raw data
    df = pd.read_csv(args.input)
    windows = pd.read_csv(args.windows)

    if "t_sec" not in df.columns:
        raise ValueError("Input data must contain 't_sec'.")

    for c in ["start_idx", "end_idx"]:
        if c not in windows.columns:
            raise ValueError(f"Windows file missing '{c}'.")
    
    signal_cols = [c.strip() for c in args.signals.split(",")]
    
    missing = [c for c in signal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}")
    
    # Feature params
    if args.safe:
        stat_params = choose_safe_stat_calculators(fs)
        spec_params = choose_safe_spectral_calculators(fs)
        tf_params = choose_safe_timefreq_calculators(fs)
    else:
        stat_params = settings.StatisticalFeatureParams(fs)
        spec_params = settings.SpectralFeatureParams(fs)
        tf_params = settings.TimeFrequencyFeatureParams(fs)

    rows: List[Dict[str, float]] = []
    
    for i, w in windows.iterrows():
        start = int(w.start_idx)
        end= int(w.end_idx)
        segment = df.iloc[start:end][signal_cols]

        out = {
            "window_id": i,
            "start_idx": start,
            "end_idx": end,
        }

        # Optional metadata if present
        for meta in ["t_start", "t_end", "n_samples"]:
            if meta in windows.columns:
                out[meta] = w.loc[i, meta]

        # Guard against empty windows
        if len(segment) == 0:
            # fill nothing else; keep row so alignment stays correct
            rows.append(out)
            continue

        # Compute features
        # TIFEX expects DataFrame with columns specified; output is indexed by column names
        try:
            if args.features == "stat":
                feats = extraction.calculate_statistical_features(segment, stat_params, columns=signal_cols, njobs=args.njobs)
            elif args.features == "spec":
                feats = extraction.calculate_spectral_features(segment, spec_params, columns=signal_cols, njobs=args.njobs)
            elif args.features == "tf":
                feats = extraction.calculate_time_frequency_features(segment, tf_params, columns=signal_cols, njobs=args.njobs)
            else:
                feats = extraction.calculate_all_features(
                    segment, stat_params, spec_params, tf_params, columns=signal_cols, njobs=args.njobs
                )

            out.update(flatten_axis_feature_df(feats))

        except Exception as e:
            # If something unexpected blows up (not just TIFEX internal NaNs),
            # keep the window metadata and record the error string.
            out["error"] = str(e)

        rows.append(out)

        # light progress
        if (i + 1) % 200 == 0:
            print(f"[tifex_from_windows] processed {i+1}/{windows} windows", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    ensure_dir_for_file(args.out)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {len(out_df)} windows x features to: {args.out}")
    print(f"Columns: {len(out_df.columns)}")


if __name__ == "__main__":
    main()
