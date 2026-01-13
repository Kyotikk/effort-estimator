#!/usr/bin/env python3
# features/tifex.py
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
def flatten_axis_feature_df(feat_df: pd.DataFrame, prefix_sep: str = "__") -> Dict[str, float]:
    """
    Flatten TIFEX output:
    index = signal name, columns = feature names
    -> {signal__feature: value}
    """
    flat: Dict[str, float] = {}
    # index contains axis names; columns are feature names
    for signal_name, row in feat_df.iterrows():
        for feat_name, val in row.items():
            flat[f"{signal_name}{prefix_sep}{feat_name}"] = val
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
        safe = [c for c in safe_candidates if c in available]
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

# ---- Pure function ----
def run_tifex(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    fs: int,
    signal_cols: List[str],
    feature_set: str = "stat",
    safe: bool = True,
    njobs: int = 1,
    quiet: bool = True,
) -> pd.DataFrame:
    """
    Run TIFEX feature extraction on pre-windowed signals.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed signal data (indexed or row-based).
    windows : pd.DataFrame
        Window definitions with start_idx, end_idx.
    fs : int
        Sampling frequency in Hz.
    signal_cols : list[str]
        Signal columns to extract features from.
    feature_set : {"stat", "spec", "tf", "all"}
    safe : bool
        Use conservative calculator subsets.
    njobs : int
        Parallel jobs for TIFEX.
    quiet : bool
        Suppress warnings.

    Returns
    -------
    pd.DataFrame
        One row per window, flattened feature columns.
    """

    fs = int(round(fs))

    if quiet:
        warnings.filterwarnings("ignore")

    for c in ["start_idx", "end_idx"]:
        if c not in windows.columns:
            raise ValueError(f"Windows DataFrame missing '{c}'.")
        
    missing = [c for c in signal_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}")

    # Feature params
    if safe:
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
        end = int(w.end_idx)
        segment = data.iloc[start:end][signal_cols]

        out = {
            "window_id": i,
            "start_idx": start,
            "end_idx": end,
            "valid": True,
        }

        # Optional metadata if present
        for meta in ["t_start", "t_center", "t_end", "n_samples"]:
            if meta in windows.columns:
                out[meta] = w[meta]

        # Guard against empty windows
        if len(segment) == 0:
            # fill nothing else; keep row so alignment stays correct
            out["valid"] = False
            rows.append(out)
            continue

        # Compute features
        # TIFEX expects DataFrame with columns specified; output is indexed by column names
        try:
            if feature_set == "stat":
                feats = extraction.calculate_statistical_features(segment, stat_params, columns=signal_cols, njobs=njobs)
            elif feature_set == "spec":
                feats = extraction.calculate_spectral_features(segment, spec_params, columns=signal_cols, njobs=njobs)
            elif feature_set == "tf":
                feats = extraction.calculate_time_frequency_features(segment, tf_params, columns=signal_cols, njobs=njobs)
            elif feature_set == "all":
                feats = extraction.calculate_all_features(
                    segment, stat_params, spec_params, tf_params, columns=signal_cols, njobs=njobs
                )
            else:
                raise ValueError(f"Unknown feature_set: {feature_set}")

            if feats.empty:
                out["valid"] = False
            else:
                out.update(flatten_axis_feature_df(feats))

        except Exception as e:
            # If something unexpected blows up (not just TIFEX internal NaNs),
            # keep the window metadata and record the error string.
            out["valid"] = False
            out["error"] = str(e)

        rows.append(out)

    feats_df = pd.DataFrame(rows)

    return feats_df
    
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

    df = pd.read_csv(args.input)
    windows = pd.read_csv(args.windows)
    signal_cols = [s.strip() for s in args.signals.split(",")]

    feat_df = run_tifex(
        data=df,
        windows=windows,
        fs=int(args.fs),
        signal_cols=signal_cols,
        feature_set=args.features,
        safe=args.safe,
        njobs=args.njobs,
        quiet=args.quiet,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    feat_df.to_csv(args.out, index=False)

    print(f"Wrote features for {len(feat_df)} windows to {args.out}")
    print("Example:")
    print(feat_df.head(3).to_string(index=False))
    
if __name__ == "__main__": main()