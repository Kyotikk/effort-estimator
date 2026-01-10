#!/usr/bin/env python3
"""
Stage B: Run TIFEX feature extraction per precomputed window (start_idx, end_idx)
and write one row per window with flattened feature names.

Example:
python3 data/feature_extraction/scripts/tifex_from_windows.py \
  --input data/interim/parsingsim3/sim_elderly3/corsano_bioz_acc/2025-12-04_acc_clean.csv \
  --windows data/feature_extraction/tifex_out/parsingsim3_sim_elderly3_acc_windows_5s_70.csv \
  --out data/feature_extraction/tifex_out/parsingsim3_sim_elderly3_acc_5s_stat_safe.csv \
  --fs 32 \
  --njobs 1 \
  --acc_unit mg \
  --add_magnitude \
  --features stat \
  --safe
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tifex_py.feature_extraction import settings, extraction


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def convert_acc_units(df: pd.DataFrame, cols: List[str], acc_unit: str) -> pd.DataFrame:
    """
    Convert accelerometer units.

    - raw: no change
    - mg : divide by 1000 -> g
    - g  : no change
    - ms2: convert g -> m/s^2 via * 9.80665 (assumes input is g)
           If your input is mg, this will first convert mg->g then ->m/s^2.
    """
    out = df.copy()

    if acc_unit not in {"raw", "mg", "g", "ms2"}:
        raise ValueError(f"Unsupported --acc_unit {acc_unit}. Use raw|mg|g|ms2")

    if acc_unit == "raw":
        return out

    # If mg: convert to g
    if acc_unit == "mg":
        for c in cols:
            out[c] = out[c].astype(float) / 1000.0
        return out

    # If g: keep
    if acc_unit == "g":
        for c in cols:
            out[c] = out[c].astype(float)
        return out

    # If ms2: interpret input as g unless it's clearly mg-like
    # We will do mg->g->m/s2 if magnitudes look like mg integers.
    # (Your current file shows values like -489 which is very plausibly mg.)
    # Rule: if typical abs value > 20, treat as mg (since g should be around ~1).
    median_abs = np.median(np.abs(out[cols].to_numpy().astype(float)))
    if median_abs > 20:
        # treat as mg
        for c in cols:
            out[c] = out[c].astype(float) / 1000.0
    else:
        for c in cols:
            out[c] = out[c].astype(float)

    for c in cols:
        out[c] = out[c] * 9.80665
    return out


def add_magnitude_column(df: pd.DataFrame, x: str, y: str, z: str, name: str = "acc_mag") -> pd.DataFrame:
    out = df.copy()
    out[name] = np.sqrt(out[x].astype(float) ** 2 + out[y].astype(float) ** 2 + out[z].astype(float) ** 2)
    return out


def flatten_axis_feature_df(feat_df: pd.DataFrame, prefix_sep: str = "__") -> Dict[str, float]:
    """
    TIFEX returns a DataFrame indexed by axis (acc_x, acc_y, ...) and columns as features.
    We flatten into { "acc_x__mean": value, ... } so we can store a single row per window.
    """
    flat: Dict[str, float] = {}
    # index contains axis names; columns are feature names
    for axis_name, row in feat_df.iterrows():
        for feat_name, val in row.items():
            flat[f"{axis_name}{prefix_sep}{feat_name}"] = val
    return flat


def choose_safe_stat_calculators(fs_int: int) -> settings.StatisticalFeatureParams:
    """
    Conservative "safe" statistical feature subset to avoid the common error-prone ones
    you saw (HFD, moving_average, cardinality, hurst_exponent, etc.).

    IMPORTANT: We must pass fs_int to StatisticalFeatureParams.
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
        "benford_correlation",
        "lempel_ziv_complexity",
        "number_cwt_peaks_1",
        "number_cwt_peaks_5",
        "has_duplicates",
        "max_has_duplicates",
        "min_has_duplicates",
        "large_std",
    ]

    # Build a default params object so we can discover which calculators are available
    default_params = settings.StatisticalFeatureParams(fs_int)
    available = getattr(default_params, "calculators", None)

    if isinstance(available, list) and len(available) > 0:
        safe = [c for c in safe_candidates if c in available]
        # If none matched (unexpected), fall back to defaults (but you'll get NaNs for some features)
        if len(safe) == 0:
            return default_params
        return settings.StatisticalFeatureParams(fs_int, calculators=safe)

    # If the installed version doesn't expose calculators as a list, just return defaults.
    return default_params


def choose_safe_spectral_calculators(fs_int: int) -> settings.SpectralFeatureParams:
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

    default_params = settings.SpectralFeatureParams(fs_int)
    available = getattr(default_params, "calculators", None)

    if isinstance(available, list) and len(available) > 0:
        safe = [c for c in safe_candidates if c in available]
        if len(safe) == 0:
            return default_params
        return settings.SpectralFeatureParams(fs_int, calculators=safe)

    return default_params


def choose_safe_timefreq_calculators(fs_int: int) -> settings.TimeFrequencyFeatureParams:
    """
    Conservative time-frequency subset: only TKEO representation (usually robust),
    and use safe statistical params for the representation.
    """
    sf_params = choose_safe_stat_calculators(fs_int)

    default_params = settings.TimeFrequencyFeatureParams(fs_int)
    available = getattr(default_params, "calculators", None)

    # We prefer only "tkeo_features" if it exists
    if isinstance(available, list) and "tkeo_features" in available:
        return settings.TimeFrequencyFeatureParams(fs_int, calculators=["tkeo_features"], tkeo_sf_params=sf_params)

    # If not available, fall back to defaults (may be heavy)
    return default_params


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw accel CSV with columns: t_sec, acc_x, acc_y, acc_z")
    ap.add_argument("--windows", required=True, help="Windows CSV from segment_windows.py (start_idx, end_idx, ...)")
    ap.add_argument("--out", required=True, help="Output feature CSV (one row per window)")
    ap.add_argument("--fs", required=True, type=float, help="Sampling frequency (Hz), e.g. 32")
    ap.add_argument("--njobs", default=1, type=int, help="Number of CPU cores for TIFEX (passed to njobs)")
    ap.add_argument("--acc_unit", default="raw", choices=["raw", "mg", "g", "ms2"], help="Unit of acc columns")
    ap.add_argument("--add_magnitude", action="store_true", help="Add acc magnitude column acc_mag")
    ap.add_argument(
        "--features",
        default="stat",
        choices=["stat", "spec", "tf", "all"],
        help="Which TIFEX feature categories to compute",
    )
    ap.add_argument("--safe", action="store_true", help="Use conservative safe feature subsets to reduce errors")
    ap.add_argument("--quiet", action="store_true", help="Reduce warnings")
    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    fs_int = int(round(float(args.fs)))

    if args.quiet:
        warnings.filterwarnings("ignore")

    # Load raw data
    df = pd.read_csv(args.input)
    required_cols = ["t_sec", "acc_x", "acc_y", "acc_z"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in input. Found: {list(df.columns)}")

    # Load windows
    w = pd.read_csv(args.windows)
    for c in ["start_idx", "end_idx"]:
        if c not in w.columns:
            raise ValueError(f"Windows file must contain '{c}'. Found: {list(w.columns)}")

    # Prepare accel columns
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    df = convert_acc_units(df, acc_cols, args.acc_unit)

    if args.add_magnitude:
        df = add_magnitude_column(df, "acc_x", "acc_y", "acc_z", name="acc_mag")
        acc_cols = ["acc_x", "acc_y", "acc_z", "acc_mag"]

    # Params
    if args.safe:
        stat_params = choose_safe_stat_calculators(fs_int)
        spec_params = choose_safe_spectral_calculators(fs_int)
        tf_params = choose_safe_timefreq_calculators(fs_int)
    else:
        stat_params = settings.StatisticalFeatureParams(fs_int)
        spec_params = settings.SpectralFeatureParams(fs_int)
        tf_params = settings.TimeFrequencyFeatureParams(fs_int)

    rows: List[Dict[str, float]] = []
    n = len(w)

    for i in range(n):
        start = int(w.loc[i, "start_idx"])
        end = int(w.loc[i, "end_idx"])  # IMPORTANT: end_idx is treated as EXCLUSIVE (matches your 160 sample windows)

        win = df.iloc[start:end].copy()

        out_row: Dict[str, float] = {
            "window_id": i,
            "start_idx": start,
            "end_idx": end,
        }

        # Optional metadata if present
        for meta in ["t_start", "t_end", "n_samples"]:
            if meta in w.columns:
                out_row[meta] = w.loc[i, meta]

        # Guard against empty windows
        if len(win) == 0:
            # fill nothing else; keep row so alignment stays correct
            rows.append(out_row)
            continue

        # Compute features
        # TIFEX expects DataFrame with columns specified; output is indexed by column names
        try:
            if args.features == "stat":
                feat_df = extraction.calculate_statistical_features(win, stat_params, columns=acc_cols, njobs=args.njobs)
                out_row.update(flatten_axis_feature_df(feat_df))
            elif args.features == "spec":
                feat_df = extraction.calculate_spectral_features(win, spec_params, columns=acc_cols, njobs=args.njobs)
                out_row.update(flatten_axis_feature_df(feat_df))
            elif args.features == "tf":
                feat_df = extraction.calculate_time_frequency_features(win, tf_params, columns=acc_cols, njobs=args.njobs)
                out_row.update(flatten_axis_feature_df(feat_df))
            elif args.features == "all":
                feat_df = extraction.calculate_all_features(
                    win, stat_params, spec_params, tf_params, columns=acc_cols, njobs=args.njobs
                )
                out_row.update(flatten_axis_feature_df(feat_df))
            else:
                raise ValueError(f"Unknown --features {args.features}")
        except Exception as e:
            # If something unexpected blows up (not just TIFEX internal NaNs),
            # keep the window metadata and record the error string.
            out_row["error"] = str(e)

        rows.append(out_row)

        # light progress
        if (i + 1) % 200 == 0:
            print(f"[tifex_from_windows] processed {i+1}/{n} windows", file=sys.stderr)

    out_df = pd.DataFrame(rows)

    ensure_dir_for_file(args.out)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {len(out_df)} windows x features to: {args.out}")
    print(f"Columns: {len(out_df.columns)}")


if __name__ == "__main__":
    main()
