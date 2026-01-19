#!/usr/bin/env python3
# features/tifex.py
"""
Run TIFEX feature extraction per precomputed window and write one row per window.

Supports:
- safe mode (conservative stat/spec/tf subsets)
- top mode (compute ONLY the stat calculators needed for a fixed allowlist of columns,
  and optionally filter output to exactly those columns)
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Set

import pandas as pd

from tifex_py.feature_extraction import settings, extraction


# ----------------------------
# Top feature allowlist (EXACT columns as in your screenshot)
# ----------------------------
TOP_FEATURE_COLUMNS: List[str] = [
    "acc_x_dyn__harmonic_mean_of_abs",
    "acc_x_dyn__quantile_0.4",
    "acc_z_dyn__approximate_entropy_0.1",
    "acc_z_dyn__quantile_0.4",
    "acc_x_dyn__sample_entropy",
    "acc_y_dyn__harmonic_mean_of_abs",
    "acc_y_dyn__sample_entropy",
    "acc_z_dyn__sum_of_absolute_changes",
    "acc_y_dyn__avg_amplitude_change",
    "acc_z_dyn__quantile_0.6",
    "acc_z_dyn__variance_of_absolute_differences",
    "acc_x_dyn__quantile_0.6",
    "acc_z_dyn__sample_entropy",
    "acc_y_dyn__variance_of_absolute_differences",
    "acc_x_dyn__max",
    "acc_y_dyn__quantile_0.4",
    "acc_y_dyn__tsallis_entropy",
    "acc_y_dyn__katz_fractal_dimension",
    "acc_x_dyn__cardinality",
    "acc_x_dyn__variance_of_absolute_differences",
    "acc_x_dyn__quantile_0.3",
    "acc_x_dyn__quantile_0.9",
    "acc_z_dyn__harmonic_mean_of_abs",
    "acc_x_dyn__approximate_entropy_0.1",
    "acc_y_dyn__quantile_0.3",
    "acc_z_dyn__lower_complete_moment",
    "acc_x_dyn__harmonic_mean",
    "acc_x_dyn__katz_fractal_dimension",
    "acc_z_dyn__katz_fractal_dimension",
    "acc_x_dyn__approximate_entropy_0.9",
]
TOP_FEATURE_SET: Set[str] = set(TOP_FEATURE_COLUMNS)


# ----------------------------
# Helper functions
# ----------------------------
def flatten_axis_feature_df(feat_df: pd.DataFrame, prefix_sep: str = "__") -> Dict[str, float]:
    """
    Flatten TIFEX output:
      index = signal name (e.g., acc_x_dyn)
      columns = feature names (e.g., mean, quantile_0.4)
    -> {signal__feature: value}  (e.g., acc_x_dyn__quantile_0.4)
    """
    flat: Dict[str, float] = {}
    for signal_name, row in feat_df.iterrows():
        for feat_name, val in row.items():
            flat[f"{signal_name}{prefix_sep}{feat_name}"] = val
    return flat


def _available_calculators(param_obj) -> Optional[List[str]]:
    """
    Try to read calculators list from a params object.
    Some versions expose .calculators, others might not.
    """
    available = getattr(param_obj, "calculators", None)
    return available if isinstance(available, list) else None


# ----------------------------
# Calculator selection
# ----------------------------
def choose_safe_stat_calculators(fs: int) -> settings.StatisticalFeatureParams:
    """
    Conservative statistical subset (generic safe baseline).
    """
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

    default = settings.StatisticalFeatureParams(fs)
    available = _available_calculators(default)
    if available is None:
        return default

    chosen = [c for c in safe_candidates if c in available]
    return settings.StatisticalFeatureParams(fs, calculators=chosen)


def choose_top_stat_calculators(fs: int) -> settings.StatisticalFeatureParams:
    """
    Compute ONLY the statistical calculators needed to produce TOP_FEATURE_COLUMNS.
    Note: calculators are feature-name only (e.g. "quantile_0.4"), NOT axis-prefixed.
    """
    # Extract feature-name part after the first "__"
    # acc_x_dyn__quantile_0.4 -> quantile_0.4
    top_calculators = sorted({name.split("__", 1)[1] for name in TOP_FEATURE_COLUMNS})

    default = settings.StatisticalFeatureParams(fs)
    available = _available_calculators(default)
    if available is None:
        return default

    chosen = [c for c in top_calculators if c in available]
    return settings.StatisticalFeatureParams(fs, calculators=chosen)


def choose_safe_spectral_calculators(fs: int) -> settings.SpectralFeatureParams:
    """
    Conservative spectral subset.
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
    available = _available_calculators(default)
    if available is None:
        return default

    chosen = [c for c in safe_candidates if c in available]
    return settings.SpectralFeatureParams(fs, calculators=chosen)


def choose_safe_timefreq_calculators(fs: int, stat_params: settings.StatisticalFeatureParams) -> settings.TimeFrequencyFeatureParams:
    """
    Conservative time-frequency subset: only TKEO representation (usually robust),
    with provided stat_params for the representation.
    """
    default = settings.TimeFrequencyFeatureParams(fs)
    available = _available_calculators(default)
    if available is None:
        return default

    if "tkeo_features" in available:
        return settings.TimeFrequencyFeatureParams(
            fs,
            calculators=["tkeo_features"],
            tkeo_sf_params=stat_params,
        )

    return default


# ----------------------------
# Core runner
# ----------------------------
def run_tifex(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    fs: int,
    signal_cols: List[str],
    feature_set: str = "stat",
    safe: bool = True,
    mode: str = "safe",            # "safe" or "top"
    filter_top_output: bool = True, # keep only TOP_FEATURE_COLUMNS when mode="top" & feature_set="stat"
    njobs: int = 1,
    quiet: bool = True,
) -> pd.DataFrame:
    """
    Run TIFEX feature extraction on pre-windowed signals (windows defined by start_idx/end_idx).

    mode:
      - "safe": use conservative subsets (generic)
      - "top":  compute only stat calculators needed for TOP_FEATURE_COLUMNS
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
        if mode == "top":
            stat_params = choose_top_stat_calculators(fs)
        else:
            stat_params = choose_safe_stat_calculators(fs)

        spec_params = choose_safe_spectral_calculators(fs)
        tf_params = choose_safe_timefreq_calculators(fs, stat_params=stat_params)
    else:
        stat_params = settings.StatisticalFeatureParams(fs)
        spec_params = settings.SpectralFeatureParams(fs)
        tf_params = settings.TimeFrequencyFeatureParams(fs)

    rows: List[Dict[str, float]] = []

    for i, w in windows.iterrows():
        start = int(w.start_idx)
        end = int(w.end_idx)
        segment = data.iloc[start:end][signal_cols]

        out: Dict[str, object] = {
            "window_id": i,
            "start_idx": start,
            "end_idx": end,
            "valid": True,
        }

        # Optional metadata if present
        for meta in ["t_start", "t_center", "t_end", "n_samples", "win_sec"]:
            if meta in windows.columns:
                out[meta] = w[meta]

        # Guard against empty windows
        if len(segment) == 0:
            out["valid"] = False
            rows.append(out)
            continue

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
                flat = flatten_axis_feature_df(feats)
                if mode == "top" and filter_top_output and feature_set == "stat":
                    # Keep EXACT columns listed in TOP_FEATURE_COLUMNS (plus metadata already in out)
                    flat = {k: v for k, v in flat.items() if k in TOP_FEATURE_SET}
                out.update(flat)

        except Exception as e:
            out["valid"] = False
            out["error"] = str(e)

        rows.append(out)

        if (i + 1) % 200 == 0:
            print(f"[tifex_from_windows] processed {i+1}/{len(windows)} windows", file=sys.stderr)

    return pd.DataFrame(rows)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TIFEX on pre-windowed signals.")
    ap.add_argument("--input", required=True, help="Preprocessed CSV with signal columns")
    ap.add_argument("--windows", required=True, help="Window CSV with start_idx, end_idx")
    ap.add_argument("--out", required=True, help="Output feature CSV")
    ap.add_argument("--fs", required=True, type=float, help="Sampling frequency (Hz)")
    ap.add_argument("--signals", required=True, help="Comma-separated list of signal columns")
    ap.add_argument("--features", choices=["stat", "spec", "tf", "all"], default="stat")
    ap.add_argument("--safe", action="store_true", help="Use conservative feature subsets")
    ap.add_argument("--mode", choices=["safe", "top"], default="safe", help="safe=generic conservative, top=only TOP_FEATURE_COLUMNS (stat)")
    ap.add_argument("--no-filter-top-output", action="store_true", help="When mode=top, do NOT filter output columns to TOP_FEATURE_COLUMNS")
    ap.add_argument("--njobs", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
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
        mode=args.mode,
        filter_top_output=(not args.no_filter_top_output),
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


if __name__ == "__main__":
    main()
