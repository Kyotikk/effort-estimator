#!/usr/bin/env python3
# features/manual_features_imu.py
"""
Fast manual IMU feature extraction (TOP feature set only).
Replaces TIFEX for IMU to avoid slow extraction.

Input:
- preprocessed IMU CSV with signal columns (e.g., acc_x_dyn, acc_y_dyn, acc_z_dyn)
- windows CSV with start_idx, end_idx (and optional t_start/t_center/t_end/n_samples/win_sec)

Output:
- one row per window with metadata + TOP_FEATURE_COLUMNS
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Top feature allowlist (EXACT)
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
# Config
# ----------------------------
@dataclass
class ManualIMUFeatureConfig:
    # SampEn defaults
    sampen_m: int = 2
    sampen_r_factor: float = 0.2  # r = factor * std

    # ApEn defaults (r parsed from feature name: 0.1, 0.9, ...)
    apen_m: int = 2

    # Tsallis entropy histogram settings
    tsallis_q: float = 2.0
    tsallis_bins: int = 32

    # lower_complete_moment r
    lcm_r: int = 2

    # cardinality rounding for float stability
    cardinality_round_decimals: int = 3

    # numerical stability
    eps: float = 1e-12


# ----------------------------
# Utilities
# ----------------------------
def _as_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x


def _parse_feature_name(feature: str) -> Tuple[str, str]:
    if "__" not in feature:
        raise ValueError(f"Invalid feature name (missing '__'): {feature}")
    sig, fname = feature.split("__", 1)
    return sig, fname


def _parse_quantile(fname: str) -> float:
    # "quantile_0.4" -> 0.4
    return float(fname.split("_", 1)[1])


def _parse_apen_r(fname: str) -> float:
    # "approximate_entropy_0.1" -> 0.1
    return float(fname.split("_")[-1])


# ----------------------------
# Feature primitives
# ----------------------------
def harmonic_mean(x: np.ndarray, eps: float = 1e-12) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    denom = np.mean(1.0 / np.clip(x, eps, None))
    return 1.0 / denom if denom > 0 else np.nan


def harmonic_mean_of_abs(x: np.ndarray, eps: float = 1e-12) -> float:
    return harmonic_mean(np.abs(x), eps=eps)


def quantile(x: np.ndarray, q: float) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def max_(x: np.ndarray) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    return float(np.max(x))


def sum_of_absolute_changes(x: np.ndarray) -> float:
    x = _as_float_array(x)
    if x.size < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(x))))


def avg_amplitude_change(x: np.ndarray) -> float:
    x = _as_float_array(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))))


def variance_of_absolute_differences(x: np.ndarray) -> float:
    x = _as_float_array(x)
    if x.size < 3:
        return np.nan
    d = np.abs(np.diff(x))
    return float(np.var(d, ddof=0))


def cardinality(x: np.ndarray, round_decimals: int = 3) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    xr = np.round(x, round_decimals)
    return float(np.unique(xr).size)


def katz_fractal_dimension(x: np.ndarray, eps: float = 1e-12) -> float:
    x = _as_float_array(x)
    n = x.size
    if n < 2:
        return np.nan
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    if L < eps or d < eps:
        return np.nan
    return float(np.log10(n) / (np.log10(d / L) + np.log10(n)))


def lower_complete_moment(x: np.ndarray, r: int = 2) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    mu = float(np.mean(x))
    below = x[x < mu]
    if below.size == 0:
        return 0.0
    return float(np.mean((mu - below) ** r))


# ----------------------------
# Entropy
# ----------------------------
def _phi_approx_entropy(x: np.ndarray, m: int, r: float) -> float:
    x = _as_float_array(x)
    N = x.size
    if N <= m + 1:
        return np.nan
    Xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    counts = []
    for i in range(Xm.shape[0]):
        dist = np.max(np.abs(Xm - Xm[i]), axis=1)  # Chebyshev
        Ci = np.mean(dist <= r)
        counts.append(Ci)
    c = np.asarray(counts, dtype=float)
    c = np.clip(c, 1e-300, None)
    return float(np.mean(np.log(c)))


def approximate_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    if r is None:
        r = 0.2 * float(np.std(x, ddof=0))
    if r <= 0 or not np.isfinite(r):
        return np.nan
    phi_m = _phi_approx_entropy(x, m=m, r=r)
    phi_m1 = _phi_approx_entropy(x, m=m + 1, r=r)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    x = _as_float_array(x)
    N = x.size
    if N <= m + 1:
        return np.nan
    if r is None:
        r = 0.2 * float(np.std(x, ddof=0))
    if r <= 0 or not np.isfinite(r):
        return np.nan

    Xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    Xm1 = np.array([x[i : i + m + 1] for i in range(N - (m + 1) + 1)])

    def _count_matches(X: np.ndarray) -> int:
        nvec = X.shape[0]
        matches = 0
        for i in range(nvec):
            dist = np.max(np.abs(X - X[i]), axis=1)
            matches += int(np.sum(dist <= r) - 1)  # exclude self
        return matches

    B = _count_matches(Xm)
    A = _count_matches(Xm1)

    if B <= 0 or A <= 0:
        return float("inf")
    return float(-np.log(A / B))


def tsallis_entropy(x: np.ndarray, q: float = 2.0, bins: int = 32, eps: float = 1e-12) -> float:
    x = _as_float_array(x)
    if x.size == 0 or bins < 2:
        return np.nan
    if not np.isfinite(q) or abs(q - 1.0) < eps:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p /= s
    p = p[p > 0]
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


# ----------------------------
# Core function (for pipeline)
# ----------------------------
def compute_top_imu_features_from_windows(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    signal_cols: List[str],
    config: Optional[ManualIMUFeatureConfig] = None,
    top_features: Optional[List[str]] = None,
    quiet: bool = True,
) -> pd.DataFrame:
    cfg = config or ManualIMUFeatureConfig()
    top_features = top_features or TOP_FEATURE_COLUMNS

    for c in ["start_idx", "end_idx"]:
        if c not in windows.columns:
            raise ValueError(f"Windows DataFrame missing '{c}'.")

    missing = [c for c in signal_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing signal columns in data: {missing}")

    per_signal: Dict[str, List[str]] = {s: [] for s in signal_cols}
    for f in top_features:
        sig, _ = _parse_feature_name(f)
        if sig in per_signal:
            per_signal[sig].append(f)
        else:
            if not quiet:
                print(f"[manual_features_imu] ignoring feature for unknown signal '{sig}': {f}", file=sys.stderr)

    rows: List[Dict[str, float]] = []

    for i, w in windows.iterrows():
        start = int(w.start_idx)
        end = int(w.end_idx)

        out: Dict[str, float] = {
            "window_id": float(i),
            "start_idx": float(start),
            "end_idx": float(end),
            "valid": 1.0,
        }

        for meta in ["t_start", "t_center", "t_end", "n_samples", "win_sec"]:
            if meta in windows.columns:
                out[meta] = w[meta]

        if end <= start:
            out["valid"] = 0.0
            rows.append(out)
            continue

        seg = data.iloc[start:end][signal_cols]
        if len(seg) == 0:
            out["valid"] = 0.0
            rows.append(out)
            continue

        try:
            for sig in signal_cols:
                feats_for_sig = per_signal.get(sig, [])
                if not feats_for_sig:
                    continue

                x = _as_float_array(seg[sig].to_numpy(dtype=float, copy=False))
                if x.size == 0:
                    for f in feats_for_sig:
                        out[f] = np.nan
                    continue

                std = float(np.std(x, ddof=0))

                for f in feats_for_sig:
                    _, fname = _parse_feature_name(f)

                    if fname == "harmonic_mean_of_abs":
                        out[f] = harmonic_mean_of_abs(x, eps=cfg.eps)

                    elif fname == "harmonic_mean":
                        out[f] = harmonic_mean(x, eps=cfg.eps)

                    elif fname.startswith("quantile_"):
                        out[f] = quantile(x, _parse_quantile(fname))

                    elif fname == "max":
                        out[f] = max_(x)

                    elif fname == "sum_of_absolute_changes":
                        out[f] = sum_of_absolute_changes(x)

                    elif fname == "avg_amplitude_change":
                        out[f] = avg_amplitude_change(x)

                    elif fname == "variance_of_absolute_differences":
                        out[f] = variance_of_absolute_differences(x)

                    elif fname == "cardinality":
                        out[f] = cardinality(x, round_decimals=cfg.cardinality_round_decimals)

                    elif fname == "katz_fractal_dimension":
                        out[f] = katz_fractal_dimension(x, eps=cfg.eps)

                    elif fname == "sample_entropy":
                        r = cfg.sampen_r_factor * std
                        out[f] = sample_entropy(x, m=cfg.sampen_m, r=r)

                    elif fname.startswith("approximate_entropy_"):
                        r_scale = _parse_apen_r(fname)
                        out[f] = approximate_entropy(x, m=cfg.apen_m, r=r_scale * std)

                    elif fname == "tsallis_entropy":
                        out[f] = tsallis_entropy(x, q=cfg.tsallis_q, bins=cfg.tsallis_bins, eps=cfg.eps)

                    elif fname == "lower_complete_moment":
                        out[f] = lower_complete_moment(x, r=cfg.lcm_r)

                    else:
                        out[f] = np.nan
                        out["valid"] = 0.0
                        out["error"] = f"Unknown feature '{fname}'"

        except Exception as e:
            out["valid"] = 0.0
            out["error"] = str(e)

        rows.append(out)

        if (i + 1) % 200 == 0:
            print(f"[manual_features_imu] processed {i+1}/{len(windows)} windows", file=sys.stderr)

    df_out = pd.DataFrame(rows)

    # Ensure stable schema: all top columns exist
    for c in top_features:
        if c not in df_out.columns:
            df_out[c] = np.nan

    meta_cols = [c for c in ["window_id", "start_idx", "end_idx", "valid", "t_start", "t_center", "t_end", "n_samples", "win_sec", "error"] if c in df_out.columns]
    df_out = df_out[meta_cols + top_features]

    return df_out


# ----------------------------
# CLI (optional)
# ----------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Manual IMU top-feature extraction (no TIFEX).")
    ap.add_argument("--input", required=True, help="Preprocessed IMU CSV")
    ap.add_argument("--windows", required=True, help="Windows CSV with start_idx,end_idx")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--signals", required=True, help="Comma-separated signal cols (e.g., acc_x_dyn,acc_y_dyn,acc_z_dyn)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(args.input)
    windows = pd.read_csv(args.windows)
    signal_cols = [s.strip() for s in args.signals.split(",") if s.strip()]

    out_df = compute_top_imu_features_from_windows(
        data=df,
        windows=windows,
        signal_cols=signal_cols,
        quiet=args.quiet,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} windows -> {args.out}")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
