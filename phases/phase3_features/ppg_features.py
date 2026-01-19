#!/usr/bin/env python3
# features/ppg_features.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Found: {list(df.columns)[:60]}")


def _safe_skew(x: np.ndarray) -> float:
    s = pd.Series(x)
    v = s.skew()
    return float(v) if np.isfinite(v) else np.nan


def _safe_kurt(x: np.ndarray) -> float:
    s = pd.Series(x)
    v = s.kurtosis()
    return float(v) if np.isfinite(v) else np.nan


def _basic_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}

    dx = np.diff(x)
    ddx = np.diff(dx) if dx.size > 1 else np.array([], dtype=float)

    def safe(arr: np.ndarray, fn, default=np.nan):
        try:
            return fn(arr) if arr.size else default
        except Exception:
            return default

    p = prefix

    feats = {
        f"{p}mean": float(np.mean(x)),
        f"{p}std": float(np.std(x, ddof=0)),
        f"{p}min": float(np.min(x)),
        f"{p}max": float(np.max(x)),
        f"{p}range": float(np.max(x) - np.min(x)),
        f"{p}median": float(np.median(x)),
        f"{p}iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        f"{p}mad": float(np.median(np.abs(x - np.median(x)))),
        f"{p}skew": _safe_skew(x),
        f"{p}kurtosis": _safe_kurt(x),

        f"{p}diff_mean": safe(dx, lambda a: float(np.mean(a))),
        f"{p}diff_std": safe(dx, lambda a: float(np.std(a, ddof=0))),
        f"{p}diff_mean_abs": safe(dx, lambda a: float(np.mean(np.abs(a)))),
        f"{p}diff_rms": safe(dx, lambda a: float(np.sqrt(np.mean(a**2)))),

        f"{p}signal_rms": float(np.sqrt(np.mean(x**2))),
        f"{p}signal_energy": float(np.sum(x**2)),

        f"{p}zcr": float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if x.size > 1 else 0.0,

        f"{p}p1": float(np.percentile(x, 1)),
        f"{p}p5": float(np.percentile(x, 5)),
        f"{p}p10": float(np.percentile(x, 10)),
        f"{p}p90": float(np.percentile(x, 90)),
        f"{p}p95": float(np.percentile(x, 95)),
        f"{p}p99": float(np.percentile(x, 99)),
        f"{p}p90_p10": float(np.percentile(x, 90) - np.percentile(x, 10)),
        f"{p}p95_p5": float(np.percentile(x, 95) - np.percentile(x, 5)),
        f"{p}p99_p1": float(np.percentile(x, 99) - np.percentile(x, 1)),

        f"{p}trim_mean_10": float(
            pd.Series(x)
              .clip(lower=np.percentile(x, 10), upper=np.percentile(x, 90))
              .mean()
        ),

        f"{p}mean_abs": float(np.mean(np.abs(x))),
        f"{p}rms": float(np.sqrt(np.mean(x**2))),
        f"{p}crest_factor": float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12)),
        f"{p}shape_factor": float((np.sqrt(np.mean(x**2)) + 1e-12) / (np.mean(np.abs(x)) + 1e-12)),
        f"{p}impulse_factor": float((np.max(np.abs(x)) + 1e-12) / (np.mean(np.abs(x)) + 1e-12)),
        f"{p}clearance_factor": float(
            (np.max(np.abs(x)) + 1e-12) / ((np.mean(np.sqrt(np.abs(x))) + 1e-12) ** 2)
        ),
        f"{p}mean_cross_rate": float(
            np.mean((x[1:] - np.mean(x)) * (x[:-1] - np.mean(x)) < 0)
        ) if x.size > 1 else 0.0,

        # derivative “energy”
        f"{p}tke_mean": safe(dx, lambda a: float(np.mean(a**2))),
        f"{p}tke_std": safe(dx, lambda a: float(np.std(a**2, ddof=0))),
        f"{p}tke_mean_abs": safe(dx, lambda a: float(np.mean(np.abs(a**2)))),
        f"{p}tke_p95_abs": safe(dx, lambda a: float(np.percentile(np.abs(a), 95))),

        # derivatives
        f"{p}dx_mean": safe(dx, lambda a: float(np.mean(a))),
        f"{p}dx_std": safe(dx, lambda a: float(np.std(a, ddof=0))),
        f"{p}ddx_mean": safe(ddx, lambda a: float(np.mean(a))),
        f"{p}ddx_std": safe(ddx, lambda a: float(np.std(a, ddof=0))),
        f"{p}dx_kurtosis": safe(dx, lambda a: float(pd.Series(a).kurtosis())),
        f"{p}ddx_kurtosis": safe(ddx, lambda a: float(pd.Series(a).kurtosis())),
    }

    return feats


# -------------------------
# Core
# -------------------------
def extract_ppg_features(
    ppg_csv: str,
    windows_csv: str,
    time_col: str,
    signal_col: str,
    prefix: str,
) -> pd.DataFrame:
    sig = pd.read_csv(ppg_csv)
    win = pd.read_csv(windows_csv)

    _require_cols(sig, [time_col, signal_col], "PPG preprocessed CSV")
    _require_cols(win, ["start_idx", "end_idx", "t_start", "t_center", "t_end"], "Windows CSV")

    if "window_id" not in win.columns:
        win = win.reset_index(drop=True)
        win["window_id"] = win.index.astype(int)

    x = pd.to_numeric(sig[signal_col], errors="coerce").to_numpy(dtype=float)

    rows = []
    for _, w in win.iterrows():
        s = int(w["start_idx"])
        e = int(w["end_idx"])
        feats = _basic_features(x[s:e], prefix=prefix)

        # ALWAYS attach window meta (fusion + alignment need this)
        feats.update({
            "start_idx": s,
            "end_idx": e,
            "window_id": int(w["window_id"]),
            "t_start": float(w["t_start"]),
            "t_center": float(w["t_center"]),
            "t_end": float(w["t_end"]),
        })
        rows.append(feats)

    out = pd.DataFrame(rows)

    meta = ["window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end"]
    cols = meta + [c for c in out.columns if c not in meta]
    return out[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppg", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)

    # These are passed by your run_pipeline.py (must exist, even if we don’t use fs directly)
    ap.add_argument("--fs", required=True)
    ap.add_argument("--time_col", default="t_sec")
    ap.add_argument("--signal_col", default="value")
    ap.add_argument("--prefix", default="ppg_")

    args = ap.parse_args()

    df = extract_ppg_features(
        ppg_csv=args.ppg,
        windows_csv=args.windows,
        time_col=args.time_col,
        signal_col=args.signal_col,
        prefix=args.prefix,
    )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    # Hard check: fail early if meta missing
    required = ["t_center", "t_start", "t_end", "start_idx", "end_idx", "window_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"BUG: output missing {missing}. cols={list(df.columns)[:40]}")

    print(f"✓ Wrote PPG features → {outp} | rows={len(df)} | cols={df.shape[1]}")


if __name__ == "__main__":
    main()
