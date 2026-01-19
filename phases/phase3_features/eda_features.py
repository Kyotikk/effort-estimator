#!/usr/bin/env python3
# features/eda_features.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Found: {list(df.columns)[:60]}")


def _safe_skew(x: np.ndarray) -> float:
    v = pd.Series(x).skew()
    return float(v) if np.isfinite(v) else np.nan


def _safe_kurt(x: np.ndarray) -> float:
    v = pd.Series(x).kurtosis()
    return float(v) if np.isfinite(v) else np.nan


def _basic_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}

    dx = np.diff(x)

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
        f"{p}rms": float(np.sqrt(np.mean(x**2))),
        f"{p}mean_abs_diff": safe(dx, lambda a: float(np.mean(np.abs(a)))),
        f"{p}slope": safe(np.arange(x.size), lambda t: float(np.polyfit(t, x, 1)[0])),
        f"{p}skew": _safe_skew(x),
        f"{p}kurtosis": _safe_kurt(x),
    }
    return feats


def extract_eda_features(
    eda_csv: str,
    windows_csv: str,
    time_col: str,
    cc_col: str,
    stress_col: str,
    prefix: str,
) -> pd.DataFrame:
    sig = pd.read_csv(eda_csv)
    win = pd.read_csv(windows_csv)

    _require_cols(sig, [time_col, cc_col], "EDA preprocessed CSV")
    _require_cols(win, ["start_idx", "end_idx", "t_start", "t_center", "t_end"], "Windows CSV")

    if "window_id" not in win.columns:
        win = win.reset_index(drop=True)
        win["window_id"] = win.index.astype(int)

    cc = pd.to_numeric(sig[cc_col], errors="coerce").to_numpy(dtype=float)
    stress = pd.to_numeric(sig[stress_col], errors="coerce").to_numpy(dtype=float) if stress_col in sig.columns else None

    rows = []
    for _, w in win.iterrows():
        s = int(w["start_idx"])
        e = int(w["end_idx"])

        feats = {}

        # CC features
        feats.update(_basic_features(cc[s:e], prefix=f"{prefix}cc_"))

        # Stress features (if present)
        if stress is not None:
            feats.update(_basic_features(stress[s:e], prefix=f"{prefix}stress_skin_"))

        # attach window meta (THIS fixes fusion)
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
    ap.add_argument("--eda", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)

    # must match run_pipeline.py call
    ap.add_argument("--fs", required=True)
    ap.add_argument("--time_col", default="t_sec")
    ap.add_argument("--cc_col", default="eda_cc")
    ap.add_argument("--stress_col", default="eda_stress_skin")
    ap.add_argument("--prefix", default="eda_")

    args = ap.parse_args()

    df = extract_eda_features(
        eda_csv=args.eda,
        windows_csv=args.windows,
        time_col=args.time_col,
        cc_col=args.cc_col,
        stress_col=args.stress_col,
        prefix=args.prefix,
    )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    # hard check
    required = ["t_center", "t_start", "t_end", "start_idx", "end_idx", "window_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"BUG: output missing {missing}. cols={list(df.columns)[:40]}")

    print(f"✓ Wrote EDA features → {outp} | rows={len(df)} | cols={df.shape[1]}")


if __name__ == "__main__":
    main()
