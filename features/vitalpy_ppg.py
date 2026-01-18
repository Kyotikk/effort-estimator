#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# ---- VitalPy path injection (local vendored repo) ----
VITALPY_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "VitalPy", "src")
)
if VITALPY_SRC not in sys.path:
    sys.path.insert(0, VITALPY_SRC)

from vitalpython.ppg.PPGSignal import PPGSignal  # noqa: E402


def _upsample_linear(t: np.ndarray, x: np.ndarray, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear upsample to fs_target. Assumes t is increasing and in seconds.
    """
    if len(t) < 2:
        return t, x
    dt = 1.0 / float(fs_target)
    t_new = np.arange(float(t[0]), float(t[-1]) + 1e-12, dt)
    x_new = np.interp(t_new, t, x)
    return t_new, x_new


def _as_feature_dict(feats: Any) -> Dict[str, Any]:
    """
    Convert VitalPy return type to a dict safely.
    """
    if feats is None:
        return {}
    if isinstance(feats, dict):
        return feats
    if isinstance(feats, pd.Series):
        return feats.to_dict()
    try:
        return dict(feats)
    except Exception:
        return {}


def _run_one_window_debug(
    t_all: np.ndarray,
    x_all: np.ndarray,
    w: pd.DataFrame,
    idx: int,
    fs_ppg: float,
    fs_vitalpy: float,
) -> None:
    """
    Run VitalPy check_keypoints on a single window and exit.
    """
    if idx < 0 or idx >= len(w):
        raise ValueError(f"--debug_idx {idx} out of range (0..{len(w)-1})")

    r0 = w.iloc[idx]
    s0 = int(r0["start_idx"])
    e0 = int(r0["end_idx"])

    t0 = t_all[s0:e0].copy()
    x0 = x_all[s0:e0].copy()

    if len(t0) < 2:
        raise ValueError("Selected window has <2 samples.")

    # make time relative
    t0 = t0 - t0[0]

    # upsample (important: do it in debug too)
    if fs_vitalpy and fs_vitalpy > fs_ppg:
        t0, x0 = _upsample_linear(t0, x0, fs_target=fs_vitalpy)

    wf0 = pd.DataFrame({"t": t0, "ppg": x0})

    print(f"[DEBUG] window_id={idx} start_idx={s0} end_idx={e0} n={len(wf0)}")
    print(f"[DEBUG] t_start={float(r0.get('t_start', np.nan))} t_end={float(r0.get('t_end', np.nan))}")
    print(f"[DEBUG] fs_ppg={fs_ppg} fs_vitalpy={fs_vitalpy} (upsampled={fs_vitalpy > fs_ppg})")

    sig0 = PPGSignal(wf0, verbose=1)
    sig0.check_keypoints()  # opens plots / prints internal messages


def run_vitalpy_on_windows(
    ppg_csv: str,
    windows_csv: str,
    out_csv: str,
    fs_ppg: float = 32.0,
    fs_vitalpy: float = 128.0,
    time_col: str = "t_sec",
    signal_col: str = "value",
    verbose: int = 0,
    debug_idx: int = -1,
) -> None:
    """
    Extract VitalPy features per window.
    - If debug_idx >= 0: run check_keypoints on that window (verbose=1) and exit.
    - Otherwise: extract_features per window, store failures in _vitalpy_error.
    """
    df = pd.read_csv(ppg_csv)
    w = pd.read_csv(windows_csv)

    if time_col not in df.columns:
        raise ValueError(f"PPG CSV missing time_col='{time_col}'. Found: {list(df.columns)}")
    if signal_col not in df.columns:
        raise ValueError(f"PPG CSV missing signal_col='{signal_col}'. Found: {list(df.columns)}")

    required_w = ["start_idx", "end_idx"]
    missing_w = [c for c in required_w if c not in w.columns]
    if missing_w:
        raise ValueError(f"Windows CSV missing columns {missing_w}. Found: {list(w.columns)}")

    t_all = df[time_col].to_numpy(dtype=float)
    x_all = df[signal_col].to_numpy(dtype=float)

    # ---- Debug single window and exit ----
    if debug_idx >= 0:
        _run_one_window_debug(
            t_all=t_all,
            x_all=x_all,
            w=w,
            idx=debug_idx,
            fs_ppg=fs_ppg,
            fs_vitalpy=fs_vitalpy,
        )
        return

    rows: List[Dict[str, Any]] = []

    for i, r in w.iterrows():
        s = int(r["start_idx"])
        e = int(r["end_idx"])

        t = t_all[s:e]
        x = x_all[s:e]

        # minimal sanity
        if len(t) < 20:
            rows.append({
                "window_id": int(i),
                "t_start": float(r.get("t_start", np.nan)),
                "t_end": float(r.get("t_end", np.nan)),
                "vitalpy_ok": 0,
                "_vitalpy_error": f"too_short:{len(t)}",
            })
            continue

        # time relative per window
        t = t - t[0]

        # upsample so VitalPy is closer to its intended regime
        if fs_vitalpy and fs_vitalpy > fs_ppg:
            t_u, x_u = _upsample_linear(t, x, fs_target=fs_vitalpy)
        else:
            t_u, x_u = t, x

        waveform_df = pd.DataFrame({"t": t_u, "ppg": x_u})

        try:
            sig = PPGSignal(waveform_df, verbose=verbose)
            feats_raw = sig.extract_features()
            feats = _as_feature_dict(feats_raw)

            # attach metadata
            feats["window_id"] = int(i)
            feats["t_start"] = float(r.get("t_start", np.nan))
            feats["t_end"] = float(r.get("t_end", np.nan))
            feats["vitalpy_ok"] = 1 if len(feats) > 3 else 0

            rows.append(feats)

        except Exception as ex:
            rows.append({
                "window_id": int(i),
                "t_start": float(r.get("t_start", np.nan)),
                "t_end": float(r.get("t_end", np.nan)),
                "vitalpy_ok": 0,
                "_vitalpy_error": str(ex),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows to {out_csv}")
    print("Columns:", len(out.columns))
    print(out.head(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract VitalPy PPG features per window (vendored VitalPy).")
    ap.add_argument("--ppg", required=True, help="Preprocessed PPG CSV (must contain time_col + signal_col).")
    ap.add_argument("--windows", required=True, help="Windows CSV from windowing/windows.py.")
    ap.add_argument("--out", required=True, help="Output CSV with one row per window.")
    ap.add_argument("--fs_ppg", type=float, default=32.0, help="Native PPG sampling rate.")
    ap.add_argument("--fs_vitalpy", type=float, default=128.0, help="Upsample rate used before VitalPy.")
    ap.add_argument("--time_col", default="t_sec", help="Time column in PPG CSV.")
    ap.add_argument("--signal_col", default="value", help="Signal column in PPG CSV.")
    ap.add_argument("--verbose", type=int, default=0, help="VitalPy verbose level (0 quiet, 1+ debug).")
    ap.add_argument("--debug_idx", type=int, default=-1, help="If >=0, run check_keypoints() for that window and exit.")
    args = ap.parse_args()

    run_vitalpy_on_windows(
        ppg_csv=args.ppg,
        windows_csv=args.windows,
        out_csv=args.out,
        fs_ppg=args.fs_ppg,
        fs_vitalpy=args.fs_vitalpy,
        time_col=args.time_col,
        signal_col=args.signal_col,
        verbose=args.verbose,
        debug_idx=args.debug_idx,
    )


if __name__ == "__main__":
    main()
