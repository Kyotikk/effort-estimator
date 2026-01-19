import argparse
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_windows(
    df: pd.DataFrame,
    fs: float,
    win_sec: float,
    overlap: float,
) -> pd.DataFrame:
    """
    Create sliding windows over uniformly sampled time series.

    If df contains 't_unix', windows are defined in unix seconds.
    Else they are defined in 't_sec'.

    Returns pd.DataFrame with:
      start_idx, end_idx, t_start, t_center, t_end, n_samples, win_sec
    """
    time_col = "t_unix" if "t_unix" in df.columns else "t_sec"

    if time_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{time_col}'. Found {list(df.columns)}")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    # Verify uniform sampling on the grid we actually use
    dt = df[time_col].diff().dropna()
    median_dt = dt.median()
    if not np.allclose(dt, median_dt, rtol=1e-3, atol=1e-6):
        raise ValueError(f"Non-uniform sampling detected in {time_col}. Resample before windowing.")

    win_len = int(round(fs * win_sec))
    if win_len < 2:
        raise ValueError("Window length too short (<2 seconds).")

    hop = int(round(win_len * (1.0 - overlap)))
    hop = max(1, hop)

    n = len(df)
    rows = []

    for start in range(0, n - win_len + 1, hop):
        end = start + win_len

        rows.append({
            "start_idx": int(start),
            "end_idx": int(end),  # end-exclusive
            "t_start": float(df[time_col].iloc[start]),
            "t_center": float(df[time_col].iloc[start + win_len // 2]),
            "t_end": float(df[time_col].iloc[end - 1]),
            "n_samples": int(win_len),
            "win_sec": float(win_sec),
        })

    windows = pd.DataFrame(rows)

    logger.info(
        "Created %d windows (time_col=%s, win_sec=%.3f, overlap=%.2f)",
        len(windows), time_col, win_sec, overlap
    )

    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description="Create sliding windows from uniformly sampled signals.")
    ap.add_argument("--input", required=True, help="Input CSV with time columns (t_unix preferred) and signal")
    ap.add_argument("--out", required=True, help="Output CSV of window indices")
    ap.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz")
    ap.add_argument("--win_sec", type=float, required=True, help="Window length in seconds")
    ap.add_argument("--overlap", type=float, required=True, help="Fractional overlap in [0,1)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    windows = create_windows(
        df=df,
        fs=args.fs,
        win_sec=args.win_sec,
        overlap=args.overlap,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    windows.to_csv(args.out, index=False)

    print(f"Wrote {len(windows)} windows to {args.out}")
    print("Example:")
    print(windows.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
