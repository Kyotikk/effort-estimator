import argparse
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----- Core window creation -----
def create_windows(
    df: pd.DataFrame,
    fs: float,
    win_sec: float,
    overlap: float,
) -> pd.DataFrame:
    """
    Create sliding windows over uniformly sampled time series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 't_sec'
    fs : float
        Sampling frequency in Hz
    win_sec : float
        Window length in seconds (has to be >= 2 seconds)
    overlap : float
        Fractional overlap in [0, 1)

    Returns
    -------
    pd.DataFrame with columns:
        start_idx, end_idx, t_start, t_center, t_end, n_samples
    """

    # Verify inputs
    if "t_sec" not in df.columns:
        raise ValueError(f"Input DataFrame must contain 't_sec'. Found {list(df.columns)}")
    
    
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")
    
    # Verify uniform sampling
    dt = df["t_sec"].diff().dropna()
    median_dt = dt.median()
    if not np.allclose(dt, median_dt, rtol=1e-3, atol=1e-6):
        raise ValueError("Non-uniform sampling detected. Resample before windowing.")

    # Compute window parameters
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
            "start_idx": start,
            "end_idx": end,  # end-exclusive
            "t_start": float(df["t_sec"].iloc[start]),
            "t_center": float(df["t_sec"].iloc[start + win_len // 2]),
            "t_end": float(df["t_sec"].iloc[end - 1]),
            "n_samples": win_len,
            "win_sec": win_sec,
        })

    windows = pd.DataFrame(rows)

    logger.info(
        "Created %d windows (win_sec=%.3f, overlap=%.2f)",
        len(windows), win_sec, overlap
    )

    return windows

# ----- CLI compatability --> necessary? ----- 
def main() -> None:
    ap = argparse.ArgumentParser(description="Create sliding windows from uniformly sampled signals.")
    ap.add_argument("--input", required=True, help="Input CSV with column 't_sec'")
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