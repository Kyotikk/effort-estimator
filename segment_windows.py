import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw CSV with t_sec, acc_x, acc_y, acc_z")
    ap.add_argument("--out", required=True, help="Output CSV of window indices")
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--win_sec", type=float, required=True)
    ap.add_argument("--overlap", type=float, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    required = ["t_sec", "acc_x", "acc_y", "acc_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    fs = int(args.fs)
    win_len = int(round(fs * float(args.win_sec)))
    if win_len < 2:
        raise ValueError("win_len < 2. Increase --win_sec or --fs.")

    if not (0.0 <= float(args.overlap) < 1.0):
        raise ValueError("--overlap must be in [0,1).")

    hop = int(round(win_len * (1.0 - float(args.overlap))))
    hop = max(1, hop)

    n = len(df)
    rows = []
    for start in range(0, n - win_len + 1, hop):
        end = start + win_len  # python slicing end-exclusive
        rows.append({
            "start_idx": start,
            "end_idx": end,
            "t_start": float(df["t_sec"].iloc[start]),
            "t_end": float(df["t_sec"].iloc[end - 1]),
            "n_samples": win_len
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    expected = 0
    if n >= win_len:
        expected = 1 + (n - win_len) // hop

    print(f"Wrote {len(out_df)} windows (expected ~{expected}) to {args.out}")
    print("Example:", out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

