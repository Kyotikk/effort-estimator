#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Optional

def preprocess_rr(
    in_path: str,
    out_path: str,
    time_col: str = "time",
    rr_col: str = "rr",
) -> None:
    """
    RR is event-based (irregular sampling). DO NOT resample.
    Output:
      t_sec, rr
    where rr is kept as given (often ms). You can normalize later.
    """
    df = pd.read_csv(in_path, compression="infer")

    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' in {in_path}")
    if rr_col not in df.columns:
        raise ValueError(f"Missing '{rr_col}' in {in_path}. Columns: {list(df.columns)[:30]}")

    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    rr = pd.to_numeric(df[rr_col], errors="coerce").astype(float).to_numpy()

    m = np.isfinite(t) & np.isfinite(rr)
    t, rr = t[m], rr[m]
    if len(t) < 2:
        raise ValueError("Not enough RR samples after cleaning")

    order = np.argsort(t)
    t, rr = t[order], rr[order]
    t_sec = t - float(t[0])

    out = pd.DataFrame({"t_sec": t_sec, "rr": rr}).drop_duplicates("t_sec").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"✓ RR preprocessed → {out_path} | rows={len(out)} | irregular sampling (no resample)")
