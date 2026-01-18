#!/usr/bin/env python3
"""
Temperature preprocessing (Corsano bioz_temperature).

Observed columns:
  time, temp_sk1, temp_sk2, temp_amb

Output:
  t_sec + temp_* columns

Note:
- Temperature is slow → by default NO resampling.
- Later you aggregate into your 10s windows for fusion.
"""

from typing import Optional, Sequence
import numpy as np
import pandas as pd


DEFAULT_COLS = ["temp_sk1", "temp_sk2", "temp_amb"]


def preprocess_temp(
    in_path: str,
    out_path: str,
    time_col: str = "time",
    cols: Optional[Sequence[str]] = None,
    do_resample: bool = False,   # IMPORTANT: default off
    fs: float = 32.0,            # only used if do_resample=True
) -> None:
    df = pd.read_csv(in_path)

    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {in_path}. Columns: {list(df.columns)[:30]}")

    use_cols = list(cols) if cols is not None else DEFAULT_COLS
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {in_path}. Columns: {list(df.columns)[:30]}")

    t = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    out = pd.DataFrame({"t": t})

    for c in use_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    out = out[np.isfinite(out["t"])].copy()
    if len(out) < 2:
        raise ValueError(f"Not enough valid rows in {in_path}")

    t0 = float(out["t"].min())
    out["t_sec"] = out["t"] - t0
    out = out.drop(columns=["t"]).sort_values("t_sec").reset_index(drop=True)
    out = out[~out["t_sec"].duplicated(keep="first")].reset_index(drop=True)

    # Optional uniform resample (usually not needed for temperature)
    if do_resample and len(out) >= 2:
        dt = 1.0 / float(fs)
        t_new = np.arange(out["t_sec"].iloc[0], out["t_sec"].iloc[-1] + 1e-9, dt)
        res = pd.DataFrame({"t_sec": t_new})

        t_old = out["t_sec"].to_numpy()
        for c in use_cols:
            y = out[c].to_numpy()
            m = np.isfinite(y) & np.isfinite(t_old)
            if m.sum() < 2:
                res[c] = np.nan
            else:
                res[c] = np.interp(t_new, t_old[m], y[m])

        out = res

    out = out.rename(columns={c: f"temp_{c.replace('temp_', '')}" for c in use_cols})
    out.to_csv(out_path, index=False)
    print(f"✓ TEMP preprocessed → {out_path} | rows={len(out)} | resample={do_resample}")
