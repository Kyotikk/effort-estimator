#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Optional

def preprocess_bioz(
    in_path: str,
    out_path: str,
    fs: float,
    time_col: str = "time",
    metric_id: Optional[str] = None,
    do_resample: bool = True,
) -> None:
    df = pd.read_csv(in_path, compression="infer")

    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' in {in_path}")
    if "value" not in df.columns:
        raise ValueError(f"Missing 'value' in {in_path}. Columns: {list(df.columns)[:30]}")

    if metric_id is not None and "metric_id" in df.columns:
        df = df[df["metric_id"].astype(str) == str(metric_id)]
        if len(df) == 0:
            raise ValueError("Filtering removed all rows — check metric_id")

    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    x = pd.to_numeric(df["value"], errors="coerce").astype(float).to_numpy()

    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    if len(t) < 2:
        raise ValueError("Not enough valid samples after cleaning")

    order = np.argsort(t)
    t, x = t[order], x[order]
    t_sec = t - float(t[0])

    out = pd.DataFrame({"t_sec": t_sec, "value": x}).drop_duplicates("t_sec").reset_index(drop=True)

    if do_resample and len(out) >= 2:
        dt = 1.0 / fs
        t_new = np.arange(out["t_sec"].iloc[0], out["t_sec"].iloc[-1] + 1e-9, dt)
        x_new = np.interp(t_new, out["t_sec"].to_numpy(), out["value"].to_numpy())
        out = pd.DataFrame({"t_sec": t_new, "value": x_new})

    out.to_csv(out_path, index=False)
    print(f"✓ BIOZ preprocessed → {out_path} | rows={len(out)} | fs={fs}")
