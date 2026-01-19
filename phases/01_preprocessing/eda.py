from __future__ import annotations

import numpy as np
import pandas as pd


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def preprocess_eda(
    in_path: str,
    out_path: str,
    fs: float = 32.0,
    time_col: str = "time",
    cc_col: str = "cc",
    stress_col: str = "stress_skin",
    quality_col: str = "quality",
    stress_quality_col: str = "stress_skin_quality",
    do_resample: bool = True,
) -> None:
    df = pd.read_csv(in_path)

    need = [time_col, cc_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in {in_path}. Columns: {list(df.columns)[:40]}")

    t_unix = _to_float_series(df[time_col])
    cc = _to_float_series(df[cc_col])

    stress = _to_float_series(df[stress_col]) if stress_col in df.columns else pd.Series(np.nan, index=df.index)
    q = _to_float_series(df[quality_col]) if quality_col in df.columns else pd.Series(np.nan, index=df.index)
    sq = _to_float_series(df[stress_quality_col]) if stress_quality_col in df.columns else pd.Series(np.nan, index=df.index)

    m = np.isfinite(t_unix.to_numpy()) & np.isfinite(cc.to_numpy())
    t_unix = t_unix[m].to_numpy(dtype=float)
    cc = cc[m].to_numpy(dtype=float)
    stress = stress[m].to_numpy(dtype=float)
    q = q[m].to_numpy(dtype=float)
    sq = sq[m].to_numpy(dtype=float)

    if len(t_unix) < 2:
        raise ValueError("Too few valid EDA samples after cleaning.")

    t0 = float(np.min(t_unix))
    t_sec = t_unix - t0

    out = pd.DataFrame({
        "t_unix": t_unix,
        "t_sec": t_sec,
        "eda_cc": cc,
        "eda_stress_skin": stress,
        "eda_quality": q,
        "eda_stress_skin_quality": sq,
    }).sort_values("t_unix").reset_index(drop=True)

    out = out.loc[~out["t_unix"].duplicated(keep="first")].reset_index(drop=True)

    if not do_resample:
        out.to_csv(out_path, index=False)
        print(f"✓ EDA preprocessed → {out_path} | rows={len(out)} | resample=False")
        return

    dt = 1.0 / float(fs)
    t_sec_new = np.arange(out["t_sec"].iloc[0], out["t_sec"].iloc[-1] + 1e-9, dt)

    def interp(col: str) -> np.ndarray:
        y = out[col].to_numpy(dtype=float)
        if not np.any(np.isfinite(y)):
            return np.full_like(t_sec_new, np.nan, dtype=float)
        return np.interp(t_sec_new, out["t_sec"].to_numpy(dtype=float), y)

    cc_new = interp("eda_cc")
    stress_new = interp("eda_stress_skin")

    # carry absolute time by interpolation onto same grid
    t_unix_new = np.interp(
        t_sec_new,
        out["t_sec"].to_numpy(dtype=float),
        out["t_unix"].to_numpy(dtype=float),
    )

    # quality columns: nearest/ffill
    q_ser = pd.Series(out["eda_quality"].to_numpy(dtype=float), index=out["t_sec"].to_numpy(dtype=float))
    sq_ser = pd.Series(out["eda_stress_skin_quality"].to_numpy(dtype=float), index=out["t_sec"].to_numpy(dtype=float))

    q_new = (
        q_ser.reindex(t_sec_new, method="nearest", tolerance=dt * 2)
        .ffill().bfill()
        .to_numpy(dtype=float)
    )
    sq_new = (
        sq_ser.reindex(t_sec_new, method="nearest", tolerance=dt * 2)
        .ffill().bfill()
        .to_numpy(dtype=float)
    )

    out2 = pd.DataFrame({
        "t_unix": t_unix_new,
        "t_sec": t_sec_new,
        "eda_cc": cc_new,
        "eda_stress_skin": stress_new,
        "eda_quality": q_new,
        "eda_stress_skin_quality": sq_new,
    })

    out2.to_csv(out_path, index=False)
    print(f"✓ EDA preprocessed → {out_path} | rows={len(out2)} | fs={fs}")
