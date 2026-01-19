from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class UnixSpec:
    # possible raw timestamp column names you might see
    candidates: tuple[str, ...] = (
        "t_unix", "unix", "epoch", "timestamp", "time_unix",
        "t", "time", "ts", "timestamp_ms", "time_ms", "ts_ms",
        "timestamp_us", "time_us", "ts_us",
        "timestamp_ns", "time_ns", "ts_ns",
    )

def ensure_t_unix(
    df: pd.DataFrame,
    fs_hz: float | None = None,
    unix_col: str = "t_unix",
) -> pd.DataFrame:
    """
    Guarantees df has a float-seconds epoch column `t_unix`.

    If epoch exists in ms/us/ns -> converts.
    If only sample_idx exists + fs_hz + first timestamp exists -> reconstructs.
    Otherwise raises with a clear message.
    """
    df = df.copy()

    # If already present, normalize to float seconds
    if unix_col in df.columns:
        df[unix_col] = pd.to_numeric(df[unix_col], errors="coerce")
        if df[unix_col].dropna().empty:
            raise ValueError(f"{unix_col} exists but is all NaN.")
        # heuristic: if values look like ms (1e12) or ns (1e18), convert
        v = float(df[unix_col].dropna().iloc[0])
        if v > 1e15:   # ns
            df[unix_col] = df[unix_col] / 1e9
        elif v > 1e12: # us
            df[unix_col] = df[unix_col] / 1e6
        elif v > 1e10: # ms
            df[unix_col] = df[unix_col] / 1e3
        return df

    # Find any candidate column
    spec = UnixSpec()
    found = [c for c in spec.candidates if c in df.columns]
    if found:
        c = found[0]
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dropna().empty:
            raise ValueError(f"Found timestamp column '{c}' but it is all NaN.")
        v = float(s.dropna().iloc[0])

        # unit conversion by magnitude + name hints
        if ("_ns" in c) or (v > 1e15):
            df[unix_col] = s / 1e9
        elif ("_us" in c) or (v > 1e12):
            df[unix_col] = s / 1e6
        elif ("_ms" in c) or (v > 1e10):
            df[unix_col] = s / 1e3
        else:
            df[unix_col] = s.astype(float)

        return df

    # Reconstruction path (only if you still have some index + fs)
    idx_candidates = [c for c in ("sample_idx", "idx", "i", "packet_counter") if c in df.columns]
    if fs_hz is not None and idx_candidates:
        ic = idx_candidates[0]
        i = pd.to_numeric(df[ic], errors="coerce")
        if i.dropna().empty:
            raise ValueError(f"Index column '{ic}' exists but is all NaN.")
        i0 = float(i.dropna().iloc[0])
        # use relative time since first sample (no absolute anchor -> not allowed)
        raise ValueError(
            f"No epoch timestamp column found. I can’t create absolute unix time from '{ic}' alone. "
            f"You need at least one epoch value (start time) or keep the original timestamp column."
        )

    raise ValueError(
        "No usable timestamp column found to build t_unix. "
        f"Columns present: {list(df.columns)}"
    )
