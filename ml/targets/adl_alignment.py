import numpy as np
import pandas as pd
from pathlib import Path

def parse_adl_intervals(adl_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(adl_csv)

    df = df.rename(columns={
        df.columns[0]: "timestamp",
        df.columns[1]: "event",
        df.columns[2]: "borg",
    })

    # extract date & ms: "04-12-2025-17-46-22" and "075"
    parts = df["timestamp"].astype(str).str.extract(
        r'(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})-(\d{1,6})'
    )
    df["_ts"] = parts[0] + "-" + parts[1].str.pad(6, side="right", fillchar="0")

    # Parse timestamps as JST (UTC+9), then convert to UTC
    df["t_sec"] = pd.to_datetime(
        df["_ts"],
        format="%d-%m-%Y-%H-%M-%S-%f",
        utc=False,  # Parse as naive datetime first
        errors="coerce",   # turns unparsable rows into NaT instead of throwing
    )
    # Localize to JST, then convert to UTC
    df["t_sec"] = df["t_sec"].dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC").astype("int64") / 1e9

    df = df.drop(columns=["_ts"])

    intervals = []
    current = {}

    for _, row in df.iterrows():
        label = str(row["event"])

        if label.endswith("Start"):
            current = {
                "activity": label.replace(" Start", ""),
                "t_start": row["t_sec"],
            }

        elif label.endswith("End") and not pd.isna(row["borg"]):
            current["t_end"] = row["t_sec"]
            current["borg"] = float(row["borg"])
            intervals.append(current)
            current = {}

    return pd.DataFrame(intervals)

def align_windows_to_borg(
    windows: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    out = windows.copy()
    out["borg"] = np.nan

    for _, row in intervals.iterrows():
        mask = (
            (out["t_center"] >= row["t_start"]) &
            (out["t_center"] <= row["t_end"])
        )
        out.loc[mask, "borg"] = row["borg"]

    return out

