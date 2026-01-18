import re
import numpy as np
import pandas as pd
from pathlib import Path


def _read_adl_any_sep(adl_csv: Path) -> tuple:
    """
    Read ADL CSV robustly. Returns (dataframe, metadata_dict).
    Metadata includes "recording_start" if present.
    """
    adl_csv = Path(adl_csv)
    
    # Try to extract metadata from first 3 lines
    metadata = {}
    try:
        import gzip
        with gzip.open(adl_csv, 'rt') if str(adl_csv).endswith('.gz') else open(adl_csv, 'r') as f:
            lines = [f.readline() for _ in range(3)]
            for line in lines[:2]:  # Check first 2 lines
                if "Start of Recording:" in line:
                    parts = line.split("Start of Recording:")[1].strip().split(",")[0]
                    metadata["recording_start_str"] = parts
    except Exception:
        pass
    
    candidates = []
    for sep in [",", ";", "\t", "|"]:
        try:
            # Read with dtype=str to preserve empty values as empty strings, not NaN
            df = pd.read_csv(adl_csv, sep=sep, compression="infer", skiprows=2, dtype=str)
            candidates.append((df.shape[1], sep, df))
        except Exception:
            pass

    if not candidates:
        df = pd.read_csv(adl_csv, sep=None, engine="python", compression="infer", skiprows=2)
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        df = candidates[0][2]
    
    return df, metadata


def _norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_adl_intervals(adl_csv: Path) -> pd.DataFrame:
    """
    Supports ADL files with:
    A) 2 columns: time, ADLs  (event log with Start/End plus NA/Available noise)
    B) 3 columns: time, ADLs, Borg  (event log + Borg scale attached to End events)
    C) >=3 columns: timestamp, event, borg (legacy)

    Returns:
    activity | t_start | t_end | borg
    """
    df, metadata = _read_adl_any_sep(adl_csv)
    df.columns = [c.strip() for c in df.columns]
    
    # Normalize column names to lowercase for case-insensitive matching
    col_lower = {c: c.lower() for c in df.columns}
    df_renamed = df.rename(columns=col_lower)
    has_time = "time" in df_renamed.columns
    has_adls = "adls" in df_renamed.columns

    # ---------- CASE A/B: 2-3 column event log ----------
    if df.shape[1] in [2, 3] and has_time and has_adls:
        events = df_renamed.rename(columns={"time": "t_str", "adls": "event"}).copy()
        events["event"] = events["event"].astype(str).map(_norm)
        
        # Parse time: try both unix seconds and DD-MM-YYYY-HH-MM-SS-milliseconds format
        t_unix = pd.to_numeric(events["t_str"], errors="coerce")
        
        if t_unix.notna().sum() > 0:
            # Unix seconds format (original)
            events["t_sec"] = t_unix
        else:
            # DD-MM-YYYY-HH-MM-SS-milliseconds format (new CSV format)
            def parse_time_str(ts):
                try:
                    parts = str(ts).split("-")
                    if len(parts) == 7:  # DD-MM-YYYY-HH-MM-SS-milliseconds
                        from datetime import datetime
                        dt_str = f"{parts[2]}-{parts[1]}-{parts[0]} {parts[3]}:{parts[4]}:{parts[5]}"
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        # Assume ADL timestamps are in Japan time (UTC+9)
                        # Convert to UTC first: subtract 9 hours
                        # Then add Zurich time offset (UTC+1): add 1 hour
                        # Net: subtract 8 hours = 28800 seconds
                        return dt.timestamp() + float(f"0.{parts[6]}") - 28800
                except Exception:
                    pass
                return np.nan
            
            events["t_sec"] = events["t_str"].apply(parse_time_str)
        
        # Extract Borg/Effort if present (3-column format)
        has_borg_col = df.shape[1] >= 3
        if has_borg_col:
            # Look for "effort" or "borg" column (case-insensitive)
            borg_cols = [c for c in df_renamed.columns if c.lower() in ["effort", "borg"]]
            if borg_cols:
                events["_borg_val"] = pd.to_numeric(df_renamed[borg_cols[0]], errors="coerce")
            else:
                events["_borg_val"] = np.nan
        else:
            events["_borg_val"] = np.nan
        
        events = events.sort_values("t_sec").reset_index(drop=True)

        # Accept Start/End and ignore NA/Available (and drop any totally unknown formats)
        m = events["event"].str.extract(r"^(.*)\s+(Start|End|NA|Available)$", expand=True)
        events["_base"] = m[0]
        events["_state"] = m[1]
        events = events.dropna(subset=["_base", "_state"]).copy()

        events["activity"] = events["_base"].astype(str).map(_norm)
        events["state"] = events["_state"].astype(str).map(_norm)

        # Keep only Start/End for interval creation
        se = events[events["state"].isin(["Start", "End"])].copy()
        if se.empty:
            raise ValueError(
                f"No Start/End rows found in ADL file. "
                f"States seen: {sorted(events['state'].unique().tolist())}"
            )

        se = se.sort_values("t_sec").reset_index(drop=True)

        intervals = []
        open_start = {}

        for _, row in se.iterrows():
            t = float(row["t_sec"])
            a = row["activity"]
            if row["state"] == "Start":
                open_start[a] = t
            else:  # End
                if a in open_start:
                    t0 = open_start.pop(a)
                    if t > t0:
                        # Attach Borg label if available (use End event's Borg)
                        borg_val = row["_borg_val"] if "_borg_val" in row.index else np.nan
                        intervals.append({
                            "activity": a, 
                            "t_start": t0, 
                            "t_end": t, 
                            "borg": borg_val
                        })

        out = pd.DataFrame(intervals)
        if out.empty:
            # show a few examples to debug pairing issues
            ex = se["event"].head(10).tolist()
            raise ValueError(
                f"Parsed 0 intervals from Start/End rows. "
                f"Examples of Start/End rows: {ex}"
            )

        return out

    # ---------- CASE B: legacy >=3-column format ----------
    if df.shape[1] < 3:
        raise ValueError(
            f"ADL file has {df.shape[1]} columns but must have "
            "either (time, ADLs) or (timestamp, event, borg)."
        )

    df = df.rename(columns={
        df.columns[0]: "timestamp",
        df.columns[1]: "event",
        df.columns[2]: "borg",
    })

    parts = df["timestamp"].astype(str).str.extract(
        r'(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})-(\d{1,6})'
    )
    df["_ts"] = parts[0] + "-" + parts[1].str.pad(6, side="right", fillchar="0")

    df["t_sec"] = pd.to_datetime(
        df["_ts"],
        format="%d-%m-%Y-%H-%M-%S-%f",
        errors="coerce",
    )
    df["t_sec"] = (
        df["t_sec"]
        .dt.tz_localize("Asia/Tokyo")
        .dt.tz_convert("UTC")
        .astype("int64") / 1e9
    )

    df = df.drop(columns=["_ts"])

    intervals = []
    current = {}

    for _, row in df.iterrows():
        label = str(row["event"])
        if label.endswith("Start"):
            current = {"activity": label.replace(" Start", ""), "t_start": row["t_sec"]}
        elif label.endswith("End") and not pd.isna(row["borg"]):
            current["t_end"] = row["t_sec"]
            current["borg"] = float(row["borg"])
            intervals.append(current)
            current = {}

    return pd.DataFrame(intervals)


def align_windows_to_borg(windows: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    out = windows.copy()
    out["borg"] = np.nan

    for _, row in intervals.iterrows():
        mask = (
            (out["t_center"] >= row["t_start"]) &
            (out["t_center"] <= row["t_end"])
        )
        out.loc[mask, "borg"] = row["borg"]

    return out
