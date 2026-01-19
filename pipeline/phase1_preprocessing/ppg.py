import numpy as np
import pandas as pd
from typing import Optional
from scipy import signal


def preprocess_ppg(
    in_path: str,
    out_path: str,
    fs: float = 32.0,
    time_col: str = "time",
    metric_id: Optional[str] = None,
    led_pd_pos: Optional[int] = None,
    led: Optional[int] = None,
    do_resample: bool = True,
    apply_hpf: bool = False,
    hpf_cutoff: float = 0.5,
) -> None:
    df = pd.read_csv(in_path)

    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'. Found: {list(df.columns)[:40]}")
    if "value" not in df.columns:
        raise ValueError("Missing required column 'value'")

    # optional filters
    if metric_id is not None and "metric_id" in df.columns:
        df = df[df["metric_id"].astype(str) == str(metric_id)]
    if led_pd_pos is not None and "led_pd_pos" in df.columns:
        df = df[df["led_pd_pos"].astype("Int64") == int(led_pd_pos)]
    if led is not None and "led" in df.columns:
        df = df[df["led"].astype("Int64") == int(led)]

    if len(df) == 0:
        raise ValueError("Filtering removed all rows — check metric_id / led_pd_pos / led")

    # absolute time (unix seconds, or similar)
    t_unix = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    x = pd.to_numeric(df["value"], errors="coerce").astype(float)

    mask = np.isfinite(t_unix) & np.isfinite(x)
    t_unix = t_unix[mask].to_numpy(dtype=float)
    x = x[mask].to_numpy(dtype=float)

    if len(t_unix) < 2:
        raise ValueError("Too few valid PPG samples after cleaning.")

    # relative time for resampling
    t0 = float(np.min(t_unix))
    t_sec = t_unix - t0

    out = (
        pd.DataFrame({"t_unix": t_unix, "t_sec": t_sec, "value": x})
        .sort_values("t_unix")
        .reset_index(drop=True)
    )

    # Drop duplicate absolute timestamps
    out = out.loc[~out["t_unix"].duplicated(keep="first")].reset_index(drop=True)

    if do_resample:
        dt = 1.0 / float(fs)

        # uniform relative grid
        t_sec_new = np.arange(out["t_sec"].iloc[0], out["t_sec"].iloc[-1] + 1e-9, dt)

        # resample signal on t_sec
        x_new = np.interp(t_sec_new, out["t_sec"].to_numpy(dtype=float), out["value"].to_numpy(dtype=float))

        # carry absolute time by interpolation onto the same grid
        t_unix_new = np.interp(t_sec_new, out["t_sec"].to_numpy(dtype=float), out["t_unix"].to_numpy(dtype=float))

        out = pd.DataFrame({"t_unix": t_unix_new, "t_sec": t_sec_new, "value": x_new})

    # Optional highpass filter for removing baseline drift and enhancing weak signals
    if apply_hpf:
        try:
            # Butterworth 4th order highpass to remove drift while preserving cardiac pulsations
            sos = signal.butter(4, hpf_cutoff, 'hp', fs=fs, output='sos')
            out["value"] = signal.sosfilt(sos, out["value"].to_numpy(dtype=float))
            print(f"  ✓ Applied Butterworth HPF ({hpf_cutoff} Hz) to enhance signal")
        except Exception as e:
            print(f"  ⚠ HPF filtering failed: {e}")

    out.to_csv(out_path, index=False)
    filter_info = f" | HPF={hpf_cutoff}Hz" if apply_hpf else ""
    print(f"✓ PPG preprocessed → {out_path} | rows={len(out)} | fs={fs}{filter_info}")
