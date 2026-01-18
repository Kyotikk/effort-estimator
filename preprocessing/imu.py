import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# -------------------------
# Filters
# -------------------------
def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


# -------------------------
# Load
# -------------------------
def load_imu_csv(path: str) -> pd.DataFrame:
    """
    Expects raw IMU csv(.gz) with columns:
      time (unix seconds), accX, accY, accZ

    Returns:
      t_unix (epoch seconds), t_sec (relative), acc_x/y/z
    """
    df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)

    required = ["time", "accX", "accY", "accZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Found: {list(df.columns)[:40]}")

    df = df[required].copy()

    df["t_unix"] = pd.to_numeric(df["time"], errors="coerce").astype(float)
    df["acc_x"] = pd.to_numeric(df["accX"], errors="coerce").astype(float)
    df["acc_y"] = pd.to_numeric(df["accY"], errors="coerce").astype(float)
    df["acc_z"] = pd.to_numeric(df["accZ"], errors="coerce").astype(float)

    df = df.dropna(subset=["t_unix", "acc_x", "acc_y", "acc_z"]).sort_values("t_unix").reset_index(drop=True)

    t0 = float(df["t_unix"].iloc[0])
    df["t_sec"] = df["t_unix"] - t0

    return df[["t_unix", "t_sec", "acc_x", "acc_y", "acc_z"]]


# -------------------------
# Resample
# -------------------------
def resample_imu(df: pd.DataFrame, fs_out: float) -> pd.DataFrame:
    """
    Uniform grid in t_sec, interpolate acc axes AND t_unix.
    """
    t_sec = df["t_sec"].to_numpy(dtype=float)
    t_unix = df["t_unix"].to_numpy(dtype=float)

    if len(t_sec) < 2:
        raise ValueError("Too few IMU samples for resampling.")

    dt = 1.0 / float(fs_out)
    t_sec_new = np.arange(t_sec[0], t_sec[-1] + 1e-9, dt)

    out = {
        "t_sec": t_sec_new,
        "t_unix": np.interp(t_sec_new, t_sec, t_unix),
    }

    for axis in ["acc_x", "acc_y", "acc_z"]:
        out[axis] = np.interp(t_sec_new, t_sec, df[axis].to_numpy(dtype=float))

    return pd.DataFrame(out)


# -------------------------
# Filtering
# -------------------------
def lowpass_filter_imu(df: pd.DataFrame, fs: float, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    for axis in ["acc_x", "acc_y", "acc_z"]:
        out[axis] = butter_lowpass(out[axis].to_numpy(dtype=float), cutoff, fs)
    return out


def add_gravity_and_dynamic(df: pd.DataFrame, fs: float, gravity_cutoff: float) -> pd.DataFrame:
    out = df.copy()
    for axis in ["acc_x", "acc_y", "acc_z"]:
        grav = butter_lowpass(df[axis].to_numpy(dtype=float), gravity_cutoff, fs)
        out[f"{axis}_grav"] = grav
        out[f"{axis}_dyn"] = df[axis].to_numpy(dtype=float) - grav
    return out


# -------------------------
# Main callable
# -------------------------
def preprocess_imu(
    path: str,
    fs_out: float,
    noise_cutoff: float = 5.0,
    gravity_cutoff: float = 0.3,
) -> pd.DataFrame:
    """
    Returns:
      t_unix, t_sec,
      acc_x/y/z,
      acc_x/y/z_grav,
      acc_x/y/z_dyn
    """
    df = load_imu_csv(path)
    df = resample_imu(df, fs_out)
    df = lowpass_filter_imu(df, fs_out, noise_cutoff)
    df = add_gravity_and_dynamic(df, fs_out, gravity_cutoff)
    return df
