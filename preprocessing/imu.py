import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Helpers
# ----- Butterworth filter functions -----
def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int=4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int=4) ->np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


# ----- Load IMU data -----
def load_imu_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    
    required = ["time", "accX", "accY", "accZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    
    df = df[required].copy()
    df = df.rename(
        columns={
            "time": "t_sec",
            "accX": "acc_x",
            "accY": "acc_y",
            "accZ": "acc_z",
        }
    )

    df[["t_sec", "acc_x", "acc_y", "acc_z"]] = df[
        ["t_sec", "acc_x", "acc_y", "acc_z"]
    ].apply(pd.to_numeric, errors="coerce")

    df = df.dropna().sort_values("t_sec").reset_index(drop=True)

    return df

# ----- Optional resampling -----
def resample_imu(df: pd.DataFrame, fs_out: float) -> pd.DataFrame:
    t = df["t_sec"].values

    t_new = np.arange(t[0], t[-1], 1.0 / fs_out)

    out = {"t_sec": t_new}

    for axis in ["acc_x", "acc_y", "acc_z"]:
        out[axis] = np.interp(t_new, t, df[axis].values)

    return pd.DataFrame(out)

# ===== Filtering =====
# ----- Noise filtering at 5Hz -----
def lowpass_filter_imu(df: pd.DataFrame, fs: float, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    for axis in ["acc_x", "acc_y", "acc_z"]:
        out[axis] = butter_lowpass(out[axis].values, cutoff, fs)

    return out

# ----- Gravity estimation and removal -----
def add_gravity_and_dynamic(
    df: pd.DataFrame,
    fs: float,
    gravity_cutoff: float
) -> pd.DataFrame:
    out = df.copy()

    for axis in ["acc_x", "acc_y", "acc_z"]:
        grav = butter_lowpass(df[axis].values, gravity_cutoff, fs)
        out[f"{axis}_grav"] = grav
        out[f"{axis}_dyn"] = df[axis].values - grav

    return out

# ----- Full preprocessing pipeline -----
# path = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_healthy_3\\corsano_bioz_acc\\2025-12-04.csv.gz"
def preprocess_imu(path: str, fs_out: int, noise_cutoff: float = 5.0, gravity_cutoff: float = 0.3) -> pd.DataFrame:
    """
    Returns a numeric dataframe with:
    - t_sec
    - acc_x, acc_y, acc_z
    - acc_x_dyn, acc_y_dyn, acc_z_dyn
    - acc_x_grav, acc_y_grav, acc_z_grav
    """

    # Load
    df = load_imu_csv(path)

    # Resample
    df = resample_imu(df, fs_out)

    # Activity low-pass
    df = lowpass_filter_imu(df, fs_out, noise_cutoff)

    # Gravity + dynamic
    df = add_gravity_and_dynamic(df, fs_out, gravity_cutoff)

    return df

