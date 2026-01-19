"""
Phase 1: Preprocessing
======================
Cleans and preprocesses raw signals from 7 modalities:
- IMU (bioz & wrist)
- PPG (green, infra, red)
- EDA
- RR

Each function:
1. Loads raw CSV/CSV.gz
2. Validates columns
3. Applies signal conditioning (filtering, normalization)
4. Returns normalized DataFrame with t_unix, t_sec, and cleaned signals
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pathlib import Path


# ============================================================================
# UTILITIES: Filters & Loaders
# ============================================================================

def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth band-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def _load_raw_csv(path: str, required_cols: list) -> pd.DataFrame:
    """Load CSV/CSV.gz and validate required columns."""
    path = str(path)
    df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Found: {list(df.columns)[:20]}")
    
    return df


def _normalize_time(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Normalize unix time to t_unix and t_sec columns."""
    df["t_unix"] = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    df = df.dropna(subset=["t_unix"]).sort_values("t_unix").reset_index(drop=True)
    t0 = float(df["t_unix"].iloc[0])
    df["t_sec"] = df["t_unix"] - t0
    return df


# ============================================================================
# IMU PREPROCESSING (bioz & wrist)
# ============================================================================

def preprocess_imu(path: str, fs: float = 125.0, lowcut: float = 0.5, highcut: float = None) -> pd.DataFrame:
    """
    Preprocess IMU signal (accelerometer).
    
    Args:
        path: Path to raw IMU CSV/CSV.gz (columns: time, accX, accY, accZ)
        fs: Sampling frequency (Hz)
        lowcut: Band-pass lower cutoff (Hz)
        highcut: Band-pass upper cutoff (Hz). If None, uses fs/2.5 (safe default)
    
    Returns:
        DataFrame with t_unix, t_sec, acc_x, acc_y, acc_z (filtered)
    """
    if highcut is None:
        highcut = fs / 2.5  # Safe default: well below Nyquist
    
    df = _load_raw_csv(path, ["time", "accX", "accY", "accZ"])
    df = df[["time", "accX", "accY", "accZ"]].copy()
    
    df = _normalize_time(df)
    
    # Parse signals
    df["acc_x"] = pd.to_numeric(df["accX"], errors="coerce").astype(float)
    df["acc_y"] = pd.to_numeric(df["accY"], errors="coerce").astype(float)
    df["acc_z"] = pd.to_numeric(df["accZ"], errors="coerce").astype(float)
    
    # Remove rows with NaN
    df = df.dropna(subset=["acc_x", "acc_y", "acc_z"])
    
    # Apply band-pass filter
    df["acc_x"] = butter_bandpass(df["acc_x"].values, lowcut, highcut, fs)
    df["acc_y"] = butter_bandpass(df["acc_y"].values, lowcut, highcut, fs)
    df["acc_z"] = butter_bandpass(df["acc_z"].values, lowcut, highcut, fs)
    
    # Compute dynamic component (subtract DC)
    df["acc_x_dyn"] = df["acc_x"] - df["acc_x"].mean()
    df["acc_y_dyn"] = df["acc_y"] - df["acc_y"].mean()
    df["acc_z_dyn"] = df["acc_z"] - df["acc_z"].mean()
    
    return df[["t_unix", "t_sec", "acc_x", "acc_y", "acc_z", "acc_x_dyn", "acc_y_dyn", "acc_z_dyn"]]


# ============================================================================
# PPG PREPROCESSING (green, infra, red)
# ============================================================================

def preprocess_ppg(path: str, fs: float = 32.0, signal_col: str = "value", lowcut: float = 0.4, highcut: float = 5.0) -> pd.DataFrame:
    """
    Preprocess PPG signal.
    
    Args:
        path: Path to raw PPG CSV/CSV.gz (columns: time, signal_col)
        fs: Sampling frequency (Hz)
        signal_col: Name of the PPG signal column (default: "value")
        lowcut: Band-pass lower cutoff (Hz)
        highcut: Band-pass upper cutoff (Hz)
    
    Returns:
        DataFrame with t_unix, t_sec, ppg_signal
    """
    df = _load_raw_csv(path, ["time", signal_col])
    df = df[["time", signal_col]].copy()
    
    df = _normalize_time(df)
    
    df["ppg_signal"] = pd.to_numeric(df[signal_col], errors="coerce").astype(float)
    df = df.dropna(subset=["ppg_signal"])
    
    # Apply band-pass filter
    df["ppg_signal"] = butter_bandpass(df["ppg_signal"].values, lowcut, highcut, fs)
    
    return df[["t_unix", "t_sec", "ppg_signal"]]


# ============================================================================
# EDA PREPROCESSING
# ============================================================================

def preprocess_eda(path: str, fs: float = 32.0, signal_col: str = "cz", lowcut: float = 0.05, highcut: float = 5.0) -> pd.DataFrame:
    """
    Preprocess EDA (electrodermal activity) signal.
    
    Args:
        path: Path to raw EDA CSV/CSV.gz (columns: time, signal_col)
        fs: Sampling frequency (Hz)
        signal_col: Name of the EDA signal column (default: "cz" for skin conductance)
        lowcut: Band-pass lower cutoff (Hz)
        highcut: Band-pass upper cutoff (Hz)
    
    Returns:
        DataFrame with t_unix, t_sec, eda_signal
    """
    df = _load_raw_csv(path, ["time", signal_col])
    df = df[["time", signal_col]].copy()
    
    df = _normalize_time(df)
    
    df["eda_signal"] = pd.to_numeric(df[signal_col], errors="coerce").astype(float)
    df = df.dropna(subset=["eda_signal"])
    
    # Apply band-pass filter
    df["eda_signal"] = butter_bandpass(df["eda_signal"].values, lowcut, highcut, fs)
    
    return df[["t_unix", "t_sec", "eda_signal"]]


# ============================================================================
# RR PREPROCESSING (Respiration Rate / R-R Interval)
# ============================================================================

def preprocess_rr(path: str, fs: float = 1.0, signal_col: str = "rr") -> pd.DataFrame:
    """
    Preprocess RR (respiration rate / inter-beat interval) signal.
    
    Args:
        path: Path to raw RR CSV/CSV.gz (columns: time, signal_col)
        fs: Sampling frequency (Hz) - typically 1 Hz for R-R intervals
        signal_col: Name of the RR signal column (default: "rr")
    
    Returns:
        DataFrame with t_unix, t_sec, rr_signal
    """
    df = _load_raw_csv(path, ["time", signal_col])
    df = df[["time", signal_col]].copy()
    
    df = _normalize_time(df)
    
    df["rr_signal"] = pd.to_numeric(df[signal_col], errors="coerce").astype(float)
    df = df.dropna(subset=["rr_signal"])
    
    # RR is typically already clean; apply light low-pass filter if fs > 1 Hz
    if fs > 1.5:
        df["rr_signal"] = butter_lowpass(df["rr_signal"].values, cutoff=0.5, fs=fs)
    
    return df[["t_unix", "t_sec", "rr_signal"]]
