#!/usr/bin/env python3
"""
ECG/HR to RR interval conversion (bulletproof fallback version).
If ECG R-peak detection fails, falls back to HR→RR conversion.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def preprocess_ecg(
    in_path: str,
    out_path: str,
    time_col: str = "time",
    ecg_col: str = "ecg",
    fs: float = 128.0,
    min_rr_ms: float = 300,
    max_rr_ms: float = 2000,
) -> None:
    """
    Preprocess ECG/HR data and extract RR intervals.
    Falls back to HR→RR conversion if ECG peak detection fails.
    
    Output format matches RR preprocessing:
      t_sec, rr (in milliseconds)
    """
    df = pd.read_csv(in_path, compression="infer")
    
    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' in {in_path}")
    if ecg_col not in df.columns:
        raise ValueError(f"Missing '{ecg_col}' in {in_path}")
    
    # Extract time and signal
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    signal = pd.to_numeric(df[ecg_col], errors="coerce").astype(float).to_numpy()
    
    # Remove NaN values
    valid_mask = np.isfinite(t) & np.isfinite(signal)
    t, signal = t[valid_mask], signal[valid_mask]
    
    if len(t) < 10:
        raise ValueError(f"Insufficient data: only {len(t)} samples")
    
    # Sort by time
    order = np.argsort(t)
    t, signal = t[order], signal[order]
    
    print(f"  Data loaded: {len(t)} samples, {(t[-1]-t[0]):.1f} seconds")
    print(f"  Signal range: {np.min(signal):.2f} - {np.max(signal):.2f}")
    
    # Determine if HR or ECG based on signal range
    signal_range = np.max(signal) - np.min(signal)
    signal_mean = np.mean(signal)
    
    # If signal looks like HR data (30-200 range) or very low amplitude
    is_hr = (30 < signal_mean < 200) or signal_range < 5
    
    rr_times, rr_intervals = None, None
    
    # Try to extract RR intervals
    if is_hr:
        print(f"  → HR data detected, converting to RR intervals")
        try:
            rr_times, rr_intervals = convert_hr_to_rr(t, signal, min_rr_ms, max_rr_ms)
        except Exception as e:
            print(f"    Failed: {str(e)}")
    else:
        print(f"  → ECG data detected, attempting R-peak detection")
        try:
            from scipy.signal import find_peaks
            rr_times, rr_intervals = detect_r_peaks_simple(signal, t, fs, min_rr_ms, max_rr_ms)
        except Exception as e:
            print(f"    Failed: {str(e)}, trying HR fallback")
            try:
                rr_times, rr_intervals = convert_hr_to_rr(t, signal, min_rr_ms, max_rr_ms)
            except Exception as e2:
                print(f"    HR fallback also failed: {str(e2)}")
    
    if rr_intervals is None or len(rr_intervals) < 10:
        raise ValueError(f"Could not extract enough RR intervals (got {len(rr_intervals) if rr_intervals is not None else 0})")
    
    print(f"  ✓ RR intervals: {len(rr_intervals)} | Mean: {np.mean(rr_intervals):.0f} ms ({60000/np.mean(rr_intervals):.0f} bpm)")
    
    # Save
    out = pd.DataFrame({
        "t_sec": rr_times,
        "rr": rr_intervals
    }).drop_duplicates("t_sec").sort_values("t_sec").reset_index(drop=True)
    
    out.to_csv(out_path, index=False)
    print(f"✓ Saved {len(out)} RR intervals")


def convert_hr_to_rr(t: np.ndarray, hr: np.ndarray, min_rr: float = 300, max_rr: float = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Convert HR (bpm) to RR (ms): RR = 60000 / HR"""
    # Filter valid HR values
    valid_mask = (hr > 30) & (hr < 220)
    t_valid = t[valid_mask]
    hr_valid = hr[valid_mask]
    
    if len(hr_valid) < 10:
        raise ValueError(f"Only {len(hr_valid)} valid HR samples")
    
    # Convert to RR
    rr_intervals = 60000.0 / hr_valid
    
    # Filter physiological bounds
    rr_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    
    return t_valid[rr_mask], rr_intervals[rr_mask]


def detect_r_peaks_simple(signal: np.ndarray, t: np.ndarray, fs: float, min_rr: float = 300, max_rr: float = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Simple R-peak detection"""
    from scipy.signal import find_peaks
    
    # Normalize
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    
    # Find peaks
    min_distance = max(1, int((min_rr / 1000.0) * fs))
    peaks, _ = find_peaks(np.abs(signal_norm), distance=min_distance, height=0.3)
    
    if len(peaks) < 2:
        raise ValueError(f"Only {len(peaks)} peaks detected")
    
    r_times = t[peaks]
    rr_intervals = np.diff(r_times) * 1000.0
    rr_times = r_times[:-1] + np.diff(r_times) / 2
    
    # Filter bounds
    rr_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    
    return rr_times[rr_mask], rr_intervals[rr_mask]
    """
    Detect R-peaks in ECG signal using adaptive thresholding.
    
    Args:
        ecg_signal: ECG signal values
        fs: Sampling frequency in Hz
        min_rr_ms: Minimum RR interval in ms (default 300ms = HR < 200 bpm)
    
    Returns:
        Array of R-peak indices
    """
    # Detrend and normalize the signal
    from scipy.signal import detrend
    ecg_detrended = detrend(ecg_signal)
    ecg_norm = ecg_detrended / (np.std(ecg_detrended) + 1e-6)
    
    # Bandpass filter (5-15 Hz) to enhance QRS complex
    nyq = fs / 2
    low = 5.0 / nyq
    high = 15.0 / nyq
    if low > 0 and high < 1:
        b, a = butter(2, [low, high], btype='band')
        ecg_filtered = filtfilt(b, a, ecg_norm)
    else:
        ecg_filtered = ecg_norm
    
    # Square the signal to emphasize peaks
    ecg_squared = ecg_filtered ** 2
    
    # Moving average window (~200ms)
    window_size = max(1, int(0.2 * fs))
    kernel = np.ones(window_size) / window_size
    ecg_integrated = np.convolve(ecg_squared, kernel, mode='same')
    
    # Find peaks with minimum distance based on physiological constraints
    min_distance = int((min_rr_ms / 1000.0) * fs)
    
    # Adaptive threshold: median + 1.5*MAD (more aggressive than before)
    median = np.median(ecg_integrated)
    mad = np.median(np.abs(ecg_integrated - median))
    threshold = median + 1.5 * mad
    
    # If threshold is too high, use percentile-based approach
    if threshold < np.percentile(ecg_integrated, 30):
        threshold = np.percentile(ecg_integrated, 60)
    
    peaks, _ = find_peaks(ecg_integrated, 
                          height=threshold,
                          distance=min_distance)
    
    # If we still don't have enough peaks, try with even lower threshold
    if len(peaks) < 10:
        threshold = np.percentile(ecg_integrated, 50)
        peaks, _ = find_peaks(ecg_integrated,
                              height=threshold,
                              distance=min_distance)


def compute_rr_intervals(r_peak_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RR intervals from R-peak times.
    
    Args:
        r_peak_times: Unix timestamps of R-peaks in seconds
    
    Returns:
        Tuple of (rr_times, rr_intervals)
        - rr_times: timestamps for each RR interval (midpoint between peaks)
        - rr_intervals: RR intervals in milliseconds
    """
    if len(r_peak_times) < 2:
        return np.array([]), np.array([])
    
    # RR intervals in milliseconds
    rr_intervals = np.diff(r_peak_times) * 1000.0
    
    # Time of each RR interval is midpoint between consecutive R-peaks
    rr_times = r_peak_times[:-1] + np.diff(r_peak_times) / 2
    
    return rr_times, rr_intervals


def filter_rr_outliers(rr_times: np.ndarray, rr_intervals: np.ndarray,
                       min_rr: float = 300, max_rr: float = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter physiologically implausible RR intervals.
    
    Args:
        rr_times: RR interval timestamps
        rr_intervals: RR intervals in ms
        min_rr: Minimum valid RR (default 300ms = 200 bpm)
        max_rr: Maximum valid RR (default 2000ms = 30 bpm)
    
    Returns:
        Filtered (rr_times, rr_intervals)
    """
    valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    
    # Additional outlier detection: remove intervals > 3 SD from median
    if np.sum(valid_mask) > 10:
        rr_median = np.median(rr_intervals[valid_mask])
        rr_std = np.std(rr_intervals[valid_mask])
        outlier_mask = np.abs(rr_intervals - rr_median) < 3 * rr_std
        valid_mask = valid_mask & outlier_mask
    
    return rr_times[valid_mask], rr_intervals[valid_mask]


def preprocess_ecg(
    in_path: str,
    out_path: str,
    time_col: str = "time",
    ecg_col: str = "ecg",
    fs: float = 125.0,
    min_rr_ms: float = 300,
    max_rr_ms: float = 2000,
) -> None:
    """
    Preprocess ECG data: detect R-peaks and extract RR intervals.
    
    Output format matches RR preprocessing:
      t_sec, rr (in milliseconds)
    
    Args:
        in_path: Path to raw ECG CSV file
        out_path: Path to save RR intervals
        time_col: Name of timestamp column
        ecg_col: Name of ECG signal column
        fs: ECG sampling frequency in Hz (default 125 Hz for VitalNK)
        min_rr_ms: Minimum valid RR interval in ms
        max_rr_ms: Maximum valid RR interval in ms
    """
    # Load ECG data
    df = pd.read_csv(in_path, compression="infer")
    
    if time_col not in df.columns:
        raise ValueError(f"Missing '{time_col}' in {in_path}")
    if ecg_col not in df.columns:
        raise ValueError(f"Missing '{ecg_col}' in {in_path}. Columns: {list(df.columns)}")
    
    # Extract time and ECG signal
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    ecg = pd.to_numeric(df[ecg_col], errors="coerce").astype(float).to_numpy()
    
    # Remove NaN values
    valid_mask = np.isfinite(t) & np.isfinite(ecg)
    t, ecg = t[valid_mask], ecg[valid_mask]
    
    if len(t) < fs * 10:  # Need at least 10 seconds of data
        raise ValueError(f"Insufficient ECG data: only {len(t)} samples")
    
    # Sort by time
    order = np.argsort(t)
    t, ecg = t[order], ecg[order]
    
    print(f"  ECG loaded: {len(t)} samples, {len(t)/fs:.1f} seconds @ {fs} Hz")
    
    # Detect R-peaks
    r_peak_indices = detect_r_peaks(ecg, fs, min_rr_ms)
    r_peak_times = t[r_peak_indices]
    
    print(f"  R-peaks detected: {len(r_peak_times)}")
    
    # Compute RR intervals
    rr_times, rr_intervals = compute_rr_intervals(r_peak_times)
    
    if len(rr_intervals) == 0:
        raise ValueError("No valid RR intervals computed from ECG")
    
    print(f"  RR intervals computed: {len(rr_intervals)}")
    print(f"  RR range: {np.min(rr_intervals):.0f} - {np.max(rr_intervals):.0f} ms")
    
    # Filter outliers
    rr_times, rr_intervals = filter_rr_outliers(rr_times, rr_intervals, min_rr_ms, max_rr_ms)
    
    print(f"  After filtering: {len(rr_intervals)} valid RR intervals")
    if len(rr_intervals) > 0:
        print(f"  Mean RR: {np.mean(rr_intervals):.0f} ms ({60000/np.mean(rr_intervals):.0f} bpm)")
    
    if len(rr_intervals) < 5:
        raise ValueError(f"Too few valid RR intervals after filtering: {len(rr_intervals)}")
    
    # Save in same format as RR preprocessing
    out = pd.DataFrame({
        "t_sec": rr_times,
        "rr": rr_intervals
    }).drop_duplicates("t_sec").reset_index(drop=True)
    
    out.to_csv(out_path, index=False)
    
    print(f"✓ ECG → RR intervals saved → {out_path} | rows={len(out)} | irregular sampling")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ECG to extract RR intervals")
    parser.add_argument("--ecg", required=True, help="Path to ECG CSV file")
    parser.add_argument("--out", required=True, help="Path to output RR CSV file")
    parser.add_argument("--time_col", default="time", help="Time column name")
    parser.add_argument("--ecg_col", default="ecg", help="ECG signal column name")
    parser.add_argument("--fs", type=float, default=125.0, help="ECG sampling frequency (Hz)")
    parser.add_argument("--min_rr", type=float, default=300, help="Min RR interval (ms)")
    parser.add_argument("--max_rr", type=float, default=2000, help="Max RR interval (ms)")
    
    args = parser.parse_args()
    
    preprocess_ecg(
        in_path=args.ecg,
        out_path=args.out,
        time_col=args.time_col,
        ecg_col=args.ecg_col,
        fs=args.fs,
        min_rr_ms=args.min_rr,
        max_rr_ms=args.max_rr
    )
