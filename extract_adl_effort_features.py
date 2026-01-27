#!/usr/bin/env python3
"""
Extract EDA and PPG/HRV features per ADL activity for effort estimation.

Aggregates PPG (green) and EDA signals aligned to ADL activity bouts from
parsingsim 3, 4, 5 across all conditions (elderly, healthy, severe).

Output: One row per activity with EDA arousal features + PPG/HRV workload features.

EDA Features (Romine et al. discriminative features for mental effort):
- Tonic: mean/median z-score, std, IQR, coefficient of variation
- Phasic: SCR count/min, sum amplitude, max amplitude
- Distribution: 99th percentile, skewness, kurtosis

PPG/HRV Features (physical/mental workload):
- Mean HR, Mean IBI, SDNN, RMSSD, pNN50
- LF/HF ratio (if window is long enough)

References:
- Romine et al.: EDA features for mental effort discrimination
- Buchheit et al. (2007): HRV recovery & training load
- Plews et al. (2012): Monitoring training with HRV
"""

import numpy as np
import pandas as pd
import os
import glob
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis


# =============================================================================
# EDA Feature Extraction (Effort / Arousal)
# =============================================================================

def decompose_eda(eda_signal: np.ndarray, fs: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose EDA into tonic (SCL) and phasic (SCR) components.
    Uses low-pass filter for tonic extraction.
    
    Args:
        eda_signal: Raw EDA signal (microsiemens)
        fs: Sampling frequency (Hz)
        
    Returns:
        tonic: Skin conductance level (SCL)
        phasic: Skin conductance response (SCR)
    """
    if len(eda_signal) < 10:
        return eda_signal, np.zeros_like(eda_signal)
    
    # Tonic = very low frequency (< 0.05 Hz)
    nyquist = 0.5 * fs
    cutoff = min(0.05 / nyquist, 0.99)  # Ensure valid cutoff
    
    try:
        b, a = butter(3, cutoff, btype='low')
        tonic = filtfilt(b, a, eda_signal)
    except ValueError:
        # If filter fails, use rolling mean as fallback
        window = max(int(fs * 5), 1)  # 5-second window
        tonic = pd.Series(eda_signal).rolling(window, min_periods=1, center=True).mean().values
    
    phasic = eda_signal - tonic
    return tonic, phasic


def detect_scr_peaks(phasic: np.ndarray, fs: float = 4.0,
                     min_amplitude: float = 0.01,
                     min_distance_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect skin conductance response (SCR) peaks in phasic component.
    
    Args:
        phasic: Phasic SCR signal
        fs: Sampling frequency (Hz)
        min_amplitude: Minimum SCR amplitude (µS)
        min_distance_sec: Minimum time between peaks (seconds)
        
    Returns:
        peak_indices: Indices of SCR peaks
        peak_amplitudes: Amplitudes of peaks
    """
    min_distance = max(int(fs * min_distance_sec), 1)
    
    try:
        peaks, properties = find_peaks(phasic, height=min_amplitude, distance=min_distance)
        amplitudes = properties.get('peak_heights', np.array([]))
    except Exception:
        peaks = np.array([])
        amplitudes = np.array([])
    
    return peaks, amplitudes


def extract_eda_features(eda_signal: np.ndarray, fs: float = 4.0,
                         duration_sec: Optional[float] = None) -> Dict[str, float]:
    """
    Extract comprehensive EDA features for effort/arousal estimation.
    
    Based on Romine et al. discriminative features:
    - mean z-score, skewness, kurtosis, 99th percentile
    - global max deflection, global mean, SCR count
    
    Args:
        eda_signal: Raw EDA signal (µS)
        fs: Sampling frequency (Hz)
        duration_sec: Window duration (for rate calculations)
        
    Returns:
        Dictionary of EDA features
    """
    features = {}
    
    if len(eda_signal) < 5 or np.all(np.isnan(eda_signal)):
        # Return NaN features for invalid data
        for key in ['eda_tonic_mean', 'eda_tonic_median', 'eda_tonic_std', 'eda_tonic_iqr',
                    'eda_tonic_cv', 'eda_tonic_zscore_mean', 'eda_tonic_zscore_median',
                    'eda_scr_count', 'eda_scr_rate_per_min', 'eda_scr_sum_amplitude',
                    'eda_scr_max_amplitude', 'eda_p99', 'eda_skewness', 'eda_kurtosis']:
            features[key] = np.nan
        return features
    
    # Clean signal
    eda_clean = eda_signal[np.isfinite(eda_signal)]
    if len(eda_clean) < 5:
        return {k: np.nan for k in features.keys()}
    
    if duration_sec is None:
        duration_sec = len(eda_clean) / fs
    
    # Decompose into tonic and phasic
    tonic, phasic = decompose_eda(eda_clean, fs)
    
    # === TONIC FEATURES (SCL) ===
    features['eda_tonic_mean'] = float(np.mean(tonic))
    features['eda_tonic_median'] = float(np.median(tonic))
    features['eda_tonic_std'] = float(np.std(tonic, ddof=0))
    features['eda_tonic_iqr'] = float(np.percentile(tonic, 75) - np.percentile(tonic, 25))
    
    # Coefficient of variation (dispersion)
    tonic_mean = np.mean(tonic)
    tonic_std = np.std(tonic, ddof=0)
    features['eda_tonic_cv'] = float(tonic_std / (tonic_mean + 1e-10))
    
    # Z-scored tonic (within-window normalization)
    if tonic_std > 1e-10:
        tonic_z = (tonic - tonic_mean) / tonic_std
        features['eda_tonic_zscore_mean'] = float(np.mean(tonic_z))
        features['eda_tonic_zscore_median'] = float(np.median(tonic_z))
    else:
        features['eda_tonic_zscore_mean'] = 0.0
        features['eda_tonic_zscore_median'] = 0.0
    
    # === PHASIC FEATURES (SCR) ===
    peak_indices, peak_amplitudes = detect_scr_peaks(phasic, fs)
    
    features['eda_scr_count'] = len(peak_indices)
    features['eda_scr_rate_per_min'] = float(len(peak_indices) / duration_sec * 60) if duration_sec > 0 else 0.0
    
    if len(peak_amplitudes) > 0:
        features['eda_scr_sum_amplitude'] = float(np.sum(peak_amplitudes))
        features['eda_scr_max_amplitude'] = float(np.max(peak_amplitudes))
    else:
        features['eda_scr_sum_amplitude'] = 0.0
        features['eda_scr_max_amplitude'] = 0.0
    
    # === DISTRIBUTION FEATURES (Peak Intensity) ===
    features['eda_p99'] = float(np.percentile(eda_clean, 99))
    features['eda_skewness'] = float(skew(eda_clean, nan_policy='omit'))
    features['eda_kurtosis'] = float(kurtosis(eda_clean, nan_policy='omit'))
    
    # Additional: global max deflection (max - baseline)
    features['eda_max_deflection'] = float(np.max(eda_clean) - np.percentile(eda_clean, 5))
    features['eda_global_mean'] = float(np.mean(eda_clean))
    
    return features


# =============================================================================
# PPG/HRV Feature Extraction (Effort / Workload)
# =============================================================================

def extract_ibi_from_ppg(ppg_signal: np.ndarray, fs: float = 32.0,
                         min_distance_samples: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract inter-beat intervals (IBI) from PPG signal via peak detection.
    
    Args:
        ppg_signal: PPG waveform (1D array)
        fs: Sampling frequency (Hz)
        min_distance_samples: Minimum samples between peaks
        
    Returns:
        ibi_ms: Inter-beat intervals in milliseconds
        peak_indices: Indices of detected peaks
    """
    if len(ppg_signal) < 100:
        return np.array([]), np.array([])
    
    # Clean signal
    ppg_clean = ppg_signal[np.isfinite(ppg_signal)]
    if len(ppg_clean) < 100:
        return np.array([]), np.array([])
    
    # Bandpass filter (0.5 - 8 Hz) to enhance pulse
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = min(8.0 / nyquist, 0.99)
    
    try:
        b, a = butter(3, [low, high], btype='band')
        ppg_filtered = filtfilt(b, a, ppg_clean)
    except ValueError:
        ppg_filtered = ppg_clean
    
    # Invert if needed (PPG peaks are often negative deflections)
    # Try both orientations and pick the one with more physiological beats
    
    def find_beats(signal, invert=False):
        s = -signal if invert else signal
        threshold = np.mean(s) + 0.5 * np.std(s)
        peaks, _ = find_peaks(s, height=threshold, distance=min_distance_samples)
        return peaks
    
    peaks_normal = find_beats(ppg_filtered, invert=False)
    peaks_inverted = find_beats(ppg_filtered, invert=True)
    
    # Choose orientation with more reasonable beat count
    expected_beats = (len(ppg_clean) / fs) * (60 / 60)  # ~60 bpm baseline
    if abs(len(peaks_inverted) - expected_beats) < abs(len(peaks_normal) - expected_beats):
        peaks = peaks_inverted
    else:
        peaks = peaks_normal
    
    if len(peaks) < 2:
        return np.array([]), np.array([])
    
    # Convert to IBI in milliseconds
    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fs) * 1000
    
    # Filter physiologically plausible IBIs (30-200 bpm → 300-2000 ms)
    valid_mask = (ibi_ms >= 300) & (ibi_ms <= 2000)
    ibi_ms = ibi_ms[valid_mask]
    
    return ibi_ms, peaks


def compute_frequency_domain_hrv(ibi_ms: np.ndarray, fs_resample: float = 4.0) -> Dict[str, float]:
    """
    Compute frequency domain HRV features (LF, HF, LF/HF).
    
    Requires at least 2 minutes of data for reliable LF/HF estimation.
    
    Args:
        ibi_ms: Inter-beat intervals in milliseconds
        fs_resample: Resampling frequency for spectral analysis
        
    Returns:
        Dictionary with LF_power, HF_power, LF_HF_ratio (normalized)
    """
    features = {'hrv_lf_power': np.nan, 'hrv_hf_power': np.nan, 'hrv_lf_hf_ratio': np.nan}
    
    if len(ibi_ms) < 30:  # Need enough beats for spectral analysis
        return features
    
    try:
        from scipy.signal import welch
        from scipy.interpolate import interp1d
        
        # Create time axis for IBI
        t_ibi = np.cumsum(ibi_ms) / 1000  # seconds
        t_ibi = t_ibi - t_ibi[0]  # start at 0
        
        # Interpolate to uniform sampling
        t_uniform = np.arange(0, t_ibi[-1], 1/fs_resample)
        if len(t_uniform) < 60:  # Need ~60 samples minimum
            return features
        
        f_interp = interp1d(t_ibi, ibi_ms, kind='linear', fill_value='extrapolate')
        ibi_uniform = f_interp(t_uniform)
        
        # Compute power spectral density
        freqs, psd = welch(ibi_uniform, fs=fs_resample, nperseg=min(256, len(ibi_uniform)//2))
        
        # LF band: 0.04 - 0.15 Hz
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        
        # HF band: 0.15 - 0.40 Hz
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        
        # Normalize
        total_power = lf_power + hf_power + 1e-10
        features['hrv_lf_power'] = float(lf_power / total_power)
        features['hrv_hf_power'] = float(hf_power / total_power)
        features['hrv_lf_hf_ratio'] = float(lf_power / (hf_power + 1e-10))
        
    except Exception:
        pass
    
    return features


def extract_hrv_features(ppg_signal: np.ndarray, fs: float = 32.0) -> Dict[str, float]:
    """
    Extract HRV features from PPG signal.
    
    Time-domain features:
    - Mean HR, Mean IBI, SDNN, RMSSD, pNN50
    
    Frequency-domain features (if window >= 2 min):
    - LF power, HF power, LF/HF ratio
    
    Args:
        ppg_signal: PPG waveform (1D array)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dictionary of HRV features
    """
    features = {}
    
    # Extract IBI from PPG
    ibi_ms, peaks = extract_ibi_from_ppg(ppg_signal, fs)
    
    if len(ibi_ms) < 3:
        # Return NaN for insufficient data
        for key in ['hrv_mean_hr', 'hrv_mean_ibi', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
                    'hrv_lf_power', 'hrv_hf_power', 'hrv_lf_hf_ratio', 'hrv_n_beats']:
            features[key] = np.nan
        features['hrv_n_beats'] = 0
        return features
    
    # === TIME DOMAIN FEATURES ===
    
    # Mean HR (bpm)
    mean_ibi = np.mean(ibi_ms)
    features['hrv_mean_hr'] = float(60000 / mean_ibi) if mean_ibi > 0 else np.nan
    features['hrv_mean_ibi'] = float(mean_ibi)
    
    # SDNN: Standard deviation of NN intervals
    features['hrv_sdnn'] = float(np.std(ibi_ms, ddof=1))
    
    # RMSSD: Root mean square of successive differences
    successive_diff = np.diff(ibi_ms)
    features['hrv_rmssd'] = float(np.sqrt(np.mean(successive_diff**2)))
    
    # pNN50: Percentage of successive IBIs differing by > 50 ms
    pnn50 = 100 * np.sum(np.abs(successive_diff) > 50) / len(successive_diff)
    features['hrv_pnn50'] = float(pnn50)
    
    # Number of valid beats
    features['hrv_n_beats'] = len(ibi_ms)
    
    # === FREQUENCY DOMAIN FEATURES ===
    freq_features = compute_frequency_domain_hrv(ibi_ms)
    features.update(freq_features)
    
    return features


# =============================================================================
# ADL Parsing and Alignment
# =============================================================================

def parse_adl_file(adl_path: str, time_offset_hours: float = 0.0) -> pd.DataFrame:
    """
    Parse ADL file into activity bouts (start, end, activity name, optional Borg).
    
    Handles both plain CSV and gzipped files.
    
    Args:
        adl_path: Path to ADL CSV file
        time_offset_hours: Time offset to apply (e.g., timezone correction)
        
    Returns:
        DataFrame with columns: [activity, t_start, t_end, duration, borg]
    """
    # Read file (handle gzip)
    if adl_path.endswith('.gz'):
        with gzip.open(adl_path, 'rt') as f:
            # Skip metadata rows if present
            first_line = f.readline()
            f.seek(0)
            
            if 'User ID:' in first_line or 'Start of Recording:' in first_line:
                skiprows = 2
            else:
                skiprows = 0
        
        df = pd.read_csv(adl_path, skiprows=skiprows)
    else:
        # Check for metadata
        with open(adl_path, 'r') as f:
            first_line = f.readline()
        
        skiprows = 2 if ('User ID:' in first_line or 'Start of Recording:' in first_line) else 0
        df = pd.read_csv(adl_path, skiprows=skiprows)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'time' not in df.columns or 'adls' not in df.columns:
        raise ValueError(f"ADL file must have 'time' and 'ADLs' columns. Found: {df.columns.tolist()}")
    
    # Parse time - try Unix timestamp first, then date format
    df['time_sec'] = pd.to_numeric(df['time'], errors='coerce')
    
    if df['time_sec'].isna().all():
        # Try parsing DD-MM-YYYY-HH-MM-SS-milliseconds format
        def parse_time_str(ts):
            try:
                parts = str(ts).split('-')
                if len(parts) == 7:
                    from datetime import datetime
                    dt_str = f"{parts[2]}-{parts[1]}-{parts[0]} {parts[3]}:{parts[4]}:{parts[5]}"
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    return dt.timestamp() + float(f"0.{parts[6]}")
            except Exception:
                pass
            return np.nan
        
        df['time_sec'] = df['time'].apply(parse_time_str)
    
    # Apply time offset
    df['time_sec'] = df['time_sec'] + (time_offset_hours * 3600)
    
    # Check for Borg/Effort column
    borg_col = None
    for col in df.columns:
        if col in ['borg', 'effort']:
            borg_col = col
            break
    
    # Parse Start/End events into intervals
    activities = []
    open_starts = {}
    
    df = df.sort_values('time_sec').reset_index(drop=True)
    
    for _, row in df.iterrows():
        event = str(row['adls']).strip()
        t = row['time_sec']
        
        if pd.isna(t):
            continue
        
        if ' Start' in event:
            activity = event.replace(' Start', '').strip()
            open_starts[activity] = t
        elif ' End' in event:
            activity = event.replace(' End', '').strip()
            if activity in open_starts:
                t_start = open_starts.pop(activity)
                duration = t - t_start
                
                borg_val = np.nan
                if borg_col and borg_col in row.index:
                    borg_val = pd.to_numeric(row[borg_col], errors='coerce')
                
                activities.append({
                    'activity': activity,
                    't_start': t_start,
                    't_end': t,
                    'duration': duration,
                    'borg': borg_val
                })
    
    return pd.DataFrame(activities)


def get_signal_for_interval(signal_df: pd.DataFrame, t_start: float, t_end: float,
                            time_col: str = 't_sec', value_col: str = 'value') -> np.ndarray:
    """
    Extract signal values within a time interval.
    
    Args:
        signal_df: DataFrame with time and value columns
        t_start: Start time (seconds)
        t_end: End time (seconds)
        time_col: Name of time column
        value_col: Name of value column
        
    Returns:
        Signal values within interval
    """
    mask = (signal_df[time_col] >= t_start) & (signal_df[time_col] <= t_end)
    return signal_df.loc[mask, value_col].values


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def find_signal_files(dataset_path: str) -> Dict[str, Optional[str]]:
    """
    Find raw PPG and EDA files for a dataset.
    
    Looks for:
    - PPG green: corsano_wrist_ppg2_green_6/*.csv.gz
    - EDA: corsano_bioz_emography/*.csv.gz
    - ADL: scai_app/ADLs*.csv.gz
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with paths to PPG (green), EDA, and ADL files
    """
    files = {
        'ppg_green': None,
        'eda': None,
        'adl': None
    }
    
    # Find raw PPG green file (prefer latest date)
    ppg_patterns = [
        os.path.join(dataset_path, 'corsano_wrist_ppg2_green_6', '*.csv.gz'),
        os.path.join(dataset_path, 'corsano_wrist_ppg2_green_6', '*.csv'),
    ]
    for pattern in ppg_patterns:
        matches = glob.glob(pattern)
        # Filter out derived files (e.g., *_rr_intervals.csv)
        matches = [m for m in matches if '_rr_' not in m and 'processed' not in m]
        if matches:
            # Pick the one with latest date in filename
            matches.sort(reverse=True)
            files['ppg_green'] = matches[0]
            break
    
    # Find raw EDA file
    eda_patterns = [
        os.path.join(dataset_path, 'corsano_bioz_emography', '*.csv.gz'),
        os.path.join(dataset_path, 'corsano_bioz_emography', '*.csv'),
    ]
    for pattern in eda_patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(reverse=True)
            files['eda'] = matches[0]
            break
    
    # Find ADL file
    adl_patterns = [
        os.path.join(dataset_path, 'scai_app', 'ADLs*.csv.gz'),
        os.path.join(dataset_path, 'scai_app', 'ADLs*.csv'),
    ]
    for pattern in adl_patterns:
        matches = glob.glob(pattern)
        if matches:
            files['adl'] = matches[0]
            break
    
    return files


def process_dataset(dataset_path: str, patient_name: str,
                    adl_offset_hours: float = 0.0,
                    min_duration_sec: float = 30.0,
                    ppg_fs: float = 32.0,
                    eda_fs: float = 4.0) -> Optional[pd.DataFrame]:
    """
    Process a single dataset: extract EDA + HRV features per ADL activity.
    
    Args:
        dataset_path: Path to dataset directory
        patient_name: Patient/condition identifier
        adl_offset_hours: Time offset for ADL file
        min_duration_sec: Minimum activity duration to include
        ppg_fs: PPG sampling frequency
        eda_fs: EDA sampling frequency
        
    Returns:
        DataFrame with one row per activity and all features
    """
    print(f"\nProcessing {patient_name}...")
    
    # Find files
    files = find_signal_files(dataset_path)
    
    if files['adl'] is None:
        print(f"  ⚠️  No ADL file found")
        return None
    
    # Parse ADL
    try:
        adl_df = parse_adl_file(files['adl'], time_offset_hours=adl_offset_hours)
    except Exception as e:
        print(f"  ⚠️  Error parsing ADL: {e}")
        return None
    
    if len(adl_df) == 0:
        print(f"  ⚠️  No activity bouts found")
        return None
    
    # Filter by minimum duration
    adl_df = adl_df[adl_df['duration'] >= min_duration_sec].reset_index(drop=True)
    
    if len(adl_df) == 0:
        print(f"  ⚠️  No activities >= {min_duration_sec}s")
        return None
    
    print(f"  Found {len(adl_df)} activities >= {min_duration_sec}s")
    
    # Load signals
    ppg_df = None
    eda_df = None
    ppg_time_col = 'time'
    ppg_value_col = 'value'
    eda_time_col = 'time'
    eda_value_col = 'cc'  # EDA conductance column
    
    if files['ppg_green']:
        try:
            ppg_df = pd.read_csv(files['ppg_green'])
            # Raw Corsano PPG has 'time' (Unix) and 'value' columns
            ppg_time_col = 'time'
            ppg_value_col = 'value'
            
            # Convert time to float if needed
            ppg_df[ppg_time_col] = pd.to_numeric(ppg_df[ppg_time_col], errors='coerce')
            ppg_df[ppg_value_col] = pd.to_numeric(ppg_df[ppg_value_col], errors='coerce')
            
            # Estimate sampling frequency from data
            if len(ppg_df) > 100:
                dt = ppg_df[ppg_time_col].diff().median()
                if dt > 0:
                    ppg_fs = 1.0 / dt
                    print(f"  ✓ Loaded PPG: {len(ppg_df)} samples (fs≈{ppg_fs:.1f} Hz)")
            else:
                print(f"  ✓ Loaded PPG: {len(ppg_df)} samples")
        except Exception as e:
            print(f"  ⚠️  Error loading PPG: {e}")
            ppg_df = None
    else:
        print(f"  ⚠️  No PPG file found")
    
    if files['eda']:
        try:
            eda_df = pd.read_csv(files['eda'])
            # Raw Corsano EDA has 'time' (Unix) and 'cc' (conductance) columns
            eda_time_col = 'time'
            eda_value_col = 'cc'  # Skin conductance
            
            # Convert time to float if needed
            eda_df[eda_time_col] = pd.to_numeric(eda_df[eda_time_col], errors='coerce')
            eda_df[eda_value_col] = pd.to_numeric(eda_df[eda_value_col], errors='coerce')
            
            # Also load stress_skin if available
            if 'stress_skin' in eda_df.columns:
                eda_df['stress_skin'] = pd.to_numeric(eda_df['stress_skin'], errors='coerce')
            
            # Estimate original sampling frequency
            orig_eda_fs = 1.0
            if len(eda_df) > 2:
                dt = eda_df[eda_time_col].diff().median()
                if dt > 0:
                    orig_eda_fs = 1.0 / dt
            
            # Upsample EDA to target frequency (4 Hz) if very sparse
            target_eda_fs = 4.0
            if orig_eda_fs < 1.0 and len(eda_df) > 2:
                print(f"  ⚡ Upsampling EDA from {orig_eda_fs:.4f} Hz to {target_eda_fs} Hz...")
                
                # Get time range
                t_min = eda_df[eda_time_col].min()
                t_max = eda_df[eda_time_col].max()
                
                # Create uniform time grid
                t_new = np.arange(t_min, t_max, 1.0 / target_eda_fs)
                
                # Interpolate EDA (cc) - use linear interpolation
                eda_vals = eda_df[eda_value_col].values
                eda_times = eda_df[eda_time_col].values
                
                # Remove NaN values for interpolation
                valid_mask = np.isfinite(eda_vals) & np.isfinite(eda_times)
                if np.sum(valid_mask) > 2:
                    interp_func = interp1d(eda_times[valid_mask], eda_vals[valid_mask], 
                                          kind='linear', fill_value='extrapolate', bounds_error=False)
                    eda_interp = interp_func(t_new)
                    
                    # Create new upsampled dataframe
                    eda_df_new = pd.DataFrame({
                        eda_time_col: t_new,
                        eda_value_col: eda_interp
                    })
                    
                    # Also interpolate stress_skin if available
                    if 'stress_skin' in eda_df.columns:
                        stress_vals = eda_df['stress_skin'].values
                        valid_stress = np.isfinite(stress_vals)
                        if np.sum(valid_mask & valid_stress) > 2:
                            interp_stress = interp1d(eda_times[valid_mask & valid_stress], 
                                                    stress_vals[valid_mask & valid_stress],
                                                    kind='linear', fill_value='extrapolate', bounds_error=False)
                            eda_df_new['stress_skin'] = interp_stress(t_new)
                    
                    eda_df = eda_df_new
                    eda_fs = target_eda_fs
                    print(f"  ✓ Upsampled EDA: {len(eda_df)} samples (fs={eda_fs} Hz)")
                else:
                    print(f"  ⚠️  Not enough valid EDA samples for interpolation")
                    eda_fs = orig_eda_fs
            else:
                eda_fs = orig_eda_fs
                print(f"  ✓ Loaded EDA: {len(eda_df)} samples (fs≈{eda_fs:.1f} Hz)")
        except Exception as e:
            print(f"  ⚠️  Error loading EDA: {e}")
            eda_df = None
    else:
        print(f"  ⚠️  No EDA file found")
    
    # Process each activity
    results = []
    
    for idx, activity in adl_df.iterrows():
        row = {
            'patient': patient_name,
            'activity': activity['activity'],
            't_start': activity['t_start'],
            't_end': activity['t_end'],
            'duration': activity['duration'],
            'borg': activity['borg']
        }
        
        # Extract PPG/HRV features
        if ppg_df is not None:
            ppg_signal = get_signal_for_interval(
                ppg_df, activity['t_start'], activity['t_end'],
                time_col=ppg_time_col, value_col=ppg_value_col
            )
            if len(ppg_signal) > 0:
                hrv_feats = extract_hrv_features(ppg_signal, fs=ppg_fs)
                row.update(hrv_feats)
            else:
                # Add NaN HRV features
                for key in ['hrv_mean_hr', 'hrv_mean_ibi', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
                            'hrv_lf_power', 'hrv_hf_power', 'hrv_lf_hf_ratio', 'hrv_n_beats']:
                    row[key] = np.nan
        
        # Extract EDA features
        if eda_df is not None:
            eda_signal = get_signal_for_interval(
                eda_df, activity['t_start'], activity['t_end'],
                time_col=eda_time_col, value_col=eda_value_col
            )
            if len(eda_signal) > 0:
                eda_feats = extract_eda_features(eda_signal, fs=eda_fs, duration_sec=activity['duration'])
                row.update(eda_feats)
            else:
                # Add NaN EDA features
                for key in ['eda_tonic_mean', 'eda_tonic_median', 'eda_tonic_std', 'eda_tonic_iqr',
                            'eda_tonic_cv', 'eda_tonic_zscore_mean', 'eda_tonic_zscore_median',
                            'eda_scr_count', 'eda_scr_rate_per_min', 'eda_scr_sum_amplitude',
                            'eda_scr_max_amplitude', 'eda_p99', 'eda_skewness', 'eda_kurtosis',
                            'eda_max_deflection', 'eda_global_mean']:
                    row[key] = np.nan
        
        results.append(row)
    
    df_out = pd.DataFrame(results)
    print(f"  ✓ Extracted features for {len(df_out)} activities")
    
    return df_out


def main():
    """
    Process all parsingsim datasets (3, 4, 5) × conditions (elderly, healthy, severe).
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract EDA+HRV features per ADL activity')
    parser.add_argument('--base-path', default='/Users/pascalschlegel/data/interim',
                        help='Base path to data directory')
    parser.add_argument('--output', default='adl_effort_features.csv',
                        help='Output CSV file')
    parser.add_argument('--min-duration', type=float, default=30.0,
                        help='Minimum activity duration in seconds')
    parser.add_argument('--ppg-fs', type=float, default=32.0,
                        help='PPG sampling frequency')
    parser.add_argument('--eda-fs', type=float, default=4.0,
                        help='EDA sampling frequency')
    
    args = parser.parse_args()
    
    # Dataset configurations
    # ADL offset varies by dataset (timezone differences)
    # ADL times are often in a different timezone than sensor timestamps
    # Negative offset = ADL times are ahead of sensor times
    dataset_configs = {
        # parsingsim3 - ADL times are ~8h ahead of sensor data
        ('parsingsim3', 'sim_elderly3'): {'adl_offset_hours': -8.3},
        ('parsingsim3', 'sim_healthy3'): {'adl_offset_hours': -8.3},
        ('parsingsim3', 'sim_severe3'): {'adl_offset_hours': -8.3},
        # parsingsim4
        ('parsingsim4', 'sim_elderly4'): {'adl_offset_hours': -8.3},
        ('parsingsim4', 'sim_healthy4'): {'adl_offset_hours': -8.3},
        ('parsingsim4', 'sim_severe4'): {'adl_offset_hours': -8.3},
        # parsingsim5
        ('parsingsim5', 'sim_elderly5'): {'adl_offset_hours': -8.3},
        ('parsingsim5', 'sim_healthy5'): {'adl_offset_hours': -8.3},
        ('parsingsim5', 'sim_severe5'): {'adl_offset_hours': -8.3},
    }
    
    all_results = []
    
    for (sim, condition), config in dataset_configs.items():
        dataset_path = os.path.join(args.base_path, sim, condition)
        
        if not os.path.exists(dataset_path):
            print(f"\n⚠️  Dataset not found: {dataset_path}")
            continue
        
        patient_name = f"{sim}_{condition}"
        
        df = process_dataset(
            dataset_path=dataset_path,
            patient_name=patient_name,
            adl_offset_hours=config['adl_offset_hours'],
            min_duration_sec=args.min_duration,
            ppg_fs=args.ppg_fs,
            eda_fs=args.eda_fs
        )
        
        if df is not None and len(df) > 0:
            # Add dataset metadata
            df['sim'] = sim
            df['condition'] = condition.replace('sim_', '').rstrip('345')
            all_results.append(df)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns for clarity
        meta_cols = ['patient', 'sim', 'condition', 'activity', 't_start', 't_end', 'duration', 'borg']
        hrv_cols = [c for c in combined.columns if c.startswith('hrv_')]
        eda_cols = [c for c in combined.columns if c.startswith('eda_')]
        other_cols = [c for c in combined.columns if c not in meta_cols + hrv_cols + eda_cols]
        
        combined = combined[meta_cols + hrv_cols + eda_cols + other_cols]
        
        # Save
        combined.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"✓ Saved {len(combined)} activity records to {args.output}")
        print(f"  Patients: {combined['patient'].nunique()}")
        print(f"  Activities: {combined['activity'].nunique()}")
        print(f"  HRV features: {len(hrv_cols)}")
        print(f"  EDA features: {len(eda_cols)}")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("Feature summary (non-NaN counts):")
        for col in hrv_cols[:5] + eda_cols[:5]:
            valid = combined[col].notna().sum()
            print(f"  {col}: {valid}/{len(combined)} ({100*valid/len(combined):.1f}%)")
    else:
        print("\n❌ No data extracted from any dataset")


if __name__ == '__main__':
    main()
