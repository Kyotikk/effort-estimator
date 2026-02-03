#!/usr/bin/env python3
"""
Extract literature-backed statistical features for HR and IMU effort estimation.

HR Features (from exercise physiology literature):
- Banister TRIMP (Training Impulse) - exponential HR weighting
- Edwards TRIMP - time in HR zones
- HR reserve percentage
- HR recovery indicators
- HR variability proxies

IMU Features (from activity recognition literature):
- Signal Magnitude Area (SMA)
- Mean Amplitude Deviation (MAD)
- Jerk metrics (movement smoothness)
- Activity counts (ActiGraph-style)
- Spectral features (dominant frequency)
- Movement intensity classifications
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from pathlib import Path
import gzip
import json

# Data paths
DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_severe3")
OUTPUT_DIR = DATA_DIR / "effort_estimation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD RAW DATA
# ============================================================================

def load_adl_labels():
    """Load ADL activity labels with Borg ratings."""
    # Try different filename patterns
    adl_file = DATA_DIR / "scai_app" / "ADLs_1-3.csv"
    if not adl_file.exists():
        adl_file = DATA_DIR / "scai_app" / "ADLs_1.csv"
    
    # Skip header rows
    df = pd.read_csv(adl_file, skiprows=2)
    df.columns = ['Time', 'ADLs', 'Effort']
    
    # Parse timestamps (DD-MM-YYYY-HH-MM-SS-mmm)
    def parse_ts(ts_str):
        parts = ts_str.split('-')
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        hour, minute, sec = int(parts[3]), int(parts[4]), int(parts[5])
        ms = int(parts[6]) if len(parts) > 6 else 0
        
        import datetime
        dt = datetime.datetime(year, month, day, hour, minute, sec, ms * 1000)
        ts = dt.timestamp() - 28800  # 8-hour timezone correction
        return ts
    
    # Pair Start/End events
    activities = []
    current_activity = None
    current_start = None
    
    for _, row in df.iterrows():
        adl = row['ADLs']
        time = row['Time']
        effort = row['Effort']
        
        if 'Start' in str(adl):
            activity_name = adl.replace(' Start', '')
            current_activity = activity_name
            current_start = parse_ts(time)
        elif 'End' in str(adl) and current_activity is not None:
            activity_name = adl.replace(' End', '')
            if activity_name == current_activity:
                activities.append({
                    'activity': activity_name,
                    't_start': current_start,
                    't_end': parse_ts(time),
                    'borg': float(effort) if pd.notna(effort) else 0
                })
            current_activity = None
            current_start = None
    
    return pd.DataFrame(activities)


def load_hr_data():
    """Load heart rate data."""
    hr_dir = DATA_DIR / "vivalnk_vv330_heart_rate"
    
    all_hr = []
    for f in sorted(hr_dir.glob("*.gz")):
        df = pd.read_csv(f, compression='gzip')
        # Filter valid HR values (typically 40-200 BPM)
        df = df[(df['hr'] > 40) & (df['hr'] < 200)]
        df = df.rename(columns={'time': 'timestamp', 'hr': 'hr'})
        all_hr.append(df)
    
    if not all_hr:
        return pd.DataFrame(columns=['timestamp', 'hr'])
    
    return pd.concat(all_hr).sort_values('timestamp').reset_index(drop=True)


def load_acc_data():
    """Load acceleration data."""
    acc_dir = DATA_DIR / "vivalnk_vv330_acceleration"
    
    all_acc = []
    for f in sorted(acc_dir.glob("*.gz")):
        df = pd.read_csv(f, compression='gzip')
        # Convert to g (scale factor 1/4096)
        x_g = df['x'].values / 4096.0
        y_g = df['y'].values / 4096.0
        z_g = df['z'].values / 4096.0
        mag = np.sqrt(x_g**2 + y_g**2 + z_g**2)
        
        acc_df = pd.DataFrame({
            'timestamp': df['time'].values,
            'x': x_g,
            'y': y_g,
            'z': z_g,
            'mag': mag
        })
        all_acc.append(acc_df)
    
    if not all_acc:
        return pd.DataFrame(columns=['timestamp', 'x', 'y', 'z', 'mag'])
    
    return pd.concat(all_acc, ignore_index=True).sort_values('timestamp').reset_index(drop=True)


# ============================================================================
# HR FEATURES (Literature-backed)
# ============================================================================

def compute_hr_features(hr_df, t_start, t_end, hr_rest, hr_max):
    """
    Compute HR-based effort features.
    
    Based on:
    - Banister (1991) - TRIMP
    - Edwards (1993) - HR zone TRIMP  
    - Foster (2001) - Session RPE
    """
    mask = (hr_df['timestamp'] >= t_start) & (hr_df['timestamp'] <= t_end)
    hr_segment = hr_df.loc[mask, 'hr'].values
    
    if len(hr_segment) < 2:
        return None
    
    duration_s = t_end - t_start
    duration_min = duration_s / 60
    
    # Basic stats
    hr_mean = np.mean(hr_segment)
    hr_min = np.min(hr_segment)
    hr_max_obs = np.max(hr_segment)
    hr_std = np.std(hr_segment)
    hr_range = hr_max_obs - hr_min
    
    # HR delta (elevation above resting)
    hr_delta = hr_mean - hr_rest
    
    # HR Reserve Percentage (Karvonen formula)
    # %HRR = (HR - HR_rest) / (HR_max - HR_rest) × 100
    hr_reserve = (hr_mean - hr_rest) / (hr_max - hr_rest) if hr_max > hr_rest else 0
    hr_reserve = np.clip(hr_reserve, 0, 1)
    
    # ===== BANISTER TRIMP =====
    # TRIMP = duration × HRr × 0.64 × e^(1.92 × HRr)  [men]
    # Where HRr = (HR - HR_rest) / (HR_max - HR_rest)
    b = 1.92  # Male coefficient (1.67 for female)
    trimp_banister = duration_min * hr_reserve * 0.64 * np.exp(b * hr_reserve)
    
    # ===== EDWARDS TRIMP =====
    # Time in 5 HR zones, weighted 1-5
    # Zone 1: 50-60% HRmax, Zone 2: 60-70%, Zone 3: 70-80%, Zone 4: 80-90%, Zone 5: 90-100%
    zones = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    zone_times = np.zeros(5)
    for hr_val in hr_segment:
        hr_pct = hr_val / hr_max
        for i, (low, high) in enumerate(zip(zones[:-1], zones[1:])):
            if low <= hr_pct < high:
                zone_times[i] += 1
                break
        if hr_pct >= 1.0:
            zone_times[4] += 1
    
    # Weight by zone (1-5)
    trimp_edwards = sum((i+1) * t for i, t in enumerate(zone_times))
    trimp_edwards = trimp_edwards / len(hr_segment) * duration_min if len(hr_segment) > 0 else 0
    
    # ===== HR TREND =====
    # Slope of HR during activity (positive = increasing effort)
    if len(hr_segment) >= 3:
        time_points = np.linspace(0, duration_s, len(hr_segment))
        slope, _, _, _, _ = stats.linregress(time_points, hr_segment)
        hr_slope = slope  # BPM per second
    else:
        hr_slope = 0
    
    # ===== HR VARIABILITY PROXY =====
    # RMSSD approximation from beat-to-beat HR changes
    if len(hr_segment) >= 3:
        hr_diff = np.diff(hr_segment)
        hr_rmssd_proxy = np.sqrt(np.mean(hr_diff**2))
    else:
        hr_rmssd_proxy = 0
    
    # ===== CUMULATIVE HR LOAD =====
    # Traditional: HR_delta × duration
    hr_load_linear = hr_delta * duration_s
    
    # Stevens Power Law: HR_delta × √duration
    hr_load_sqrt = hr_delta * np.sqrt(duration_s)
    
    return {
        # Basic
        'hr_mean': hr_mean,
        'hr_min': hr_min,
        'hr_max': hr_max_obs,
        'hr_std': hr_std,
        'hr_range': hr_range,
        'hr_delta': hr_delta,
        
        # Physiological
        'hr_reserve_pct': hr_reserve * 100,
        
        # TRIMP variants
        'trimp_banister': trimp_banister,
        'trimp_edwards': trimp_edwards,
        
        # Dynamics
        'hr_slope': hr_slope,
        'hr_rmssd_proxy': hr_rmssd_proxy,
        
        # Load metrics
        'hr_load_linear': hr_load_linear,
        'hr_load_sqrt': hr_load_sqrt,
        
        'hr_samples': len(hr_segment)
    }


# ============================================================================
# IMU FEATURES (Literature-backed)
# ============================================================================

def compute_imu_features(acc_df, t_start, t_end):
    """
    Compute IMU-based effort features.
    
    Based on:
    - Mathie (2004) - Activity classification
    - Bouten (1997) - Energy expenditure from accelerometry
    - Bao & Intille (2004) - Activity recognition features
    - ActiGraph activity counts methodology
    """
    mask = (acc_df['timestamp'] >= t_start) & (acc_df['timestamp'] <= t_end)
    segment = acc_df.loc[mask].copy()
    
    if len(segment) < 10:
        return None
    
    duration_s = t_end - t_start
    x = segment['x'].values
    y = segment['y'].values  
    z = segment['z'].values
    mag = segment['mag'].values
    
    # Sampling rate estimation
    dt = np.median(np.diff(segment['timestamp'].values))
    fs = 1.0 / dt if dt > 0 else 25  # Default 25 Hz
    
    # ===== BASIC STATISTICS =====
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    mag_min = np.min(mag)
    mag_max = np.max(mag)
    mag_range = mag_max - mag_min
    
    # RMS of acceleration magnitude
    rms_acc = np.sqrt(np.mean(mag**2))
    
    # ===== SIGNAL MAGNITUDE AREA (SMA) =====
    # SMA = (1/T) × ∫(|x| + |y| + |z|) dt
    # Proxy for total movement intensity (Bouten 1997)
    sma = np.mean(np.abs(x) + np.abs(y) + np.abs(z))
    
    # Normalized SMA (remove gravity)
    # Subtract 1g from magnitude before computing
    mag_dynamic = np.abs(mag - 1.0)  # Dynamic component (gravity removed)
    sma_dynamic = np.mean(mag_dynamic)
    
    # ===== MEAN AMPLITUDE DEVIATION (MAD) =====
    # MAD = (1/N) × Σ|mag - mean(mag)|
    # Robust measure of movement intensity
    mad = np.mean(np.abs(mag - mag_mean))
    
    # ===== JERK (Movement Smoothness) =====
    # Jerk = d(acceleration)/dt
    # High jerk = jerky/effortful movement
    if len(mag) >= 3:
        jerk = np.diff(mag) * fs  # Derivative
        rms_jerk = np.sqrt(np.mean(jerk**2))
        jerk_mean = np.mean(np.abs(jerk))
    else:
        rms_jerk = 0
        jerk_mean = 0
    
    # ===== ACTIVITY COUNTS (ActiGraph-style) =====
    # Count threshold crossings (proxy for step-like movements)
    threshold = 0.1  # 0.1g threshold
    crossings = np.sum(np.abs(np.diff(mag > (1.0 + threshold))))
    activity_counts = crossings / duration_s  # Counts per second
    
    # ===== ACTIVE TIME =====
    # Time spent above movement threshold
    active_threshold = 0.2  # 0.2g above gravity
    active_samples = np.sum(mag_dynamic > active_threshold)
    active_time_s = active_samples / fs
    active_pct = active_time_s / duration_s * 100
    
    # ===== SPECTRAL FEATURES =====
    # Dominant frequency (related to movement periodicity)
    if len(mag) >= 64:
        # Remove DC component
        mag_centered = mag - np.mean(mag)
        
        # Compute FFT
        n = len(mag_centered)
        freqs = np.fft.fftfreq(n, 1/fs)
        fft_vals = np.abs(np.fft.fft(mag_centered))
        
        # Only positive frequencies up to 10 Hz (human movement range)
        pos_mask = (freqs > 0.1) & (freqs < 10)
        if np.any(pos_mask):
            pos_freqs = freqs[pos_mask]
            pos_fft = fft_vals[pos_mask]
            
            # Dominant frequency
            dom_freq = pos_freqs[np.argmax(pos_fft)]
            
            # Spectral entropy (movement regularity)
            psd = pos_fft / np.sum(pos_fft)
            psd = psd[psd > 0]
            spectral_entropy = -np.sum(psd * np.log2(psd))
        else:
            dom_freq = 0
            spectral_entropy = 0
    else:
        dom_freq = 0
        spectral_entropy = 0
    
    # ===== POSTURE/ORIENTATION =====
    # Mean inclination (how tilted from vertical)
    # cos(θ) = z / mag
    inclination = np.mean(np.arccos(np.clip(z / mag, -1, 1))) * 180 / np.pi
    
    # ===== ENERGY PROXY =====
    # Integral of acceleration squared (Bouten's energy expenditure proxy)
    energy_proxy = np.sum(mag_dynamic**2) / fs  # g² × s
    
    # ===== IMU LOAD METRICS =====
    # Linear: RMS × duration
    imu_load_linear = rms_acc * duration_s
    
    # Sqrt: RMS × √duration (Stevens Power Law)
    imu_load_sqrt = rms_acc * np.sqrt(duration_s)
    
    # MAD-based load
    mad_load = mad * np.sqrt(duration_s)
    
    # Jerk-based load (smoothness penalty)
    jerk_load = rms_jerk * np.sqrt(duration_s)
    
    return {
        # Basic stats
        'acc_mag_mean': mag_mean,
        'acc_mag_std': mag_std,
        'acc_mag_range': mag_range,
        'rms_acc': rms_acc,
        
        # Intensity metrics
        'sma': sma,
        'sma_dynamic': sma_dynamic,
        'mad': mad,
        
        # Jerk (smoothness)
        'rms_jerk': rms_jerk,
        'jerk_mean': jerk_mean,
        
        # Activity counts
        'activity_counts_per_s': activity_counts,
        'active_time_s': active_time_s,
        'active_pct': active_pct,
        
        # Spectral
        'dominant_freq': dom_freq,
        'spectral_entropy': spectral_entropy,
        
        # Posture
        'inclination_deg': inclination,
        
        # Energy
        'energy_proxy': energy_proxy,
        
        # Load metrics
        'imu_load_linear': imu_load_linear,
        'imu_load_sqrt': imu_load_sqrt,
        'mad_load': mad_load,
        'jerk_load': jerk_load,
        
        'acc_samples': len(segment)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("EXTRACTING LITERATURE-BACKED EFFORT FEATURES")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    adl_df = load_adl_labels()
    hr_df = load_hr_data()
    acc_df = load_acc_data()
    
    print(f"  Activities: {len(adl_df)}")
    print(f"  HR samples: {len(hr_df)}")
    print(f"  ACC samples: {len(acc_df)}")
    
    # Estimate HR parameters
    hr_rest = hr_df['hr'].quantile(0.05)  # 5th percentile as resting
    hr_max = 220 - 70  # Estimated for ~70 year old (adjust as needed)
    print(f"  HR rest (estimated): {hr_rest:.1f} BPM")
    print(f"  HR max (estimated): {hr_max} BPM")
    print()
    
    # Extract features for each activity
    print("Extracting features...")
    results = []
    
    for _, row in adl_df.iterrows():
        activity = row['activity']
        t_start = row['t_start']
        t_end = row['t_end']
        borg = row['borg']
        duration_s = t_end - t_start
        
        # HR features
        hr_feats = compute_hr_features(hr_df, t_start, t_end, hr_rest, hr_max)
        
        # IMU features
        imu_feats = compute_imu_features(acc_df, t_start, t_end)
        
        if hr_feats is None or imu_feats is None:
            print(f"  Skipping {activity}: insufficient data")
            continue
        
        result = {
            'activity': activity,
            't_start': t_start,
            't_end': t_end,
            'duration_s': duration_s,
            'borg': borg,
            **hr_feats,
            **imu_feats
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # ===== COMPUTE CORRELATIONS =====
    print()
    print("=" * 70)
    print("FEATURE CORRELATIONS WITH BORG CR10")
    print("=" * 70)
    print()
    
    # HR features
    hr_features = ['hr_mean', 'hr_delta', 'hr_reserve_pct', 'trimp_banister', 
                   'trimp_edwards', 'hr_slope', 'hr_load_linear', 'hr_load_sqrt']
    
    # IMU features
    imu_features = ['rms_acc', 'sma', 'sma_dynamic', 'mad', 'rms_jerk',
                    'activity_counts_per_s', 'active_pct', 'energy_proxy',
                    'imu_load_linear', 'imu_load_sqrt', 'mad_load', 'jerk_load']
    
    print("HR FEATURES:")
    print("-" * 50)
    hr_corrs = []
    for feat in hr_features:
        if feat in df.columns:
            r, p = stats.pearsonr(df[feat].fillna(0), df['borg'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:<25} r = {r:+.3f} {sig}")
            hr_corrs.append((feat, r, p))
    
    print()
    print("IMU FEATURES:")
    print("-" * 50)
    imu_corrs = []
    for feat in imu_features:
        if feat in df.columns:
            r, p = stats.pearsonr(df[feat].fillna(0), df['borg'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:<25} r = {r:+.3f} {sig}")
            imu_corrs.append((feat, r, p))
    
    # Best features
    print()
    print("=" * 70)
    print("TOP FEATURES")
    print("=" * 70)
    
    all_corrs = hr_corrs + imu_corrs
    all_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print()
    print("Best overall:")
    for feat, r, p in all_corrs[:5]:
        domain = "HR" if feat in hr_features else "IMU"
        print(f"  {feat:<25} ({domain}) r = {r:+.3f}")
    
    # ===== TEST COMBINATIONS =====
    print()
    print("=" * 70)
    print("COMBINED FEATURES (HR + IMU)")
    print("=" * 70)
    print()
    
    # Z-score features
    def zscore(x):
        return (x - x.mean()) / x.std()
    
    # Test combinations
    combinations = [
        ('hr_load_sqrt', 'imu_load_sqrt', 'Basic sqrt loads'),
        ('hr_load_sqrt', 'mad_load', 'HR sqrt + MAD load'),
        ('trimp_banister', 'mad_load', 'TRIMP + MAD'),
        ('hr_reserve_pct', 'sma_dynamic', 'HR reserve + SMA'),
        ('hr_load_sqrt', 'jerk_load', 'HR sqrt + Jerk'),
    ]
    
    for hr_feat, imu_feat, name in combinations:
        z_hr = zscore(df[hr_feat])
        z_imu = zscore(df[imu_feat])
        
        # Test different weights
        best_r = 0
        best_w = 0
        for w in [0.5, 0.6, 0.7, 0.8, 0.9]:
            combined = w * z_hr + (1-w) * z_imu
            r, _ = stats.pearsonr(combined, df['borg'])
            if abs(r) > abs(best_r):
                best_r = r
                best_w = w
        
        print(f"  {name:<30} r = {best_r:+.3f} (w={best_w:.1f})")
    
    # Save results
    output_file = OUTPUT_DIR / "effort_features_full.csv"
    df.to_csv(output_file, index=False)
    print()
    print(f"Saved: {output_file}")
    
    return df


if __name__ == "__main__":
    df = main()
