#!/usr/bin/env python3
"""
Compute HRV Recovery Rate from PPG signals during recovery phases.

Replaces Borg labels with physiologically-grounded HRV recovery metrics.

References:
- Buchheit et al. (2007): HRV recovery & training load in soccer
- Plews et al. (2012): Monitoring training with HRV
- Al Haddad et al. (2011): Vagal reactivation index
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


def extract_rr_intervals(ppg_signal, fs=32, min_distance=15):
    """
    Extract beat-to-beat RR intervals from PPG signal.
    
    Args:
        ppg_signal: PPG waveform (1D array)
        fs: Sampling frequency (Hz)
        min_distance: Minimum samples between peaks (prevents double-detection)
        
    Returns:
        rr_intervals: RR intervals in milliseconds
        peak_indices: Indices of detected heartbeats
    """
    if len(ppg_signal) < 100:
        return np.array([]), np.array([])
    
    # Invert signal (peaks = heartbeats, typically low in PPG)
    inverted = -ppg_signal
    
    # Find peaks with adaptive threshold
    threshold = np.mean(inverted) + 0.5 * np.std(inverted)
    peaks, _ = find_peaks(inverted, height=threshold, distance=min_distance)
    
    if len(peaks) < 2:
        return np.array([]), np.array([])
    
    # Convert peak indices to RR intervals (milliseconds)
    peak_intervals = np.diff(peaks)  # samples between beats
    rr_intervals = (peak_intervals / fs) * 1000  # convert to ms
    
    # Filter out physiologically impossible RR intervals
    # Human HR range: 30-200 bpm → RR 300-2000 ms
    rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
    
    return rr_intervals, peaks


def compute_hrv_window(rr_intervals):
    """
    Compute HRV (Heart Rate Variability) from RR intervals.
    
    Returns SDNN: Standard deviation of NN intervals (ms)
    This is the gold-standard HRV metric.
    
    Args:
        rr_intervals: RR intervals in milliseconds (1D array)
        
    Returns:
        hrv_sdnn: Standard deviation of RR intervals (ms)
    """
    if len(rr_intervals) < 2:
        return np.nan
    
    hrv = np.std(rr_intervals)
    return hrv


def compute_hrv_recovery_rate(ppg_signal_baseline, ppg_signal_activity, 
                               ppg_signal_recovery, fs=32):
    """
    Compute HRV Recovery Rate: Speed of HRV return to baseline after activity.
    
    Args:
        ppg_signal_baseline: PPG during rest before activity (1D array)
        ppg_signal_activity: PPG during activity (1D array)
        ppg_signal_recovery: PPG during recovery phase (1D array)
        fs: Sampling frequency (Hz)
        
    Returns:
        hrv_recovery_rate: Recovery rate in ms/second (how fast HRV returns to baseline)
        hrv_baseline: Baseline HRV before activity
        hrv_effort: Minimum HRV during activity
        hrv_recovery: Final HRV after recovery window
        recovery_time_sec: Actual recovery duration
    """
    
    # Extract RR intervals from each phase
    rr_baseline, _ = extract_rr_intervals(ppg_signal_baseline, fs)
    rr_activity, _ = extract_rr_intervals(ppg_signal_activity, fs)
    rr_recovery, _ = extract_rr_intervals(ppg_signal_recovery, fs)
    
    if len(rr_baseline) < 5 or len(rr_activity) < 5 or len(rr_recovery) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Compute HRV for each phase
    hrv_baseline = compute_hrv_window(rr_baseline)  # Rest baseline
    hrv_effort = compute_hrv_window(rr_activity)    # Minimum during activity
    hrv_recovery = compute_hrv_window(rr_recovery)  # After recovery window
    
    # Compute recovery rate: how much HRV recovered per second
    hrv_change = hrv_recovery - hrv_effort  # Change in HRV (ms)
    
    # Estimate recovery duration from signal length
    recovery_time_sec = len(ppg_signal_recovery) / fs
    
    if recovery_time_sec < 10:  # Need at least 10 seconds for reliable estimate
        return np.nan, hrv_baseline, hrv_effort, hrv_recovery, recovery_time_sec
    
    # Recovery rate: ms/second
    hrv_recovery_rate = hrv_change / recovery_time_sec
    
    return hrv_recovery_rate, hrv_baseline, hrv_effort, hrv_recovery, recovery_time_sec


def parse_adl_with_recovery(adl_df, data_start_time, data_end_time, recovery_buffer_sec=300):
    """
    Parse ADL intervals and add recovery phase windows.
    
    Args:
        adl_df: DataFrame with columns [activity, t_start, t_end, borg]
        data_start_time: Unix timestamp of data start
        data_end_time: Unix timestamp of data end
        recovery_buffer_sec: How long after activity to measure recovery (seconds)
        
    Returns:
        enhanced_adl: DataFrame with additional columns:
            - recovery_t_start: When recovery phase starts
            - recovery_t_end: When recovery phase ends
    """
    enhanced_adl = adl_df.copy()
    
    # Add recovery phase (starts at activity end, lasts recovery_buffer_sec)
    enhanced_adl['recovery_t_start'] = enhanced_adl['t_end']
    enhanced_adl['recovery_t_end'] = enhanced_adl['t_end'] + recovery_buffer_sec
    
    # Clip to data boundaries (can't measure recovery after data ends)
    enhanced_adl['recovery_t_end'] = enhanced_adl['recovery_t_end'].clip(
        upper=data_end_time
    )
    
    # Filter out activities where recovery extends beyond data
    enhanced_adl = enhanced_adl[enhanced_adl['recovery_t_end'] > enhanced_adl['recovery_t_start']]
    
    return enhanced_adl


def compute_hrv_recovery_for_activity(ppg_df, activity_row, fs=32):
    """
    Compute HRV recovery rate for a single activity.
    
    Args:
        ppg_df: PPG timeseries DataFrame with columns [t_sec, value] or [time, value]
        activity_row: Row from ADL with columns [t_start, t_end, recovery_t_start, recovery_t_end, borg]
        fs: Sampling frequency (Hz)
        
    Returns:
        hrv_recovery_rate: Recovery rate (ms/sec)
        metadata: Dict with detailed HRV metrics
    """
    
    # Handle both 'time' and 't_sec' column names
    time_col = 'time' if 'time' in ppg_df.columns else 't_sec'
    
    t_start = activity_row['t_start']
    t_end = activity_row['t_end']
    recovery_t_start = activity_row['recovery_t_start']
    recovery_t_end = activity_row['recovery_t_end']
    borg = activity_row['borg']
    
    # Extract baseline (1 min before activity)
    baseline_start = t_start - 60
    baseline_end = t_start
    baseline_ppg = ppg_df[
        (ppg_df[time_col] >= baseline_start) & (ppg_df[time_col] < baseline_end)
    ]['value'].values
    
    # Extract activity phase
    activity_ppg = ppg_df[
        (ppg_df[time_col] >= t_start) & (ppg_df[time_col] < t_end)
    ]['value'].values
    
    # Extract recovery phase
    recovery_ppg = ppg_df[
        (ppg_df[time_col] >= recovery_t_start) & (ppg_df[time_col] < recovery_t_end)
    ]['value'].values
    
    if len(baseline_ppg) < 100 or len(activity_ppg) < 100 or len(recovery_ppg) < 100:
        return np.nan, {
            'borg': borg,
            'hrv_baseline': np.nan,
            'hrv_effort': np.nan,
            'hrv_recovery': np.nan,
            'recovery_time': np.nan,
            'note': 'Insufficient data'
        }
    
    # Compute recovery rate
    hrv_recovery_rate, hrv_baseline, hrv_effort, hrv_recovery, recovery_time = \
        compute_hrv_recovery_rate(baseline_ppg, activity_ppg, recovery_ppg, fs)
    
    metadata = {
        'borg': borg,
        'hrv_baseline': hrv_baseline,
        'hrv_effort': hrv_effort,
        'hrv_recovery': hrv_recovery,
        'recovery_time': recovery_time,
        'hrv_recovery_rate': hrv_recovery_rate
    }
    
    return hrv_recovery_rate, metadata


def align_windows_to_hrv_recovery(windows_df, ppg_df, adl_df, fs=32):
    """
    Assign HRV recovery rates to windows as target variable.
    
    Replaces Borg labels with physiologically-grounded HRV recovery metric.
    
    Args:
        windows_df: Window metadata with columns [t_center, ...]
        ppg_df: PPG timeseries with columns [time/t_sec, value]
        adl_df: ADL intervals with columns [t_start, t_end, borg, ...]
        fs: Sampling frequency (Hz)
        
    Returns:
        labeled_windows_df: Windows with HRV recovery rate as target
            New columns: hrv_recovery_rate, hrv_baseline, hrv_effort, 
                        hrv_recovery, activity_borg
    """
    
    windows_df = windows_df.copy()
    
    # Get time column name
    time_col = 'time' if 'time' in ppg_df.columns else 't_sec'
    
    data_start = ppg_df[time_col].min()
    data_end = ppg_df[time_col].max()
    
    # Parse ADL with recovery phases (5 minutes = 300 seconds)
    adl_enhanced = parse_adl_with_recovery(adl_df, data_start, data_end, recovery_buffer_sec=300)
    
    # Initialize columns
    windows_df['hrv_recovery_rate'] = np.nan
    windows_df['hrv_baseline'] = np.nan
    windows_df['hrv_effort'] = np.nan
    windows_df['hrv_recovery'] = np.nan
    windows_df['activity_borg'] = np.nan
    windows_df['labeled'] = False
    
    print(f"\n▶ Computing HRV Recovery Rates from {len(adl_enhanced)} ADL activities...")
    
    # For each activity, compute HRV recovery and assign to recovery-phase windows
    successful = 0
    for idx, activity in adl_enhanced.iterrows():
        # Compute HRV recovery for this activity
        hrv_rate, metadata = compute_hrv_recovery_for_activity(ppg_df, activity, fs)
        
        if np.isnan(hrv_rate):
            print(f"  ⚠ Skipped activity {idx}: {metadata['note']}")
            continue
        
        # Assign to windows in recovery phase
        recovery_start = activity['recovery_t_start']
        recovery_end = activity['recovery_t_end']
        
        recovery_window_mask = (
            (windows_df['t_center'] >= recovery_start) &
            (windows_df['t_center'] <= recovery_end)
        )
        
        n_assigned = recovery_window_mask.sum()
        
        windows_df.loc[recovery_window_mask, 'hrv_recovery_rate'] = hrv_rate
        windows_df.loc[recovery_window_mask, 'hrv_baseline'] = metadata['hrv_baseline']
        windows_df.loc[recovery_window_mask, 'hrv_effort'] = metadata['hrv_effort']
        windows_df.loc[recovery_window_mask, 'hrv_recovery'] = metadata['hrv_recovery']
        windows_df.loc[recovery_window_mask, 'activity_borg'] = metadata['borg']
        windows_df.loc[recovery_window_mask, 'labeled'] = True
        
        print(f"  ✓ Activity {idx}: HRV_Recovery_Rate={hrv_rate:.3f} ms/s "
              f"(Borg={metadata['borg']}, "
              f"HRV {metadata['hrv_baseline']:.1f}→{metadata['hrv_effort']:.1f}→{metadata['hrv_recovery']:.1f} ms, "
              f"assigned to {n_assigned} windows)")
        
        successful += 1
    
    # Only keep labeled windows
    labeled_windows = windows_df[windows_df['labeled']].copy()
    labeled_windows = labeled_windows.drop('labeled', axis=1)
    
    print(f"\n✓ Successfully processed {successful}/{len(adl_enhanced)} activities")
    print(f"✓ Assigned HRV recovery rates to {len(labeled_windows)} windows")
    
    if len(labeled_windows) > 0:
        print(f"✓ HRV Recovery Rate range: {labeled_windows['hrv_recovery_rate'].min():.3f} to "
              f"{labeled_windows['hrv_recovery_rate'].max():.3f} ms/s")
        print(f"  Interpretation:")
        print(f"    Fast recovery (>1.0 ms/s):  Low effort, good fitness")
        print(f"    Moderate (0.5-1.0 ms/s):   Normal effort")
        print(f"    Slow recovery (<0.5 ms/s): High effort or fatigued")
    
    return labeled_windows


def align_windows_to_hrv_recovery_rr(windows, intervals, rr_path):
    """
    Assign HRV recovery rates to windows using pre-extracted RR intervals.
    
    This function reads RR intervals from a CSV file and aligns them with
    windows during recovery phases of ADL activities.
    
    Args:
        windows: DataFrame with columns [window_id, t_start, t_center, t_end]
        intervals: ADL intervals DataFrame with columns [t_start, t_end, borg, ...]
        rr_path: Path to CSV with RR intervals (columns: time/t_sec, rr_ms or similar)
        
    Returns:
        windows_labeled: Windows with hrv_recovery_rate column
    """
    import pandas as pd
    from pathlib import Path
    
    windows_labeled = windows.copy()
    
    # Load RR intervals
    try:
        rr_df = pd.read_csv(rr_path)
    except Exception as e:
        raise ValueError(f"Cannot load RR intervals from {rr_path}: {e}")
    
    # Handle different column name conventions
    time_col = 'time' if 'time' in rr_df.columns else 't_sec'
    rr_col = 'rr_ms' if 'rr_ms' in rr_df.columns else ('rr' if 'rr' in rr_df.columns else None)
    
    if rr_col is None:
        raise ValueError(f"RR CSV must have 'rr_ms' or 'rr' column. Found: {list(rr_df.columns)}")
    
    # Add hrv_recovery_rate column (initially NaN)
    windows_labeled['hrv_recovery_rate'] = np.nan
    
    data_start = rr_df[time_col].min()
    data_end = rr_df[time_col].max()
    
    # Get RR intervals into a working structure
    print(f"  RR data loaded: {len(rr_df)} samples from {data_start:.1f} to {data_end:.1f}")
    
    # For each ADL activity, compute recovery rate during recovery phase
    recovery_buffer_sec = 300  # 5 minutes
    successful_labels = 0
    
    for idx, activity in intervals.iterrows():
        t_start = activity['t_start']
        t_end = activity['t_end']
        borg = activity.get('borg', np.nan)
        
        # Recovery phase: starts at activity end, lasts recovery_buffer_sec
        recovery_start = t_end
        recovery_end = min(t_end + recovery_buffer_sec, data_end)
        
        # Skip if recovery phase is too short
        if recovery_end - recovery_start < 10:
            continue
        
        # Extract RR intervals during recovery phase
        recovery_rr = rr_df[
            (rr_df[time_col] >= recovery_start) & 
            (rr_df[time_col] < recovery_end)
        ][rr_col].values
        
        if len(recovery_rr) < 5:
            continue
        
        # Extract RR intervals during activity
        activity_rr = rr_df[
            (rr_df[time_col] >= t_start) & 
            (rr_df[time_col] < t_end)
        ][rr_col].values
        
        if len(activity_rr) < 5:
            continue
        
        # Compute HRV for each phase
        hrv_activity = compute_hrv_window(activity_rr)
        hrv_recovery = compute_hrv_window(recovery_rr)
        
        # Recovery rate: how fast HRV changes per second
        hrv_change = hrv_recovery - hrv_activity  # ms
        recovery_time_sec = (recovery_end - recovery_start)  # seconds
        hrv_recovery_rate = hrv_change / recovery_time_sec
        
        # Assign to windows in recovery phase
        recovery_mask = (
            (windows_labeled['t_center'] >= recovery_start) &
            (windows_labeled['t_center'] <= recovery_end)
        )
        
        n_assigned = recovery_mask.sum()
        if n_assigned > 0:
            windows_labeled.loc[recovery_mask, 'hrv_recovery_rate'] = hrv_recovery_rate
            print(f"  ✓ Activity {idx} (Borg {borg}): "
                  f"HRV_Recovery_Rate={hrv_recovery_rate:.3f} ms/s "
                  f"({hrv_activity:.1f}→{hrv_recovery:.1f} ms), "
                  f"assigned to {n_assigned} windows")
            successful_labels += 1
    
    n_labeled = windows_labeled['hrv_recovery_rate'].notna().sum()
    print(f"\n  ✓ Labeled {n_labeled} windows with HRV recovery rates from {successful_labels} activities")
    
    return windows_labeled
