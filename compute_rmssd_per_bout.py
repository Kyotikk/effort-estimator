#!/usr/bin/env python3
"""
Compute RMSSD per activity bout for all patients.
Each ADL activity (Start -> End) gets one RMSSD value computed from RR intervals during that bout.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


def remove_artifacts_malik(rr_intervals, threshold=0.10):
    """Remove ectopic beats using Malik criterion (20% deviation from previous)."""
    if len(rr_intervals) < 2:
        return rr_intervals
    
    clean_rr = [rr_intervals[0]]
    for i in range(1, len(rr_intervals)):
        prev_rr = clean_rr[-1]
        curr_rr = rr_intervals[i]
        # Check if within threshold
        if abs(curr_rr - prev_rr) / prev_rr <= threshold:
            clean_rr.append(curr_rr)
    
    return np.array(clean_rr)


def compute_rmssd(rr_intervals):
    """Compute RMSSD from RR intervals (in ms)."""
    if len(rr_intervals) < 2:
        return np.nan
    
    # Successive differences
    diff = np.diff(rr_intervals)
    
    # Root mean square of successive differences
    rmssd = np.sqrt(np.mean(diff ** 2))
    return rmssd


def compute_hrv_metrics(rr_intervals):
    """Compute multiple HRV metrics from RR intervals (in ms)."""
    if len(rr_intervals) < 2:
        return {
            'rmssd': np.nan,
            'ln_rmssd': np.nan,
            'sdnn': np.nan,
            'mean_rr': np.nan,
            'mean_hr': np.nan,
            'pnn50': np.nan,
            'n_beats': 0
        }
    
    rmssd = compute_rmssd(rr_intervals)
    ln_rmssd = np.log(rmssd) if rmssd > 0 else np.nan
    sdnn = np.std(rr_intervals, ddof=1)
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60000 / mean_rr if mean_rr > 0 else np.nan
    
    # pNN50
    diff = np.abs(np.diff(rr_intervals))
    pnn50 = 100 * np.sum(diff > 50) / len(diff) if len(diff) > 0 else np.nan
    
    return {
        'rmssd': rmssd,
        'ln_rmssd': ln_rmssd,
        'sdnn': sdnn,
        'mean_rr': mean_rr,
        'mean_hr': mean_hr,
        'pnn50': pnn50,
        'n_beats': len(rr_intervals)
    }


def parse_adl_to_bouts(adl_df):
    """Parse ADL dataframe into activity bouts (start, end, activity_name)."""
    activities = []
    i = 0
    while i < len(adl_df) - 1:
        row = adl_df.iloc[i]
        if 'Start' in str(row['ADLs']):
            activity_name = row['ADLs'].replace(' Start', '')
            start_time = row['time']
            # Find matching end
            next_row = adl_df.iloc[i + 1]
            if 'End' in str(next_row['ADLs']) and activity_name in next_row['ADLs']:
                end_time = next_row['time']
                activities.append({
                    'activity': activity_name,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return pd.DataFrame(activities)


def get_rr_for_bout(rr_df, start_time, end_time):
    """Get RR intervals within a time window."""
    # Handle different column names
    time_col = 't_sec' if 't_sec' in rr_df.columns else 'time'
    mask = (rr_df[time_col] >= start_time) & (rr_df[time_col] <= end_time)
    return rr_df.loc[mask, 'rr'].values


def process_patient(dataset_path, patient_name, adl_offset_hours=-8):
    """Process a single patient: compute RMSSD per activity bout."""
    
    # Find RR intervals file
    rr_pattern = os.path.join(dataset_path, 'effort_estimation_output', '*', 'rr', 'rr_preprocessed.csv')
    rr_files = glob.glob(rr_pattern)
    
    if not rr_files:
        # Try vivalnk heart rate merged
        rr_pattern = os.path.join(dataset_path, 'effort_estimation_output', '_merged_inputs', 'vivalnk_vv330_heart_rate_merged.csv.gz')
        if os.path.exists(rr_pattern):
            rr_df = pd.read_csv(rr_pattern)
            # Convert HR to RR if needed
            if 'rr' not in rr_df.columns and 'heart_rate' in rr_df.columns:
                rr_df['rr'] = 60000 / rr_df['heart_rate']
        else:
            print(f"  No RR data found for {patient_name}")
            return None
    else:
        rr_df = pd.read_csv(rr_files[0])
    
    # Convert RR to ms if in seconds
    if rr_df['rr'].median() < 10:  # Likely in seconds
        rr_df['rr'] = rr_df['rr'] * 1000
    
    # Find ADL file
    adl_pattern = os.path.join(dataset_path, 'scai_app', 'ADLs*.csv.gz')
    adl_files = glob.glob(adl_pattern)
    
    if not adl_files:
        print(f"  No ADL file found for {patient_name}")
        return None
    
    adl_df = pd.read_csv(adl_files[0])
    
    # Apply time offset to ADL
    adl_df['time'] = adl_df['time'] + (adl_offset_hours * 3600)
    
    # Parse into activity bouts
    bouts = parse_adl_to_bouts(adl_df)
    
    if len(bouts) == 0:
        print(f"  No activity bouts found for {patient_name}")
        return None
    
    # Compute HRV metrics for each bout
    results = []
    for idx, bout in bouts.iterrows():
        # Only consider bouts >= 60s
        if bout['duration'] < 60:
            continue
        rr_in_bout = get_rr_for_bout(rr_df, bout['start'], bout['end'])
        # Apply stricter artifact removal
        rr_clean = remove_artifacts_malik(rr_in_bout, threshold=0.10)
        # Compute metrics
        metrics = compute_hrv_metrics(rr_clean)
        results.append({
            'patient': patient_name,
            'activity': bout['activity'],
            'start': bout['start'],
            'end': bout['end'],
            'duration': bout['duration'],
            **metrics
        })
    return pd.DataFrame(results)


def main():
    """Process all patients and compute RMSSD per activity bout."""
    
    base_path = '/Users/pascalschlegel/data/interim'
    
    # Define all datasets
    datasets = []
    for sim in ['parsingsim3', 'parsingsim4', 'parsingsim5']:
        for condition in ['elderly', 'healthy', 'severe']:
            sim_num = sim[-1]
            patient_name = f"sim_{condition}{sim_num}"
            dataset_path = os.path.join(base_path, sim, patient_name)
            if os.path.exists(dataset_path):
                datasets.append((dataset_path, patient_name))
    
    print(f"Found {len(datasets)} datasets")
    
    all_results = []
    
    for dataset_path, patient_name in datasets:
        print(f"\nProcessing {patient_name}...")
        result = process_patient(dataset_path, patient_name)
        
        if result is not None and len(result) > 0:
            all_results.append(result)
            valid_bouts = result['n_beats'].sum() > 0
            n_valid = (result['n_beats'] >= 5).sum()
            print(f"  {len(result)} bouts, {n_valid} with ≥5 beats")
    
    if not all_results:
        print("No results!")
        return
    
    # Combine all results
    df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_path = '/Users/pascalschlegel/effort-estimator/rmssd_per_bout.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} activity bouts to {output_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY: RMSSD per Activity Bout")
    print("="*60)
    
    # Filter to bouts with at least 10 beats
    df_valid = df[df['n_beats'] >= 10].copy()
    print(f"\nValid bouts (≥10 beats, ≥60s): {len(df_valid)} / {len(df)}")

    # Normalize ln(RMSSD) within each patient (z-score)
    df_valid['ln_rmssd_z'] = df_valid.groupby('patient')['ln_rmssd'].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0)
    
    # Group by activity type
    print("\n--- By Activity Type (raw lnRMSSD) ---")
    activity_stats = df_valid.groupby('activity').agg({
        'ln_rmssd': ['mean', 'std', 'count'],
        'mean_hr': 'mean',
        'duration': 'mean'
    }).round(2)
    activity_stats.columns = ['ln_rmssd_mean', 'ln_rmssd_std', 'n_bouts', 'mean_hr', 'duration']
    activity_stats = activity_stats.sort_values('ln_rmssd_mean')
    print(activity_stats.to_string())

    print("\n--- By Activity Type (z-scored lnRMSSD) ---")
    activity_stats_z = df_valid.groupby('activity').agg({
        'ln_rmssd_z': ['mean', 'std', 'count'],
        'mean_hr': 'mean',
        'duration': 'mean'
    }).round(2)
    activity_stats_z.columns = ['ln_rmssd_z_mean', 'ln_rmssd_z_std', 'n_bouts', 'mean_hr', 'duration']
    activity_stats_z = activity_stats_z.sort_values('ln_rmssd_z_mean')
    print(activity_stats_z.to_string())
    
    # Group by patient
    print("\n--- By Patient (raw lnRMSSD) ---")
    patient_stats = df_valid.groupby('patient').agg({
        'ln_rmssd': ['mean', 'std', 'count'],
        'mean_hr': 'mean'
    }).round(2)
    patient_stats.columns = ['ln_rmssd_mean', 'ln_rmssd_std', 'n_bouts', 'mean_hr']
    print(patient_stats.to_string())

    print("\n--- By Patient (z-scored lnRMSSD) ---")
    patient_stats_z = df_valid.groupby('patient').agg({
        'ln_rmssd_z': ['mean', 'std', 'count'],
        'mean_hr': 'mean'
    }).round(2)
    patient_stats_z.columns = ['ln_rmssd_z_mean', 'ln_rmssd_z_std', 'n_bouts', 'mean_hr']
    print(patient_stats_z.to_string())
    
    # Categorize activities by intensity
    print("\n--- Activity Intensity Categories (raw and z-scored) ---")
    # Define intensity categories
    high_intensity = ['Sit to Stand', 'Stand to Sit', 'Lying to Sit', 'Transfer to Bed', 
                      'Transfer from Bed', 'Transfer to Toilet', 'Lower/Raise Pants']
    moderate = ['Turn Over (right)', 'Turn Over (left)', 'Sit to lying', 'Fold Clothes',
                'Arm in Sleeves', '(Un)button Shirt']
    low_intensity = ['Stand', 'Resting']
    def categorize(activity):
        if activity in high_intensity:
            return 'High (transfers)'
        elif activity in moderate:
            return 'Moderate (movements)'
        elif activity in low_intensity:
            return 'Low (static)'
        else:
            return 'Other'
    df_valid['intensity'] = df_valid['activity'].apply(categorize)
    # Raw
    intensity_stats = df_valid.groupby('intensity').agg({
        'ln_rmssd': ['mean', 'std', 'count'],
        'mean_hr': 'mean'
    }).round(3)
    intensity_stats.columns = ['ln_rmssd_mean', 'ln_rmssd_std', 'n_bouts', 'mean_hr']
    print("\nRaw lnRMSSD by intensity:")
    print(intensity_stats.to_string())
    # Z-scored
    intensity_stats_z = df_valid.groupby('intensity').agg({
        'ln_rmssd_z': ['mean', 'std', 'count'],
        'mean_hr': 'mean'
    }).round(3)
    intensity_stats_z.columns = ['ln_rmssd_z_mean', 'ln_rmssd_z_std', 'n_bouts', 'mean_hr']
    print("\nZ-scored lnRMSSD by intensity:")
    print(intensity_stats_z.to_string())
    # Expected: High intensity should have LOWER ln_rmssd (higher effort)
    print("\n--- Hypothesis Check (z-scored) ---")
    print("Expected: Higher intensity → Lower ln(RMSSD) (less vagal tone)")
    if 'High (transfers)' in intensity_stats_z.index and 'Low (static)' in intensity_stats_z.index:
        high_lnrmssd = intensity_stats_z.loc['High (transfers)', 'ln_rmssd_z_mean']
        low_lnrmssd = intensity_stats_z.loc['Low (static)', 'ln_rmssd_z_mean']
        diff = high_lnrmssd - low_lnrmssd
        print(f"High intensity z-ln(RMSSD): {high_lnrmssd:.3f}")
        print(f"Low intensity z-ln(RMSSD): {low_lnrmssd:.3f}")
        print(f"Difference: {diff:.3f}")
        if diff < 0:
            print("✓ Expected pattern: High intensity has LOWER ln(RMSSD)")
        else:
            print("⚠ Unexpected pattern: High intensity has HIGHER ln(RMSSD)")
    # Stratify by health status
    print("\n--- Stratified by Health Status (z-scored lnRMSSD) ---")
    def get_status(patient):
        if 'healthy' in patient:
            return 'Healthy'
        elif 'elderly' in patient:
            return 'Elderly'
        elif 'severe' in patient:
            return 'Severe'
        else:
            return 'Unknown'
    df_valid['status'] = df_valid['patient'].apply(get_status)
    for status in ['Healthy', 'Elderly', 'Severe']:
        sub = df_valid[df_valid['status'] == status]
        if len(sub) == 0:
            continue
        print(f"\nStatus: {status} (n={len(sub)})")
        stat = sub.groupby('intensity').agg({'ln_rmssd_z': ['mean', 'std', 'count'], 'mean_hr': 'mean'}).round(3)
        stat.columns = ['ln_rmssd_z_mean', 'ln_rmssd_z_std', 'n_bouts', 'mean_hr']
        print(stat.to_string())


if __name__ == '__main__':
    main()
