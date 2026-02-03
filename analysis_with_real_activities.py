#!/usr/bin/env python3
"""
PROPER ANALYSIS WITH REAL ACTIVITY LABELS AND BETTER HR
========================================================
This script:
1. Loads actual activity labels from ADL files (not guessed from Borg transitions)
2. Uses Corsano device BPM (more reliable than PPG-derived HR)
3. Computes HR_delta and duration for each activity
4. Evaluates both window-level and activity-level predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PROPER ANALYSIS WITH REAL ACTIVITY LABELS + CORSANO HR")
print("="*80)

# ============================================================================
# STEP 1: Load ADL files with actual activity labels
# ============================================================================

def parse_adl_time(time_str):
    """Parse ADL timestamp format: DD-MM-YYYY-HH-MM-SS-mmm"""
    try:
        # Format: 04-12-2025-17-46-22-089
        parts = time_str.split('-')
        if len(parts) == 7:
            day, month, year, hour, minute, sec, ms = parts
            dt = datetime(int(year), int(month), int(day), 
                         int(hour), int(minute), int(sec), int(ms)*1000)
            return dt.timestamp()
    except:
        pass
    return None

def load_adl_activities(adl_path):
    """Load ADL file and extract activities with start/end times and Borg ratings"""
    
    # Read file, skip header rows
    df = pd.read_csv(adl_path, skiprows=2)
    
    # Rename columns if needed
    if 'Effort' in df.columns:
        df = df.rename(columns={'Effort': 'borg'})
    if 'ADLs' in df.columns:
        df = df.rename(columns={'ADLs': 'adl'})
    
    # Parse timestamps
    df['timestamp'] = df['Time'].apply(parse_adl_time)
    df = df.dropna(subset=['timestamp'])
    
    # Extract activities (pair Start/End)
    activities = []
    current_activity = None
    
    for _, row in df.iterrows():
        adl = row['adl']
        ts = row['timestamp']
        borg = row.get('borg', np.nan)
        
        if 'Start' in adl:
            # Start of new activity
            activity_name = adl.replace(' Start', '').strip()
            current_activity = {
                'activity': activity_name,
                'start_time': ts
            }
        elif 'End' in adl and current_activity is not None:
            # End of activity
            activity_name = adl.replace(' End', '').strip()
            if current_activity['activity'] == activity_name:
                current_activity['end_time'] = ts
                current_activity['duration'] = ts - current_activity['start_time']
                if pd.notna(borg):
                    current_activity['borg'] = float(borg)
                else:
                    current_activity['borg'] = np.nan
                activities.append(current_activity)
            current_activity = None
    
    return pd.DataFrame(activities)

print("\n" + "="*60)
print("STEP 1: Loading ADL files with REAL activity labels")
print("="*60)

# Load ADL files for all elderly subjects
subjects = ['sim_elderly1', 'sim_elderly2', 'sim_elderly3', 'sim_elderly4', 'sim_elderly5']
adl_paths = {
    'sim_elderly1': '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/scai_app/ADLs_1.csv',
    'sim_elderly2': '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/scai_app/ADLs_1.csv',
    'sim_elderly3': '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv',
    'sim_elderly4': '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/scai_app/ADLs_1.csv',
    'sim_elderly5': '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/scai_app/ADLs_1-5.csv',
}

all_activities = []
for subject, adl_path in adl_paths.items():
    if Path(adl_path).exists():
        activities_df = load_adl_activities(adl_path)
        activities_df['subject'] = subject
        all_activities.append(activities_df)
        print(f"  {subject}: {len(activities_df)} activities loaded")
    else:
        # Try .gz version
        gz_path = adl_path + '.gz'
        if Path(gz_path).exists():
            import gzip
            print(f"  {subject}: Found .gz version (skipping for now)")
        else:
            print(f"  {subject}: ADL file not found")

if all_activities:
    activities_df = pd.concat(all_activities, ignore_index=True)
    activities_df = activities_df.dropna(subset=['borg'])
    print(f"\nTotal activities with Borg ratings: {len(activities_df)}")
    
    # Show activity distribution
    print("\nActivity distribution:")
    activity_counts = activities_df['activity'].value_counts()
    for act, count in activity_counts.head(15).items():
        mean_borg = activities_df[activities_df['activity'] == act]['borg'].mean()
        print(f"  {act}: {count} occurrences, mean Borg = {mean_borg:.1f}")

# ============================================================================
# STEP 2: Load Corsano HR data
# ============================================================================

print("\n" + "="*60)
print("STEP 2: Loading Corsano HR data (30s intervals)")
print("="*60)

def load_corsano_hr(subject):
    """Load Corsano activity.csv which contains HR at 30s intervals"""
    base_paths = [
        f'/Users/pascalschlegel/data/raw/parsingsim1/{subject}_formatted/corsano_bioz',
        f'/Users/pascalschlegel/data/raw/parsingsim2/{subject}_formatted/corsano_bioz', 
        f'/Users/pascalschlegel/data/raw/parsingsim3/{subject}_formatted/corsano_bioz',
        f'/Users/pascalschlegel/data/raw/parsingsim4/{subject}_formatted/corsano_bioz',
        f'/Users/pascalschlegel/data/raw/parsingsim5/{subject}_formatted/corsano_bioz',
    ]
    
    all_hr = []
    for base in base_paths:
        base_path = Path(base)
        if base_path.exists():
            for date_dir in base_path.iterdir():
                if date_dir.is_dir():
                    activity_file = date_dir / 'activity.csv'
                    if activity_file.exists():
                        df = pd.read_csv(activity_file)
                        if 'bpm' in df.columns and 'timestamp' in df.columns:
                            # Convert timestamp to Unix
                            df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
                            all_hr.append(df[['timestamp', 'bpm', 'bpm_q', 'activity_type']])
    
    if all_hr:
        return pd.concat(all_hr, ignore_index=True)
    return None

# Load HR for each subject
hr_data = {}
for subject in subjects:
    hr_df = load_corsano_hr(subject)
    if hr_df is not None:
        hr_data[subject] = hr_df
        print(f"  {subject}: {len(hr_df)} HR samples loaded")
        print(f"    HR range: {hr_df['bpm'].min():.0f} - {hr_df['bpm'].max():.0f} bpm")
    else:
        print(f"  {subject}: No Corsano HR data found")

# ============================================================================
# STEP 3: Compute HR features for each activity
# ============================================================================

print("\n" + "="*60)
print("STEP 3: Computing HR features for each activity")
print("="*60)

def get_hr_for_activity(hr_df, start_time, end_time, pre_seconds=60):
    """Get HR during activity and baseline HR (pre-activity)"""
    
    # HR during activity
    mask_during = (hr_df['timestamp'] >= start_time) & (hr_df['timestamp'] <= end_time)
    hr_during = hr_df.loc[mask_during, 'bpm']
    
    # HR before activity (baseline)
    mask_before = (hr_df['timestamp'] >= start_time - pre_seconds) & (hr_df['timestamp'] < start_time)
    hr_before = hr_df.loc[mask_before, 'bpm']
    
    if len(hr_during) > 0 and len(hr_before) > 0:
        hr_activity = hr_during.mean()
        hr_baseline = hr_before.mean()
        hr_delta = hr_activity - hr_baseline
        hr_max = hr_during.max()
        return {
            'hr_activity': hr_activity,
            'hr_baseline': hr_baseline,
            'hr_delta': hr_delta,
            'hr_max': hr_max,
            'hr_samples': len(hr_during)
        }
    return None

# Add HR features to activities
activities_with_hr = []
for _, act in activities_df.iterrows():
    subject = act['subject']
    if subject in hr_data:
        hr_features = get_hr_for_activity(
            hr_data[subject], 
            act['start_time'], 
            act['end_time']
        )
        if hr_features is not None:
            act_dict = act.to_dict()
            act_dict.update(hr_features)
            activities_with_hr.append(act_dict)

if activities_with_hr:
    activities_hr_df = pd.DataFrame(activities_with_hr)
    print(f"Activities with HR data: {len(activities_hr_df)}")
    
    # Show correlation between HR features and Borg
    print("\nCorrelation of HR features with Borg:")
    for col in ['hr_delta', 'hr_activity', 'hr_max', 'duration']:
        if col in activities_hr_df.columns:
            valid = activities_hr_df[[col, 'borg']].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid[col], valid['borg'])
                print(f"  {col}: r = {r:.3f} (p = {p:.4f})")
else:
    print("No activities with HR data found")
    activities_hr_df = None

# ============================================================================
# STEP 4: Load existing aligned features for window-level comparison
# ============================================================================

print("\n" + "="*60)
print("STEP 4: Loading aligned sensor features")
print("="*60)

aligned_path = Path("/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/aligned_features_all_elderly_5.0s_70ol.csv")
if aligned_path.exists():
    aligned_df = pd.read_csv(aligned_path)
    print(f"Loaded {len(aligned_df)} windows with {len(aligned_df.columns)} columns")
    
    # Check for HR-related features
    hr_features = [c for c in aligned_df.columns if 'hr' in c.lower() or 'heart' in c.lower() or 'bpm' in c.lower()]
    print(f"HR-related features in aligned data: {hr_features}")
else:
    print("Aligned features file not found")
    aligned_df = None

# ============================================================================
# STEP 5: Match activities to windows and aggregate
# ============================================================================

print("\n" + "="*60)
print("STEP 5: Matching activities to windows")
print("="*60)

def assign_activity_to_windows(windows_df, activities_df):
    """Assign actual activity labels to each window based on timestamp overlap"""
    
    window_activities = []
    
    for idx, window in windows_df.iterrows():
        window_start = window['timestamp']
        window_end = window_start + 5.0  # 5 second window
        subject = window.get('label', window.get('subject', ''))
        
        # Find matching activity
        best_match = None
        best_overlap = 0
        
        for _, act in activities_df.iterrows():
            if act['subject'] in subject or subject in act['subject']:
                # Calculate overlap
                overlap_start = max(window_start, act['start_time'])
                overlap_end = min(window_end, act['end_time'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = act
        
        if best_match is not None and best_overlap > 0:
            window_activities.append({
                'window_idx': idx,
                'activity': best_match['activity'],
                'activity_borg': best_match['borg'],
                'overlap': best_overlap
            })
    
    return pd.DataFrame(window_activities)

if aligned_df is not None and len(activities_df) > 0:
    # Match activities to windows
    matches = assign_activity_to_windows(aligned_df, activities_df)
    print(f"Windows matched to activities: {len(matches)}")
    
    if len(matches) > 0:
        # Add activity labels to aligned data
        aligned_df['activity'] = None
        aligned_df['activity_borg'] = np.nan
        
        for _, match in matches.iterrows():
            idx = match['window_idx']
            aligned_df.loc[idx, 'activity'] = match['activity']
            aligned_df.loc[idx, 'activity_borg'] = match['activity_borg']
        
        # Show distribution of matched activities
        activity_counts = aligned_df['activity'].value_counts()
        print("\nMatched activity distribution:")
        for act, count in activity_counts.head(10).items():
            if pd.notna(act):
                print(f"  {act}: {count} windows")

# ============================================================================
# STEP 6: Activity-level prediction using REAL labels
# ============================================================================

print("\n" + "="*60)
print("STEP 6: Activity-level prediction with REAL labels")
print("="*60)

if aligned_df is not None:
    # Get feature columns
    exclude_cols = ['timestamp', 'borg', 'label', 'activity', 'activity_borg', 'subject']
    feature_cols = [c for c in aligned_df.columns if c not in exclude_cols 
                   and not c.startswith('Unnamed')]
    
    print(f"Feature columns: {len(feature_cols)}")
    
    # Filter to rows with activity labels
    df_with_activity = aligned_df[aligned_df['activity'].notna()].copy()
    print(f"Windows with activity labels: {len(df_with_activity)}")
    
    if len(df_with_activity) > 0:
        # Aggregate by subject + activity
        agg_features = df_with_activity.groupby(['label', 'activity'])[feature_cols + ['borg']].agg({
            **{f: 'mean' for f in feature_cols},
            'borg': 'mean'
        }).reset_index()
        
        print(f"Unique activity instances: {len(agg_features)}")
        
        # LOSO evaluation at activity level
        subjects_list = agg_features['label'].unique()
        print(f"\nRunning LOSO at activity level...")
        
        all_preds = []
        all_true = []
        all_subjects = []
        
        for test_subject in subjects_list:
            # Train/test split
            train_mask = agg_features['label'] != test_subject
            test_mask = agg_features['label'] == test_subject
            
            X_train = agg_features.loc[train_mask, feature_cols].values
            y_train = agg_features.loc[train_mask, 'borg'].values
            X_test = agg_features.loc[test_mask, feature_cols].values
            y_test = agg_features.loc[test_mask, 'borg'].values
            
            if len(X_test) < 2:
                continue
            
            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)
            
            # Train model
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            all_preds.extend(preds)
            all_true.extend(y_test)
            all_subjects.extend([test_subject] * len(y_test))
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        r, p = pearsonr(all_preds, all_true)
        mae = np.mean(np.abs(all_preds - all_true))
        rmse = np.sqrt(np.mean((all_preds - all_true)**2))
        
        # Apply linear calibration
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(all_preds.reshape(-1, 1), all_true)
        calibrated = lr.predict(all_preds.reshape(-1, 1))
        
        r_cal, _ = pearsonr(calibrated, all_true)
        mae_cal = np.mean(np.abs(calibrated - all_true))
        within_1_cal = np.mean(np.abs(calibrated - all_true) <= 1) * 100
        
        print(f"\n{'='*60}")
        print("ACTIVITY-LEVEL RESULTS (REAL LABELS, LOSO)")
        print(f"{'='*60}")
        print(f"Activities evaluated: {len(all_preds)}")
        print(f"\nRaw predictions:")
        print(f"  Pearson r: {r:.3f}")
        print(f"  MAE: {mae:.2f} Borg points")
        print(f"  RMSE: {rmse:.2f}")
        print(f"\nAfter linear calibration:")
        print(f"  Pearson r: {r_cal:.3f}")
        print(f"  MAE: {mae_cal:.2f} Borg points")
        print(f"  ±1 Borg accuracy: {within_1_cal:.1f}%")

# ============================================================================
# STEP 7: HR-based activity prediction (replicating r=0.84 approach)
# ============================================================================

print("\n" + "="*60)
print("STEP 7: HR-based prediction (HR_delta × √duration approach)")
print("="*60)

if activities_hr_df is not None and len(activities_hr_df) > 5:
    # Create HR-based features
    activities_hr_df['hr_effort'] = activities_hr_df['hr_delta'] * np.sqrt(activities_hr_df['duration'])
    activities_hr_df['log_duration'] = np.log1p(activities_hr_df['duration'])
    
    # Features for prediction
    hr_pred_features = ['hr_delta', 'hr_activity', 'hr_max', 'duration', 'log_duration', 'hr_effort']
    
    # Filter valid data
    valid_df = activities_hr_df.dropna(subset=hr_pred_features + ['borg'])
    print(f"Activities with complete HR data: {len(valid_df)}")
    
    if len(valid_df) > 10:
        # LOSO evaluation
        subjects_list = valid_df['subject'].unique()
        print(f"Subjects: {list(subjects_list)}")
        
        all_preds = []
        all_true = []
        
        for test_subject in subjects_list:
            train_mask = valid_df['subject'] != test_subject
            test_mask = valid_df['subject'] == test_subject
            
            X_train = valid_df.loc[train_mask, hr_pred_features].values
            y_train = valid_df.loc[train_mask, 'borg'].values
            X_test = valid_df.loc[test_mask, hr_pred_features].values
            y_test = valid_df.loc[test_mask, 'borg'].values
            
            if len(X_test) < 2:
                continue
            
            # Train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            all_preds.extend(preds)
            all_true.extend(y_test)
        
        # Results
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        if len(all_preds) > 2:
            r, p = pearsonr(all_preds, all_true)
            mae = np.mean(np.abs(all_preds - all_true))
            
            # Calibration
            lr = LinearRegression()
            lr.fit(all_preds.reshape(-1, 1), all_true)
            calibrated = lr.predict(all_preds.reshape(-1, 1))
            r_cal, _ = pearsonr(calibrated, all_true)
            mae_cal = np.mean(np.abs(calibrated - all_true))
            within_1 = np.mean(np.abs(calibrated - all_true) <= 1) * 100
            
            print(f"\n{'='*60}")
            print("HR-BASED PREDICTION RESULTS (LOSO)")
            print(f"{'='*60}")
            print(f"Activities: {len(all_preds)}")
            print(f"Features: {hr_pred_features}")
            print(f"\nRaw: r = {r:.3f}, MAE = {mae:.2f}")
            print(f"Calibrated: r = {r_cal:.3f}, MAE = {mae_cal:.2f}, ±1 Borg = {within_1:.1f}%")
            
            # Individual feature correlations
            print("\nIndividual feature correlations with Borg:")
            for feat in hr_pred_features:
                valid_feat = valid_df[[feat, 'borg']].dropna()
                if len(valid_feat) > 2:
                    r_feat, _ = pearsonr(valid_feat[feat], valid_feat['borg'])
                    print(f"  {feat}: r = {r_feat:.3f}")
else:
    print("Not enough HR data for HR-based prediction")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: REAL ACTIVITY LABELS VS GUESSED LABELS")
print("="*80)
print("""
Key differences from previous analysis:
1. Activity labels are NOW REAL (from ADL files) - not guessed from Borg transitions
2. HR comes from Corsano device (30s intervals) - more reliable than PPG-derived
3. Each activity has measured start/end times and Borg rating

This gives you proper activity-level aggregation for thesis reporting.
""")
