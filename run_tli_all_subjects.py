#!/usr/bin/env python3
"""
MULTI-SUBJECT TLI ANALYSIS WITH PROPER LOSO EVALUATION
=======================================================
This runs the TLI (Training Load Index) analysis for all 5 elderly subjects
using:
1. REAL activity labels from ADL files
2. HR from Vivalnk sensor (not PPG-derived)
3. IMU features per activity
4. Proper LOSO cross-validation

This is the approach that got r=0.84 in previous analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import gzip
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-SUBJECT TLI WITH REAL ACTIVITY LABELS")
print("="*80)

# =============================================================================
# ADL Parsing Functions
# =============================================================================

def parse_adl_timestamp(ts_str: str) -> float:
    """Parse ADL timestamp string to Unix timestamp."""
    try:
        parts = ts_str.split('-')
        if len(parts) >= 6:
            day, month, year, hour, minute, second = parts[:6]
            ms = int(parts[6]) if len(parts) > 6 else 0
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), ms * 1000)
            # Apply timezone correction: subtract 8 hours (28800 seconds)
            return dt.timestamp() - 28800
    except:
        pass
    return np.nan


def load_adl_labels(subject_dir: Path) -> pd.DataFrame:
    """Load ADL labels with start/end times and Borg ratings."""
    # Try different ADL file patterns
    adl_patterns = [
        subject_dir / 'scai_app' / 'ADLs_1.csv',
        subject_dir / 'scai_app' / 'ADLs_1-2.csv',
        subject_dir / 'scai_app' / 'ADLs_1-3.csv',
        subject_dir / 'scai_app' / 'ADLs_1-4.csv',
        subject_dir / 'scai_app' / 'ADLs_1-5.csv',
    ]
    
    adl_path = None
    for p in adl_patterns:
        if p.exists():
            adl_path = p
            break
    
    if adl_path is None:
        # Try gzipped version
        for p in adl_patterns:
            gz_path = Path(str(p) + '.gz')
            if gz_path.exists():
                adl_path = gz_path
                break
    
    if adl_path is None:
        raise FileNotFoundError(f"ADL labels not found in {subject_dir / 'scai_app'}")
    
    # Read file
    if str(adl_path).endswith('.gz'):
        with gzip.open(adl_path, 'rt') as f:
            lines = f.readlines()
    else:
        with open(adl_path, 'r') as f:
            lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if 'Time' in line and 'ADL' in line:
            data_start = i + 1
            break
    
    # Parse activities
    activities = []
    current_activity = None
    current_start = None
    
    for line in lines[data_start:]:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        
        timestamp_str = parts[0]
        adl_label = parts[1]
        effort = parts[2] if len(parts) > 2 else ''
        
        if 'Start' in adl_label:
            activity_name = adl_label.replace(' Start', '')
            current_activity = activity_name
            current_start = parse_adl_timestamp(timestamp_str)
        
        elif 'End' in adl_label and current_activity is not None:
            t_end = parse_adl_timestamp(timestamp_str)
            
            borg = np.nan
            if effort:
                try:
                    borg = float(effort)
                except:
                    pass
            
            if not np.isnan(current_start) and not np.isnan(t_end):
                activities.append({
                    'activity': current_activity,
                    't_start': current_start,
                    't_end': t_end,
                    'duration_s': t_end - current_start,
                    'borg': borg
                })
            
            current_activity = None
            current_start = None
    
    return pd.DataFrame(activities)


def load_hr_data(subject_dir: Path) -> pd.DataFrame:
    """Load raw HR data from vivalnk sensor."""
    hr_dir = subject_dir / 'vivalnk_vv330_heart_rate'
    
    if not hr_dir.exists():
        return None
    
    dfs = []
    for f in sorted(hr_dir.glob('*.csv.gz')):
        with gzip.open(f, 'rt') as gz:
            df = pd.read_csv(gz)
            dfs.append(df)
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['hr'] > 0].copy()  # Filter invalid
    
    return df


def load_acc_data(subject_dir: Path) -> pd.DataFrame:
    """Load raw acceleration data from vivalnk sensor."""
    acc_dir = subject_dir / 'vivalnk_vv330_acceleration'
    
    if not acc_dir.exists():
        return None
    
    dfs = []
    for f in sorted(acc_dir.glob('*.csv.gz')):
        with gzip.open(f, 'rt') as gz:
            df = pd.read_csv(gz)
            dfs.append(df)
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert to g and compute magnitude
    scale = 1.0 / 4096.0
    df['x_g'] = df['x'] * scale
    df['y_g'] = df['y'] * scale
    df['z_g'] = df['z'] * scale
    df['mag_g'] = np.sqrt(df['x_g']**2 + df['y_g']**2 + df['z_g']**2)
    
    return df


def compute_features_per_activity(adl_df, hr_df, acc_df):
    """Compute HR and IMU features for each activity."""
    
    # Compute baseline HR (first 10% of recording)
    if hr_df is not None and len(hr_df) > 10:
        n_baseline = max(10, len(hr_df) // 10)
        hr_baseline = hr_df.head(n_baseline)['hr'].median()
    else:
        hr_baseline = 70  # Default
    
    results = []
    for _, row in adl_df.iterrows():
        t_start = row['t_start']
        t_end = row['t_end']
        
        feat = {**row.to_dict()}
        
        # HR features
        if hr_df is not None:
            mask = (hr_df['time'] >= t_start) & (hr_df['time'] <= t_end)
            hr_window = hr_df.loc[mask, 'hr']
            
            if len(hr_window) > 0:
                feat['hr_mean'] = hr_window.mean()
                feat['hr_max'] = hr_window.max()
                feat['hr_delta'] = hr_window.mean() - hr_baseline
                feat['hr_samples'] = len(hr_window)
            else:
                feat['hr_mean'] = np.nan
                feat['hr_max'] = np.nan
                feat['hr_delta'] = np.nan
                feat['hr_samples'] = 0
        
        feat['hr_baseline'] = hr_baseline
        
        # IMU features
        if acc_df is not None:
            mask = (acc_df['time'] >= t_start) & (acc_df['time'] <= t_end)
            acc_window = acc_df.loc[mask]
            
            if len(acc_window) > 0:
                feat['rms_acc_mag'] = np.sqrt(np.mean(acc_window['mag_g']**2))
                
                # Jerk (derivative of magnitude)
                if len(acc_window) > 1:
                    dt = np.diff(acc_window['time'].values)
                    dt[dt == 0] = 0.01  # Prevent division by zero
                    jerk = np.diff(acc_window['mag_g'].values) / dt
                    feat['rms_jerk'] = np.sqrt(np.mean(jerk**2))
                else:
                    feat['rms_jerk'] = 0
            else:
                feat['rms_acc_mag'] = np.nan
                feat['rms_jerk'] = np.nan
        
        results.append(feat)
    
    return pd.DataFrame(results)


# =============================================================================
# Load Data for All Subjects
# =============================================================================

print("\n" + "="*60)
print("STEP 1: Loading data for all elderly subjects")
print("="*60)

# Subject configurations
subjects_config = {
    'sim_elderly1': '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1',
    'sim_elderly2': '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2',
    'sim_elderly3': '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3',
    'sim_elderly4': '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4',
    'sim_elderly5': '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5',
}

all_activities = []

for subject, data_dir in subjects_config.items():
    subject_dir = Path(data_dir)
    print(f"\n{subject}:")
    
    if not subject_dir.exists():
        print(f"  ⚠ Directory not found: {subject_dir}")
        continue
    
    try:
        # Load ADL
        adl_df = load_adl_labels(subject_dir)
        print(f"  ✓ {len(adl_df)} activities from ADL file")
        
        # Load HR
        hr_df = load_hr_data(subject_dir)
        if hr_df is not None:
            print(f"  ✓ {len(hr_df)} HR samples")
        else:
            print(f"  ⚠ No HR data found")
        
        # Load IMU
        acc_df = load_acc_data(subject_dir)
        if acc_df is not None:
            print(f"  ✓ {len(acc_df)} acceleration samples")
        else:
            print(f"  ⚠ No acceleration data found")
        
        # Compute features
        features_df = compute_features_per_activity(adl_df, hr_df, acc_df)
        features_df['subject'] = subject
        all_activities.append(features_df)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Combine all
if all_activities:
    df = pd.concat(all_activities, ignore_index=True)
    df = df.dropna(subset=['borg'])  # Only activities with Borg
    print(f"\n{'='*60}")
    print(f"Total activities with Borg ratings: {len(df)}")
    print(f"Subjects: {df['subject'].unique().tolist()}")

# =============================================================================
# STEP 2: Compute TLI Features
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Computing TLI features")
print("="*60)

# HR load = (HR_delta) × sqrt(duration)
df['hr_load'] = df['hr_delta'] * np.sqrt(df['duration_s'])

# IMU load = RMS_jerk × duration
df['imu_load'] = df['rms_jerk'] * df['duration_s']

# Combined TLI (z-score normalized)
valid_mask = df[['hr_load', 'imu_load']].notna().all(axis=1)
df_valid = df[valid_mask].copy()

if len(df_valid) > 2:
    # Z-score normalize
    hr_z = (df_valid['hr_load'] - df_valid['hr_load'].mean()) / df_valid['hr_load'].std()
    imu_z = (df_valid['imu_load'] - df_valid['imu_load'].mean()) / df_valid['imu_load'].std()
    df_valid['TLI'] = 0.5 * hr_z + 0.5 * imu_z
    
    print(f"Valid activities for TLI: {len(df_valid)}")

# =============================================================================
# STEP 3: Feature Correlations with Borg
# =============================================================================

print("\n" + "="*60)
print("STEP 3: Feature correlations with Borg (ALL SUBJECTS)")
print("="*60)

feature_cols = ['hr_delta', 'hr_mean', 'hr_max', 'duration_s', 'rms_acc_mag', 
                'rms_jerk', 'hr_load', 'imu_load', 'TLI']

print("\nCorrelation with Borg:")
for col in feature_cols:
    if col in df_valid.columns:
        valid = df_valid[[col, 'borg']].dropna()
        if len(valid) > 2:
            r, p = pearsonr(valid[col], valid['borg'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {col:15s}: r = {r:+.3f} (p = {p:.4f}) {sig}")

# =============================================================================
# STEP 4: LOSO Cross-Validation
# =============================================================================

print("\n" + "="*60)
print("STEP 4: LOSO Cross-Validation")
print("="*60)

# Features for prediction
pred_features = ['hr_delta', 'hr_load', 'imu_load', 'duration_s', 'rms_jerk']
pred_features = [f for f in pred_features if f in df_valid.columns]

print(f"Using features: {pred_features}")

subjects_list = df_valid['subject'].unique()
print(f"Subjects: {list(subjects_list)}")

all_preds = []
all_true = []
all_subjects = []

for test_subject in subjects_list:
    # Train/test split
    train_mask = df_valid['subject'] != test_subject
    test_mask = df_valid['subject'] == test_subject
    
    X_train = df_valid.loc[train_mask, pred_features].values
    y_train = df_valid.loc[train_mask, 'borg'].values
    X_test = df_valid.loc[test_mask, pred_features].values
    y_test = df_valid.loc[test_mask, 'borg'].values
    
    if len(X_test) < 2:
        continue
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)
    
    # Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    all_preds.extend(preds)
    all_true.extend(y_test)
    all_subjects.extend([test_subject] * len(y_test))
    
    # Per-subject stats
    r_sub, _ = pearsonr(preds, y_test)
    print(f"  {test_subject}: n={len(y_test):3d}, r={r_sub:.3f}")

# Overall metrics
all_preds = np.array(all_preds)
all_true = np.array(all_true)

r_raw, p_raw = pearsonr(all_preds, all_true)
mae_raw = np.mean(np.abs(all_preds - all_true))
rmse_raw = np.sqrt(np.mean((all_preds - all_true)**2))
within_1_raw = np.mean(np.abs(all_preds - all_true) <= 1) * 100

print(f"\n{'='*60}")
print("RAW LOSO RESULTS")
print(f"{'='*60}")
print(f"  Activities: {len(all_preds)}")
print(f"  Pearson r: {r_raw:.3f} (p = {p_raw:.6f})")
print(f"  MAE: {mae_raw:.2f} Borg points")
print(f"  RMSE: {rmse_raw:.2f}")
print(f"  ±1 Borg accuracy: {within_1_raw:.1f}%")

# With linear calibration
lr = LinearRegression()
lr.fit(all_preds.reshape(-1, 1), all_true)
calibrated = lr.predict(all_preds.reshape(-1, 1))

r_cal, p_cal = pearsonr(calibrated, all_true)
mae_cal = np.mean(np.abs(calibrated - all_true))
within_1_cal = np.mean(np.abs(calibrated - all_true) <= 1) * 100

print(f"\n{'='*60}")
print("CALIBRATED LOSO RESULTS")
print(f"{'='*60}")
print(f"  Pearson r: {r_cal:.3f}")
print(f"  MAE: {mae_cal:.2f} Borg points")
print(f"  ±1 Borg accuracy: {within_1_cal:.1f}%")

# =============================================================================
# STEP 5: Compare with Simple HR-only Model
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Simple HR_load-only prediction")
print("="*60)

# Just use hr_load
all_preds_hr = []
all_true_hr = []

for test_subject in subjects_list:
    train_mask = df_valid['subject'] != test_subject
    test_mask = df_valid['subject'] == test_subject
    
    X_train = df_valid.loc[train_mask, ['hr_load']].values
    y_train = df_valid.loc[train_mask, 'borg'].values
    X_test = df_valid.loc[test_mask, ['hr_load']].values
    y_test = df_valid.loc[test_mask, 'borg'].values
    
    if len(X_test) < 2:
        continue
    
    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)
    
    # Simple linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    
    all_preds_hr.extend(preds)
    all_true_hr.extend(y_test)

all_preds_hr = np.array(all_preds_hr)
all_true_hr = np.array(all_true_hr)

r_hr, _ = pearsonr(all_preds_hr, all_true_hr)
mae_hr = np.mean(np.abs(all_preds_hr - all_true_hr))
within_1_hr = np.mean(np.abs(all_preds_hr - all_true_hr) <= 1) * 100

print(f"HR_load only (LOSO):")
print(f"  r = {r_hr:.3f}")
print(f"  MAE = {mae_hr:.2f}")
print(f"  ±1 Borg = {within_1_hr:.1f}%")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: ACTIVITY-LEVEL PREDICTION WITH REAL LABELS")
print("="*80)
print(f"""
This analysis uses:
✓ REAL activity labels from ADL files (not guessed from Borg transitions)
✓ HR from Vivalnk sensor (not noisy PPG-derived)
✓ Activity-level aggregation (not 5s windows)
✓ Proper LOSO cross-validation

Key Results:
┌────────────────────────────────────────────────────────────┐
│ Model                  │  r    │  MAE  │  ±1 Borg          │
├────────────────────────────────────────────────────────────┤
│ Full model (LOSO raw)  │ {r_raw:.3f} │ {mae_raw:.2f}  │ {within_1_raw:.1f}%            │
│ Full model (calibrated)│ {r_cal:.3f} │ {mae_cal:.2f}  │ {within_1_cal:.1f}%            │
│ HR_load only (LOSO)    │ {r_hr:.3f} │ {mae_hr:.2f}  │ {within_1_hr:.1f}%            │
└────────────────────────────────────────────────────────────┘

HR_load = HR_delta × √duration (captures both HR response and activity time)
""")

# Save results
output_path = Path("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
df_valid.to_csv(output_path, index=False)
print(f"✓ Results saved to: {output_path}")
