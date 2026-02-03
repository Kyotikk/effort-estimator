#!/usr/bin/env python3
"""
FINAL COMPARISON: Multimodal vs ECG HR
======================================
- Multimodal: ALL 5 subjects (PPG + EDA + IMU)
- ECG HR: 4 subjects with ECG data (elderly2-5)

Both use: LOSO + 20% per-subject calibration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FINAL COMPARISON: MULTIMODAL vs ECG HR")
print("="*80)

# =============================================================================
# LOAD ALL DATA
# =============================================================================

# Load multimodal data for all 5 subjects
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)
        print(f"Loaded elderly{i}: {len(df)} windows")

df_multimodal = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"\nTotal multimodal windows: {len(df_multimodal)}")
print(f"Subjects: {sorted(df_multimodal['subject'].unique())}")

# Load ECG HR data
tli_df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
tli_df = tli_df.dropna(subset=['hr_delta', 'borg'])
print(f"\nECG activities: {len(tli_df)}")
print(f"ECG subjects: {sorted(tli_df['subject'].unique())}")

# =============================================================================
# DEFINE FEATURE SETS
# =============================================================================

exclude_cols = ['t_center', 'borg', 'subject', 'Unnamed', 'activity_label']
all_features = [c for c in df_multimodal.columns if not any(x in c for x in exclude_cols)]

# Separate by modality
ppg_features = [c for c in all_features if 'ppg' in c.lower()]
eda_features = [c for c in all_features if 'eda' in c.lower()]
imu_features = [c for c in all_features if 'imu' in c.lower() or 'acc' in c.lower() or 'gyro' in c.lower()]

print(f"\nFeature counts:")
print(f"  PPG: {len(ppg_features)}")
print(f"  EDA: {len(eda_features)}")
print(f"  IMU: {len(imu_features)}")
print(f"  Total: {len(all_features)}")

# =============================================================================
# LOSO + CALIBRATION FUNCTION
# =============================================================================

def run_loso_calibrated(df, features, cal_frac=0.2, name=""):
    """
    LOSO with per-subject calibration.
    Returns detailed results per subject.
    """
    subjects = sorted(df['subject'].unique())
    
    all_preds = []
    all_true = []
    per_sub_results = {}
    
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub].copy()
        test_df = df[df['subject'] == test_sub].copy()
        
        # Drop NaN in features
        valid_features = [f for f in features if f in train_df.columns]
        train_df = train_df.dropna(subset=valid_features + ['borg'])
        test_df = test_df.dropna(subset=valid_features + ['borg'])
        
        if len(train_df) < 20 or len(test_df) < 10:
            print(f"  Skipping {test_sub}: train={len(train_df)}, test={len(test_df)}")
            continue
        
        # Split test into calibration and evaluation
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        
        idx = np.arange(n_test)
        np.random.shuffle(idx)
        cal_idx = idx[:n_cal]
        eval_idx = idx[n_cal:]
        
        cal_df = test_df.iloc[cal_idx]
        eval_df = test_df.iloc[eval_idx]
        
        if len(eval_df) < 5:
            print(f"  Skipping {test_sub}: eval set too small ({len(eval_df)})")
            continue
        
        # Prepare data
        X_train = train_df[valid_features].values
        y_train = train_df['borg'].values
        X_cal = cal_df[valid_features].values
        y_cal = cal_df['borg'].values
        X_eval = eval_df[valid_features].values
        y_eval = eval_df['borg'].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_cal = np.nan_to_num(X_cal, nan=0.0)
        X_eval = np.nan_to_num(X_eval, nan=0.0)
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_eval_s = scaler.transform(X_eval)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        
        # Calibrate with linear regression
        preds_cal = model.predict(X_cal_s)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Evaluate
        preds_raw = model.predict(X_eval_s)
        preds = calibrator.predict(preds_raw.reshape(-1, 1))
        
        # Store results
        all_preds.extend(preds)
        all_true.extend(y_eval)
        
        r_sub, _ = pearsonr(preds, y_eval)
        mae_sub = np.mean(np.abs(preds - y_eval))
        within1_sub = np.mean(np.abs(preds - y_eval) <= 1) * 100
        
        per_sub_results[test_sub] = {
            'r': r_sub,
            'mae': mae_sub,
            'within_1': within1_sub,
            'n_train': len(train_df),
            'n_cal': len(cal_df),
            'n_eval': len(eval_df)
        }
    
    if len(all_preds) < 10:
        return None
    
    # Overall metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    r_overall, _ = pearsonr(all_preds, all_true)
    mae_overall = np.mean(np.abs(all_preds - all_true))
    within1_overall = np.mean(np.abs(all_preds - all_true) <= 1) * 100
    
    return {
        'name': name,
        'r': r_overall,
        'mae': mae_overall,
        'within_1': within1_overall,
        'n_total': len(all_preds),
        'n_subjects': len(per_sub_results),
        'per_subject': per_sub_results
    }

# =============================================================================
# MODEL 1: MULTIMODAL (ALL 5 SUBJECTS)
# =============================================================================

print("\n" + "="*60)
print("MODEL 1: MULTIMODAL (PPG + EDA + IMU) - ALL 5 SUBJECTS")
print("="*60)

# Select top features by correlation
print("\nSelecting top features by correlation...")
correlations = []
for col in all_features:
    valid = df_multimodal[[col, 'borg']].dropna()
    if len(valid) > 50:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r), r))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features:")
for feat, abs_r, r in correlations[:20]:
    modality = 'PPG' if 'ppg' in feat.lower() else ('EDA' if 'eda' in feat.lower() else 'IMU')
    print(f"  {feat}: r = {r:.3f} ({modality})")

# Use top 30 features
top_30_features = [c[0] for c in correlations[:30]]

print(f"\nRunning LOSO + 20% calibration with {len(top_30_features)} features...")
result_multimodal = run_loso_calibrated(df_multimodal, top_30_features, cal_frac=0.2, name="Multimodal")

if result_multimodal:
    print(f"\n=== MULTIMODAL RESULTS ===")
    print(f"Overall: r = {result_multimodal['r']:.3f}, MAE = {result_multimodal['mae']:.2f}, ±1 Borg = {result_multimodal['within_1']:.1f}%")
    print(f"N = {result_multimodal['n_total']} windows from {result_multimodal['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_multimodal['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}% (train={res['n_train']}, cal={res['n_cal']}, eval={res['n_eval']})")

# =============================================================================
# MODEL 2: ECG HR ONLY (SUBJECTS WITH ECG DATA)
# =============================================================================

print("\n" + "="*60)
print("MODEL 2: ECG HR ONLY - SUBJECTS WITH ECG DATA")
print("="*60)

# Prepare ECG data (activity-level)
ecg_features = ['hr_delta', 'hr_load']
tli_df['subject_short'] = tli_df['subject'].str.replace('sim_', '')

print(f"\nECG data per subject:")
for sub in sorted(tli_df['subject_short'].unique()):
    n = len(tli_df[tli_df['subject_short'] == sub])
    print(f"  {sub}: {n} activities")

print(f"\nRunning LOSO + 20% calibration with {len(ecg_features)} features...")
result_ecg = run_loso_calibrated(tli_df, ecg_features, cal_frac=0.2, name="ECG HR")

if result_ecg:
    print(f"\n=== ECG HR RESULTS ===")
    print(f"Overall: r = {result_ecg['r']:.3f}, MAE = {result_ecg['mae']:.2f}, ±1 Borg = {result_ecg['within_1']:.1f}%")
    print(f"N = {result_ecg['n_total']} activities from {result_ecg['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_ecg['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}% (train={res['n_train']}, cal={res['n_cal']}, eval={res['n_eval']})")

# =============================================================================
# MODEL 3: MULTIMODAL WITH MORE FEATURES
# =============================================================================

print("\n" + "="*60)
print("MODEL 3: MULTIMODAL - ALL AVAILABLE FEATURES")
print("="*60)

# Use all features (not just top 30)
print(f"\nRunning LOSO + 20% calibration with ALL {len(all_features)} features...")
result_multimodal_all = run_loso_calibrated(df_multimodal, all_features, cal_frac=0.2, name="Multimodal All")

if result_multimodal_all:
    print(f"\n=== MULTIMODAL (ALL FEATURES) RESULTS ===")
    print(f"Overall: r = {result_multimodal_all['r']:.3f}, MAE = {result_multimodal_all['mae']:.2f}, ±1 Borg = {result_multimodal_all['within_1']:.1f}%")
    print(f"N = {result_multimodal_all['n_total']} windows from {result_multimodal_all['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_multimodal_all['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}%")

# =============================================================================
# MODEL 4: IMU ONLY
# =============================================================================

print("\n" + "="*60)
print("MODEL 4: IMU ONLY - ALL 5 SUBJECTS")
print("="*60)

print(f"\nRunning LOSO + 20% calibration with {len(imu_features)} IMU features...")
result_imu = run_loso_calibrated(df_multimodal, imu_features, cal_frac=0.2, name="IMU only")

if result_imu:
    print(f"\n=== IMU ONLY RESULTS ===")
    print(f"Overall: r = {result_imu['r']:.3f}, MAE = {result_imu['mae']:.2f}, ±1 Borg = {result_imu['within_1']:.1f}%")
    print(f"N = {result_imu['n_total']} windows from {result_imu['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_imu['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}%")

# =============================================================================
# MODEL 5: EDA ONLY
# =============================================================================

print("\n" + "="*60)
print("MODEL 5: EDA ONLY - ALL 5 SUBJECTS")
print("="*60)

print(f"\nRunning LOSO + 20% calibration with {len(eda_features)} EDA features...")
result_eda = run_loso_calibrated(df_multimodal, eda_features, cal_frac=0.2, name="EDA only")

if result_eda:
    print(f"\n=== EDA ONLY RESULTS ===")
    print(f"Overall: r = {result_eda['r']:.3f}, MAE = {result_eda['mae']:.2f}, ±1 Borg = {result_eda['within_1']:.1f}%")
    print(f"N = {result_eda['n_total']} windows from {result_eda['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_eda['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}%")

# =============================================================================
# MODEL 6: PPG ONLY
# =============================================================================

print("\n" + "="*60)
print("MODEL 6: PPG ONLY - ALL 5 SUBJECTS")
print("="*60)

print(f"\nRunning LOSO + 20% calibration with {len(ppg_features)} PPG features...")
result_ppg = run_loso_calibrated(df_multimodal, ppg_features, cal_frac=0.2, name="PPG only")

if result_ppg:
    print(f"\n=== PPG ONLY RESULTS ===")
    print(f"Overall: r = {result_ppg['r']:.3f}, MAE = {result_ppg['mae']:.2f}, ±1 Borg = {result_ppg['within_1']:.1f}%")
    print(f"N = {result_ppg['n_total']} windows from {result_ppg['n_subjects']} subjects")
    print("\nPer-subject breakdown:")
    for sub, res in sorted(result_ppg['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, ±1 = {res['within_1']:.1f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

results = [
    ("Multimodal (Top 30)", result_multimodal),
    ("Multimodal (All)", result_multimodal_all),
    ("PPG only", result_ppg),
    ("EDA only", result_eda),
    ("IMU only", result_imu),
    ("ECG HR only*", result_ecg),
]

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│          LOSO + 20% Per-Subject Calibration - Final Results                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Model                │ Subjects │ N      │ r     │ MAE  │ ±1 Borg │ Feats  │
│──────────────────────────────────────────────────────────────────────────────│""")

for name, res in results:
    if res:
        print(f"│  {name:<20} │    {res['n_subjects']}     │ {res['n_total']:>5}  │ {res['r']:.3f} │ {res['mae']:.2f} │ {res['within_1']:>5.1f}%  │  {'2' if 'ECG' in name else ('30' if 'Top' in name else 'All'):>4}  │")

print(f"""│──────────────────────────────────────────────────────────────────────────────│
│  * ECG HR only uses 4 subjects (elderly1 has no ECG data)                    │
└──────────────────────────────────────────────────────────────────────────────┘

NOTES:
- All multimodal models use ALL 5 subjects
- ECG HR uses activity-level data (not window-level)
- LOSO = Leave-One-Subject-Out cross-validation
- 20% calibration = use 20% of test subject's data for per-subject calibration
""")

# Per-subject comparison
print("\n" + "="*80)
print("PER-SUBJECT CORRELATION (r) COMPARISON")
print("="*80)

print(f"\n{'Subject':<12} | {'Multimodal':>10} | {'PPG':>10} | {'EDA':>10} | {'IMU':>10} | {'ECG HR':>10}")
print("-" * 75)

for sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    row = f"{sub:<12} |"
    
    for res in [result_multimodal, result_ppg, result_eda, result_imu]:
        if res and sub in res['per_subject']:
            row += f" {res['per_subject'][sub]['r']:>9.3f} |"
        else:
            row += f" {'N/A':>9} |"
    
    # ECG uses different naming
    sub_ecg = sub
    if result_ecg and sub_ecg in result_ecg['per_subject']:
        row += f" {result_ecg['per_subject'][sub_ecg]['r']:>9.3f}"
    else:
        row += f" {'N/A':>9}"
    
    print(row)
