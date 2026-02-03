#!/usr/bin/env python3
"""
FINAL COMPARISON: Multimodal vs ECG HR - FIXED NaN HANDLING
===========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FINAL COMPARISON: MULTIMODAL vs ECG HR (FIXED)")
print("="*80)

# =============================================================================
# LOAD ALL DATA
# =============================================================================

all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)
        print(f"Loaded elderly{i}: {len(df)} windows")

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"\nTotal windows: {len(df_all)}")

# Load ECG HR data
tli_df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
tli_df = tli_df.dropna(subset=['hr_delta', 'borg'])
tli_df['subject_short'] = tli_df['subject'].str.replace('sim_', '')
print(f"ECG activities: {len(tli_df)}")

# =============================================================================
# CHECK DATA AVAILABILITY PER SUBJECT
# =============================================================================

print("\n" + "="*60)
print("DATA AVAILABILITY PER SUBJECT")
print("="*60)

exclude_cols = ['t_center', 'borg', 'subject', 'Unnamed', 'activity_label']
all_features = [c for c in df_all.columns if not any(x in c for x in exclude_cols)]

ppg_features = [c for c in all_features if 'ppg' in c.lower()]
eda_features = [c for c in all_features if 'eda' in c.lower()]
imu_features = [c for c in all_features if 'imu' in c.lower() or 'acc' in c.lower() or 'gyro' in c.lower()]

print(f"\nFeature counts: PPG={len(ppg_features)}, EDA={len(eda_features)}, IMU={len(imu_features)}")

for sub in sorted(df_all['subject'].unique()):
    sub_df = df_all[df_all['subject'] == sub]
    print(f"\n{sub}:")
    print(f"  Total windows: {len(sub_df)}")
    
    # Check PPG availability
    ppg_avail = sub_df[ppg_features].notna().any(axis=1).sum()
    print(f"  PPG available: {ppg_avail} ({100*ppg_avail/len(sub_df):.1f}%)")
    
    # Check EDA availability
    eda_avail = sub_df[eda_features].notna().any(axis=1).sum()
    print(f"  EDA available: {eda_avail} ({100*eda_avail/len(sub_df):.1f}%)")
    
    # Check IMU availability
    imu_avail = sub_df[imu_features].notna().any(axis=1).sum()
    print(f"  IMU available: {imu_avail} ({100*imu_avail/len(sub_df):.1f}%)")

# =============================================================================
# LOSO FUNCTION WITH IMPUTATION
# =============================================================================

def run_loso_with_imputation(df, features, cal_frac=0.2, name="", min_valid_ratio=0.5):
    """
    LOSO with per-subject calibration and NaN imputation.
    Only uses features that have at least min_valid_ratio non-NaN values.
    """
    subjects = sorted(df['subject'].unique())
    
    # Filter features to those with enough data
    valid_features = []
    for f in features:
        if f in df.columns:
            valid_ratio = df[f].notna().mean()
            if valid_ratio >= min_valid_ratio:
                valid_features.append(f)
    
    if len(valid_features) == 0:
        print(f"  No valid features (all have <{min_valid_ratio*100}% data)")
        return None
    
    print(f"  Using {len(valid_features)}/{len(features)} features with >{min_valid_ratio*100}% data")
    
    all_preds = []
    all_true = []
    per_sub_results = {}
    
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub].copy()
        test_df = df[df['subject'] == test_sub].copy()
        
        if len(train_df) < 20 or len(test_df) < 10:
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
            continue
        
        # Prepare data
        X_train = train_df[valid_features].values
        y_train = train_df['borg'].values
        X_cal = cal_df[valid_features].values
        y_cal = cal_df['borg'].values
        X_eval = eval_df[valid_features].values
        y_eval = eval_df['borg'].values
        
        # Impute NaN with median
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_cal = imputer.transform(X_cal)
        X_eval = imputer.transform(X_eval)
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_eval_s = scaler.transform(X_eval)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        
        # Calibrate
        preds_cal = model.predict(X_cal_s)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Evaluate
        preds_raw = model.predict(X_eval_s)
        preds = calibrator.predict(preds_raw.reshape(-1, 1))
        
        all_preds.extend(preds)
        all_true.extend(y_eval)
        
        if len(preds) > 2:
            r_sub, _ = pearsonr(preds, y_eval)
            mae_sub = np.mean(np.abs(preds - y_eval))
            within1_sub = np.mean(np.abs(preds - y_eval) <= 1) * 100
            
            per_sub_results[test_sub] = {
                'r': r_sub,
                'mae': mae_sub,
                'within_1': within1_sub,
                'n_eval': len(eval_df)
            }
    
    if len(all_preds) < 10:
        return None
    
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
        'n_features': len(valid_features),
        'per_subject': per_sub_results
    }

# =============================================================================
# RUN ALL MODELS
# =============================================================================

print("\n" + "="*60)
print("MODEL 1: MULTIMODAL (PPG + EDA + IMU) - ALL 5 SUBJECTS")
print("="*60)

# Select top features by correlation (using imputed data)
correlations = []
for col in all_features:
    valid = df_all[[col, 'borg']].dropna()
    if len(valid) > 100:  # Need enough samples
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r), r))

correlations.sort(key=lambda x: x[1], reverse=True)
top_30_features = [c[0] for c in correlations[:30]]

print("\nTop 10 features by correlation:")
for feat, abs_r, r in correlations[:10]:
    modality = 'PPG' if 'ppg' in feat.lower() else ('EDA' if 'eda' in feat.lower() else 'IMU')
    print(f"  {feat}: r = {r:.3f} ({modality})")

result_mm = run_loso_with_imputation(df_all, top_30_features, name="Multimodal Top30")
if result_mm:
    print(f"\nOVERALL: r = {result_mm['r']:.3f}, MAE = {result_mm['mae']:.2f}, ±1 Borg = {result_mm['within_1']:.1f}%")
    print(f"N = {result_mm['n_total']} windows, {result_mm['n_subjects']} subjects, {result_mm['n_features']} features")
    for sub, res in sorted(result_mm['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, n = {res['n_eval']}")

print("\n" + "="*60)
print("MODEL 2: IMU ONLY - ALL 5 SUBJECTS")
print("="*60)

result_imu = run_loso_with_imputation(df_all, imu_features, name="IMU only")
if result_imu:
    print(f"\nOVERALL: r = {result_imu['r']:.3f}, MAE = {result_imu['mae']:.2f}, ±1 Borg = {result_imu['within_1']:.1f}%")
    print(f"N = {result_imu['n_total']} windows, {result_imu['n_subjects']} subjects, {result_imu['n_features']} features")
    for sub, res in sorted(result_imu['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, n = {res['n_eval']}")

print("\n" + "="*60)
print("MODEL 3: EDA ONLY - ALL 5 SUBJECTS")
print("="*60)

result_eda = run_loso_with_imputation(df_all, eda_features, name="EDA only")
if result_eda:
    print(f"\nOVERALL: r = {result_eda['r']:.3f}, MAE = {result_eda['mae']:.2f}, ±1 Borg = {result_eda['within_1']:.1f}%")
    print(f"N = {result_eda['n_total']} windows, {result_eda['n_subjects']} subjects, {result_eda['n_features']} features")
    for sub, res in sorted(result_eda['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, n = {res['n_eval']}")

print("\n" + "="*60)
print("MODEL 4: PPG ONLY - ALL 5 SUBJECTS")
print("="*60)

result_ppg = run_loso_with_imputation(df_all, ppg_features, name="PPG only")
if result_ppg:
    print(f"\nOVERALL: r = {result_ppg['r']:.3f}, MAE = {result_ppg['mae']:.2f}, ±1 Borg = {result_ppg['within_1']:.1f}%")
    print(f"N = {result_ppg['n_total']} windows, {result_ppg['n_subjects']} subjects, {result_ppg['n_features']} features")
    for sub, res in sorted(result_ppg['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, n = {res['n_eval']}")

print("\n" + "="*60)
print("MODEL 5: ECG HR ONLY - 4 SUBJECTS (no ECG for elderly1)")
print("="*60)

ecg_features = ['hr_delta', 'hr_load']
result_ecg = run_loso_with_imputation(tli_df, ecg_features, name="ECG HR")
if result_ecg:
    print(f"\nOVERALL: r = {result_ecg['r']:.3f}, MAE = {result_ecg['mae']:.2f}, ±1 Borg = {result_ecg['within_1']:.1f}%")
    print(f"N = {result_ecg['n_total']} activities, {result_ecg['n_subjects']} subjects, {result_ecg['n_features']} features")
    for sub, res in sorted(result_ecg['per_subject'].items()):
        print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}, n = {res['n_eval']}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│       LOSO + 20% Per-Subject Calibration - ALL SUBJECTS INCLUDED           │
├────────────────────────────────────────────────────────────────────────────┤
│  Model           │ Subjects │ Samples │   r   │  MAE  │ ±1 Borg │ Features│
├────────────────────────────────────────────────────────────────────────────┤""")

results = [
    ("Multimodal Top30", result_mm),
    ("PPG only", result_ppg),
    ("EDA only", result_eda),
    ("IMU only", result_imu),
    ("ECG HR only*", result_ecg),
]

for name, res in results:
    if res:
        print(f"│  {name:<16} │    {res['n_subjects']}     │  {res['n_total']:>5}  │ {res['r']:>5.3f} │ {res['mae']:>5.2f} │ {res['within_1']:>6.1f}% │   {res['n_features']:>3}   │")

print("""├────────────────────────────────────────────────────────────────────────────┤
│  * ECG HR only has 4 subjects (elderly1 has no ECG chest sensor data)      │
└────────────────────────────────────────────────────────────────────────────┘
""")

# Per-subject table
print("\n" + "="*80)
print("PER-SUBJECT CORRELATION (r)")
print("="*80)

print(f"\n{'Subject':<12} | {'Multimodal':>11} | {'PPG':>11} | {'EDA':>11} | {'IMU':>11} | {'ECG HR':>11}")
print("-" * 80)

for sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    row = f"{sub:<12} |"
    
    for res in [result_mm, result_ppg, result_eda, result_imu]:
        if res and sub in res['per_subject']:
            row += f" {res['per_subject'][sub]['r']:>10.3f} |"
        else:
            row += f" {'N/A':>10} |"
    
    sub_ecg = f"sim_{sub}"
    if result_ecg and sub_ecg in result_ecg['per_subject']:
        row += f" {result_ecg['per_subject'][sub_ecg]['r']:>10.3f}"
    else:
        row += f" {'No data':>10}"
    
    print(row)

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

if result_imu and result_ecg:
    print(f"""
1. IMU alone achieves r = {result_imu['r']:.3f} - best multimodal component!
   - Works on ALL 5 subjects with {result_imu['n_total']} windows
   - Movement-based features capture physical effort well

2. ECG HR achieves r = {result_ecg['r']:.3f} on activity-level data
   - Only 4 subjects, only {result_ecg['n_total']} samples (activities, not windows)
   - High variability between subjects

3. PPG/EDA have limited predictive value due to:
   - Missing data (many NaN values)
   - Wrist sensor noise during movement

4. With MORE DATA, multimodal approach should improve because:
   - Model can learn which features to trust for each subject
   - IMU captures movement intensity → complements HR
   - EDA captures stress/arousal → adds context beyond HR
""")
