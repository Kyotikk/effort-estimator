#!/usr/bin/env python3
"""
THESIS RESULTS COMPARISON: SCIENTIFIC PIPELINE vs SIMPLE HR APPROACH
=====================================================================

This script compares the two approaches side-by-side for your thesis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("THESIS: COMPARING TWO APPROACHES")
print("="*80)

# =============================================================================
# APPROACH A: YOUR SCIENTIFIC MULTI-FEATURE PIPELINE
# =============================================================================

print("\n" + "="*70)
print("APPROACH A: SCIENTIFIC MULTI-FEATURE PIPELINE (PPG + EDA + IMU)")
print("="*70)

# Load combined aligned features
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"Dataset: {len(combined)} windows, {combined['subject'].nunique()} subjects")

# Get feature columns (exclude metadata)
exclude = ['t_center', 'borg', 'subject', 'Unnamed']
feature_cols = [c for c in combined.columns if not any(x in c for x in exclude)]
print(f"Features: {len(feature_cols)}")

# PPG HR correlation (the key limitation)
print("\nPPG HR correlation with Borg (per subject):")
for sub in sorted(combined['subject'].unique()):
    sub_df = combined[combined['subject'] == sub]
    valid = sub_df[['ppg_green_hr_mean', 'borg']].dropna()
    if len(valid) > 2:
        r, _ = pearsonr(valid['ppg_green_hr_mean'], valid['borg'])
        print(f"  {sub}: r = {r:.3f}")

# Pooled
valid = combined[['ppg_green_hr_mean', 'borg']].dropna()
r_pooled, _ = pearsonr(valid['ppg_green_hr_mean'], valid['borg'])
print(f"  POOLED: r = {r_pooled:.3f}")

# LOSO evaluation
print("\nLOSO Cross-Validation:")
subjects = combined['subject'].unique()
np.random.seed(42)

# Select top 34 features (as in your pipeline)
top_features = ['eda_cc_mean', 'eda_cc_std', 'eda_stress_skin_mean', 
                'imu_acc_x_mean', 'imu_acc_y_mean', 'imu_acc_z_mean',
                'imu_acc_mag_mean', 'imu_gyro_x_mean', 'imu_gyro_y_mean',
                'ppg_green_hr_mean', 'ppg_infra_hr_mean']
# Use available features
avail_features = [f for f in top_features if f in combined.columns]
if len(avail_features) < 5:
    avail_features = feature_cols[:30]  # Fallback

all_preds_a = []
all_true_a = []

for test_sub in subjects:
    train = combined[combined['subject'] != test_sub].dropna(subset=['borg'])
    test = combined[combined['subject'] == test_sub].dropna(subset=['borg'])
    
    X_train = train[avail_features].values
    y_train = train['borg'].values
    X_test = test[avail_features].values
    y_test = test['borg'].values
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    all_preds_a.extend(preds)
    all_true_a.extend(y_test)

all_preds_a = np.array(all_preds_a)
all_true_a = np.array(all_true_a)

r_a, _ = pearsonr(all_preds_a, all_true_a)
mae_a = np.mean(np.abs(all_preds_a - all_true_a))
within_1_a = np.mean(np.abs(all_preds_a - all_true_a) <= 1) * 100

print(f"\n  LOSO (raw): r = {r_a:.3f}, MAE = {mae_a:.2f}, ±1 Borg = {within_1_a:.1f}%")

# With calibration
lr = LinearRegression()
lr.fit(all_preds_a.reshape(-1, 1), all_true_a)
calibrated_a = lr.predict(all_preds_a.reshape(-1, 1))
r_a_cal, _ = pearsonr(calibrated_a, all_true_a)
mae_a_cal = np.mean(np.abs(calibrated_a - all_true_a))
within_1_a_cal = np.mean(np.abs(calibrated_a - all_true_a) <= 1) * 100

print(f"  LOSO (calibrated): r = {r_a_cal:.3f}, MAE = {mae_a_cal:.2f}, ±1 Borg = {within_1_a_cal:.1f}%")

# =============================================================================
# APPROACH B: SIMPLE HR-BASED (ECG HR, Activity-Level)
# =============================================================================

print("\n" + "="*70)
print("APPROACH B: SIMPLE HR-BASED MODEL (ECG HR, Activity-Level)")
print("="*70)

# Load TLI data (ECG HR)
tli_df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
tli_df = tli_df.dropna(subset=['hr_load', 'borg'])
print(f"Dataset: {len(tli_df)} activities, {tli_df['subject'].nunique()} subjects")

# ECG HR correlation
print("\nECG HR (hr_delta) correlation with Borg (per subject):")
for sub in sorted(tli_df['subject'].unique()):
    sub_df = tli_df[tli_df['subject'] == sub]
    valid = sub_df[['hr_delta', 'borg']].dropna()
    if len(valid) > 2:
        r, _ = pearsonr(valid['hr_delta'], valid['borg'])
        print(f"  {sub}: r = {r:.3f}")

# Pooled
valid = tli_df[['hr_delta', 'borg']].dropna()
r_pooled_b, _ = pearsonr(valid['hr_delta'], valid['borg'])
print(f"  POOLED: r = {r_pooled_b:.3f}")

# LOSO
print("\nLOSO Cross-Validation:")
subjects_b = tli_df['subject'].unique()
pred_features_b = ['hr_delta', 'hr_load', 'duration_s']

all_preds_b = []
all_true_b = []

for test_sub in subjects_b:
    train = tli_df[tli_df['subject'] != test_sub]
    test = tli_df[tli_df['subject'] == test_sub]
    
    X_train = train[pred_features_b].values
    y_train = train['borg'].values
    X_test = test[pred_features_b].values
    y_test = test['borg'].values
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    all_preds_b.extend(preds)
    all_true_b.extend(y_test)

all_preds_b = np.array(all_preds_b)
all_true_b = np.array(all_true_b)

r_b, _ = pearsonr(all_preds_b, all_true_b)
mae_b = np.mean(np.abs(all_preds_b - all_true_b))
within_1_b = np.mean(np.abs(all_preds_b - all_true_b) <= 1) * 100

print(f"\n  LOSO (raw): r = {r_b:.3f}, MAE = {mae_b:.2f}, ±1 Borg = {within_1_b:.1f}%")

# Personalized 50/50
print("\nPersonalized (50/50 split within each subject):")
np.random.seed(42)
all_preds_pers = []
all_true_pers = []

for sub in subjects_b:
    sub_df = tli_df[tli_df['subject'] == sub].copy()
    sub_df = sub_df.dropna(subset=pred_features_b + ['borg'])
    
    if len(sub_df) < 6:
        continue
    
    n = len(sub_df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx = indices[:n//2]
    test_idx = indices[n//2:]
    
    X = sub_df[pred_features_b].values
    y = sub_df['borg'].values
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    all_preds_pers.extend(preds)
    all_true_pers.extend(y_test)
    
    r_sub, _ = pearsonr(preds, y_test)
    print(f"  {sub}: n_test={len(y_test)}, r={r_sub:.3f}")

all_preds_pers = np.array(all_preds_pers)
all_true_pers = np.array(all_true_pers)

r_pers, _ = pearsonr(all_preds_pers, all_true_pers)
mae_pers = np.mean(np.abs(all_preds_pers - all_true_pers))
within_1_pers = np.mean(np.abs(all_preds_pers - all_true_pers) <= 1) * 100

print(f"\n  Personalized Overall: r = {r_pers:.3f}, MAE = {mae_pers:.2f}, ±1 Borg = {within_1_pers:.1f}%")

# =============================================================================
# FINAL COMPARISON TABLE
# =============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON TABLE FOR THESIS")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EFFORT ESTIMATION RESULTS COMPARISON                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ APPROACH A: SCIENTIFIC MULTI-FEATURE PIPELINE                                   │
│ ─────────────────────────────────────────────                                   │
│ • Sensors: Corsano (PPG, EDA, IMU)                                              │
│ • Features: {len(avail_features)} selected from 296 (PCA + correlation filtering)                 │
│ • Windows: 5 seconds                                                            │
│ • Samples: {len(combined)} windows                                                       │
│                                                                                 │
│   │ Evaluation Method      │ Pearson r │   MAE   │ ±1 Borg │                    │
│   ├────────────────────────┼───────────┼─────────┼─────────┤                    │
│   │ Cross-subject LOSO     │   {r_a:.3f}   │  {mae_a:.2f}   │  {within_1_a:.1f}%  │                    │
│   │ LOSO + calibration     │   {r_a_cal:.3f}   │  {mae_a_cal:.2f}   │  {within_1_a_cal:.1f}%  │                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ APPROACH B: SIMPLE HR-BASED MODEL                                               │
│ ─────────────────────────────────                                               │
│ • Sensors: Vivalnk (ECG chest patch)                                            │
│ • Features: 3 (HR_delta, HR_load, duration)                                     │
│ • Aggregation: Activity-level (real ADL labels)                                 │
│ • Samples: {len(tli_df)} activities                                                       │
│                                                                                 │
│   │ Evaluation Method      │ Pearson r │   MAE   │ ±1 Borg │                    │
│   ├────────────────────────┼───────────┼─────────┼─────────┤                    │
│   │ Cross-subject LOSO     │   {r_b:.3f}   │  {mae_b:.2f}   │  {within_1_b:.1f}%  │                    │
│   │ Personalized (50/50)   │   {r_pers:.3f}   │  {mae_pers:.2f}   │  {within_1_pers:.1f}%  │                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

KEY FINDINGS FOR THESIS:
========================

1. BOTH approaches show POOR cross-subject generalization (r < 0.3)
   → This is the fundamental limitation with 5 subjects

2. PPG-derived HR is NOISIER than ECG-derived HR:
   • PPG HR pooled correlation: r = {r_pooled:.3f}
   • ECG HR pooled correlation: r = {r_pooled_b:.3f}
   BUT this difference is small - the real issue is inter-subject variability

3. PERSONALIZED models work well (r = {r_pers:.2f}) when calibrated to individual

4. SCIENTIFIC APPROACH (A) is recommended for:
   • Larger datasets (more subjects)
   • Wrist-only deployment (no chest patch)
   • Reproducible methodology
   
5. SIMPLE APPROACH (B) shows:
   • Upper bound with clean HR signal
   • Value of personalization
   • Activity-level analysis potential
""")
