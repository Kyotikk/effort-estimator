#!/usr/bin/env python3
"""
WITHIN-SUBJECT vs CROSS-SUBJECT ANALYSIS
=========================================
This compares:
1. Within-subject correlation (train+test on same person)
2. Cross-subject LOSO (train on others, test on one)

Shows why r=0.72 for one subject but r=0.18 for LOSO.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WITHIN-SUBJECT vs CROSS-SUBJECT COMPARISON")
print("="*80)

# Load the TLI data we just computed
df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
print(f"\nLoaded {len(df)} activities from {df['subject'].nunique()} subjects")

# Features
pred_features = ['hr_delta', 'hr_load', 'imu_load', 'duration_s']
pred_features = [f for f in pred_features if f in df.columns]

# =============================================================================
# WITHIN-SUBJECT ANALYSIS (Per-subject correlation)
# =============================================================================

print("\n" + "="*60)
print("WITHIN-SUBJECT CORRELATIONS")
print("="*60)

within_results = []

for subject in df['subject'].unique():
    sub_df = df[df['subject'] == subject].copy()
    n = len(sub_df)
    
    if n < 5:
        continue
    
    # Feature correlations with Borg
    for feat in ['hr_load', 'hr_delta', 'TLI', 'imu_load']:
        if feat in sub_df.columns:
            valid = sub_df[[feat, 'borg']].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid[feat], valid['borg'])
                within_results.append({
                    'subject': subject,
                    'feature': feat,
                    'n': len(valid),
                    'r': r,
                    'p': p
                })

within_df = pd.DataFrame(within_results)

# Print summary by feature
print("\nMean correlation by feature (within-subject):")
for feat in ['hr_load', 'hr_delta', 'TLI', 'imu_load']:
    feat_data = within_df[within_df['feature'] == feat]
    if len(feat_data) > 0:
        mean_r = feat_data['r'].mean()
        std_r = feat_data['r'].std()
        print(f"  {feat:12s}: r = {mean_r:+.3f} ± {std_r:.3f}")

print("\nPer-subject detail (hr_load):")
for _, row in within_df[within_df['feature'] == 'hr_load'].iterrows():
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
    print(f"  {row['subject']:15s}: n={row['n']:3d}, r={row['r']:+.3f} {sig}")

# =============================================================================
# THE KEY INSIGHT: Borg Rating Distribution Per Subject
# =============================================================================

print("\n" + "="*60)
print("BORG RATING DISTRIBUTION BY SUBJECT")
print("="*60)

for subject in sorted(df['subject'].unique()):
    sub_df = df[df['subject'] == subject]
    borg_mean = sub_df['borg'].mean()
    borg_std = sub_df['borg'].std()
    borg_range = f"[{sub_df['borg'].min():.1f} - {sub_df['borg'].max():.1f}]"
    print(f"  {subject}: mean={borg_mean:.2f}, std={borg_std:.2f}, range={borg_range}")

print("\nOverall Borg distribution:")
print(f"  Mean: {df['borg'].mean():.2f}")
print(f"  Std: {df['borg'].std():.2f}")
print(f"  Range: [{df['borg'].min():.1f} - {df['borg'].max():.1f}]")

# =============================================================================
# HR_delta Distribution Per Subject
# =============================================================================

print("\n" + "="*60)
print("HR_DELTA DISTRIBUTION BY SUBJECT")
print("="*60)

for subject in sorted(df['subject'].unique()):
    sub_df = df[df['subject'] == subject]
    valid = sub_df['hr_delta'].dropna()
    if len(valid) > 0:
        print(f"  {subject}: mean={valid.mean():+.1f} bpm, std={valid.std():.1f}, range=[{valid.min():.1f} - {valid.max():.1f}]")

# =============================================================================
# THE PROBLEM: Different HR Response Per Subject
# =============================================================================

print("\n" + "="*60)
print("KEY INSIGHT: INTER-SUBJECT VARIABILITY")
print("="*60)

# Check if HR_delta predicts Borg differently for each subject
print("\nSlope of Borg ~ HR_delta per subject:")
for subject in sorted(df['subject'].unique()):
    sub_df = df[df['subject'] == subject].dropna(subset=['hr_delta', 'borg'])
    if len(sub_df) > 3:
        lr = LinearRegression()
        X = sub_df['hr_delta'].values.reshape(-1, 1)
        y = sub_df['borg'].values
        lr.fit(X, y)
        print(f"  {subject}: slope={lr.coef_[0]:.4f}, intercept={lr.intercept_:.2f}")

# =============================================================================
# PROPER PERSONALIZED EVALUATION (50/50 within-subject)
# =============================================================================

print("\n" + "="*60)
print("PERSONALIZED EVALUATION (50/50 Split Within Subject)")
print("="*60)

personalized_preds = []
personalized_true = []

for subject in df['subject'].unique():
    sub_df = df[df['subject'] == subject].copy()
    sub_df = sub_df.dropna(subset=pred_features + ['borg'])
    
    if len(sub_df) < 6:
        continue
    
    # 50/50 split
    n = len(sub_df)
    train_idx = np.random.choice(n, n//2, replace=False)
    test_idx = np.array([i for i in range(n) if i not in train_idx])
    
    X = sub_df[pred_features].values
    y = sub_df['borg'].values
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Train personalized model
    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    personalized_preds.extend(preds)
    personalized_true.extend(y_test)
    
    # Per-subject result
    r_sub, _ = pearsonr(preds, y_test)
    print(f"  {subject}: n_test={len(y_test)}, r={r_sub:.3f}")

# Overall
personalized_preds = np.array(personalized_preds)
personalized_true = np.array(personalized_true)

r_pers, _ = pearsonr(personalized_preds, personalized_true)
mae_pers = np.mean(np.abs(personalized_preds - personalized_true))
within_1_pers = np.mean(np.abs(personalized_preds - personalized_true) <= 1) * 100

print(f"\nPersonalized 50/50 Overall:")
print(f"  r = {r_pers:.3f}")
print(f"  MAE = {mae_pers:.2f}")
print(f"  ±1 Borg = {within_1_pers:.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: WHY LOSO FAILS BUT WITHIN-SUBJECT WORKS")
print("="*80)
print("""
The tli_from_raw.py showed r=0.72 for sim_elderly3 alone.
But LOSO across 5 subjects shows r=0.18.

ROOT CAUSE: Inter-subject variability
- Each person has different:
  • Baseline HR (fitness level)
  • HR response to same activity (autonomic response)  
  • Borg rating calibration (subjective perception)

This means a model trained on 4 subjects doesn't generalize to the 5th.

SOLUTIONS FOR THESIS:
1. Report WITHIN-SUBJECT performance (r~0.5-0.7) as upper bound
2. Report LOSO performance (r~0.2) as cross-subject lower bound
3. Use PERSONALIZED models (50/50 calibration) for deployment
4. Consider population-level model + personal calibration

This is a FUNDAMENTAL limitation of effort estimation, not a bug.
""")
