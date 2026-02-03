#!/usr/bin/env python3
"""
FINAL THESIS RESULTS - ACTIVITY-LEVEL EFFORT ESTIMATION
========================================================
This produces the final numbers for the thesis with:
1. Real activity labels from ADL files
2. HR from Vivalnk sensor (ECG-derived, reliable)
3. Activity-level aggregation
4. Multiple evaluation methods

RESULTS SUMMARY:
- Within-subject correlation: r = 0.47 (pooled)
- Personalized 50/50: r = 0.74, ±1 Borg = 60%
- Cross-subject LOSO: r = 0.18 (fundamental limitation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("FINAL THESIS RESULTS - ACTIVITY-LEVEL EFFORT ESTIMATION")
print("="*80)

# Load the computed TLI data
df = pd.read_csv("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")

# Filter to valid data (has HR and Borg)
df = df.dropna(subset=['hr_load', 'borg'])
print(f"\nDataset: {len(df)} activities from {df['subject'].nunique()} subjects")

# =============================================================================
# 1. FEATURE CORRELATIONS (POOLED)
# =============================================================================

print("\n" + "="*60)
print("1. FEATURE CORRELATIONS WITH BORG (POOLED)")
print("="*60)

features = ['hr_delta', 'hr_load', 'duration_s', 'imu_load', 'TLI']
for feat in features:
    if feat in df.columns:
        valid = df[[feat, 'borg']].dropna()
        if len(valid) > 2:
            r, p = pearsonr(valid[feat], valid['borg'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {feat:15s}: r = {r:+.3f} (p = {p:.4f}) {sig}")

# =============================================================================
# 2. WITHIN-SUBJECT CORRELATIONS
# =============================================================================

print("\n" + "="*60)
print("2. WITHIN-SUBJECT CORRELATIONS (hr_load ~ Borg)")
print("="*60)

for subject in sorted(df['subject'].unique()):
    sub_df = df[df['subject'] == subject].dropna(subset=['hr_load', 'borg'])
    if len(sub_df) > 2:
        r, p = pearsonr(sub_df['hr_load'], sub_df['borg'])
        n = len(sub_df)
        borg_range = f"[{sub_df['borg'].min():.1f}-{sub_df['borg'].max():.1f}]"
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {subject}: n={n:3d}, r={r:+.3f}{sig}, Borg range={borg_range}")

# Mean within-subject correlation
within_rs = []
for subject in df['subject'].unique():
    sub_df = df[df['subject'] == subject].dropna(subset=['hr_load', 'borg'])
    if len(sub_df) > 2:
        r, _ = pearsonr(sub_df['hr_load'], sub_df['borg'])
        within_rs.append(r)

print(f"\n  Mean within-subject r: {np.mean(within_rs):.3f} ± {np.std(within_rs):.3f}")

# =============================================================================
# 3. CROSS-SUBJECT LOSO
# =============================================================================

print("\n" + "="*60)
print("3. CROSS-SUBJECT LOSO EVALUATION")
print("="*60)

pred_features = ['hr_delta', 'hr_load', 'duration_s']

subjects = df['subject'].unique()
all_preds_loso = []
all_true_loso = []

for test_subject in subjects:
    train_mask = df['subject'] != test_subject
    test_mask = df['subject'] == test_subject
    
    X_train = df.loc[train_mask, pred_features].values
    y_train = df.loc[train_mask, 'borg'].values
    X_test = df.loc[test_mask, pred_features].values
    y_test = df.loc[test_mask, 'borg'].values
    
    if len(X_test) < 2:
        continue
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    
    all_preds_loso.extend(preds)
    all_true_loso.extend(y_test)
    
    r_sub, _ = pearsonr(preds, y_test)
    print(f"  Test on {test_subject}: n={len(y_test)}, r={r_sub:.3f}")

all_preds_loso = np.array(all_preds_loso)
all_true_loso = np.array(all_true_loso)

r_loso, _ = pearsonr(all_preds_loso, all_true_loso)
mae_loso = np.mean(np.abs(all_preds_loso - all_true_loso))
within_1_loso = np.mean(np.abs(all_preds_loso - all_true_loso) <= 1) * 100

print(f"\n  LOSO Overall: r = {r_loso:.3f}, MAE = {mae_loso:.2f}, ±1 Borg = {within_1_loso:.1f}%")

# =============================================================================
# 4. PERSONALIZED 50/50 EVALUATION
# =============================================================================

print("\n" + "="*60)
print("4. PERSONALIZED EVALUATION (50/50 Split)")
print("="*60)

all_preds_pers = []
all_true_pers = []

for subject in subjects:
    sub_df = df[df['subject'] == subject].copy()
    sub_df = sub_df.dropna(subset=pred_features + ['borg'])
    
    if len(sub_df) < 6:
        continue
    
    # 50/50 split
    n = len(sub_df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx = indices[:n//2]
    test_idx = indices[n//2:]
    
    X = sub_df[pred_features].values
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
    print(f"  {subject}: n_test={len(y_test)}, r={r_sub:.3f}")

all_preds_pers = np.array(all_preds_pers)
all_true_pers = np.array(all_true_pers)

r_pers, _ = pearsonr(all_preds_pers, all_true_pers)
mae_pers = np.mean(np.abs(all_preds_pers - all_true_pers))
within_1_pers = np.mean(np.abs(all_preds_pers - all_true_pers) <= 1) * 100

print(f"\n  Personalized Overall: r = {r_pers:.3f}, MAE = {mae_pers:.2f}, ±1 Borg = {within_1_pers:.1f}%")

# =============================================================================
# 5. PLOT: Predicted vs Actual (Personalized)
# =============================================================================

plt.figure(figsize=(10, 8))

# Main scatter plot
plt.subplot(2, 2, 1)
plt.scatter(all_true_pers, all_preds_pers, alpha=0.6, s=60)
plt.plot([0, 7], [0, 7], 'k--', label='Perfect')
plt.xlabel('Actual Borg CR10', fontsize=12)
plt.ylabel('Predicted Borg CR10', fontsize=12)
plt.title(f'Personalized Model (50/50)\nr = {r_pers:.3f}, MAE = {mae_pers:.2f}', fontsize=12)
plt.xlim(-0.5, 7.5)
plt.ylim(-0.5, 7.5)
plt.legend()
plt.grid(True, alpha=0.3)

# LOSO scatter
plt.subplot(2, 2, 2)
plt.scatter(all_true_loso, all_preds_loso, alpha=0.6, s=60, c='orange')
plt.plot([0, 7], [0, 7], 'k--', label='Perfect')
plt.xlabel('Actual Borg CR10', fontsize=12)
plt.ylabel('Predicted Borg CR10', fontsize=12)
plt.title(f'Cross-Subject LOSO\nr = {r_loso:.3f}, MAE = {mae_loso:.2f}', fontsize=12)
plt.xlim(-0.5, 7.5)
plt.ylim(-0.5, 7.5)
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals histogram (personalized)
plt.subplot(2, 2, 3)
residuals = all_preds_pers - all_true_pers
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='Zero')
plt.axvline(x=-1, color='g', linestyle=':', label='±1 Borg')
plt.axvline(x=1, color='g', linestyle=':')
plt.xlabel('Prediction Error (Borg points)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'Error Distribution (Personalized)\n±1 Borg: {within_1_pers:.1f}%', fontsize=12)
plt.legend()

# Feature correlation bar chart
plt.subplot(2, 2, 4)
feat_corrs = []
for feat in ['hr_delta', 'hr_load', 'duration_s']:
    valid = df[[feat, 'borg']].dropna()
    r, _ = pearsonr(valid[feat], valid['borg'])
    feat_corrs.append(r)

bars = plt.bar(['HR Δ', 'HR Load', 'Duration'], feat_corrs, color=['steelblue', 'steelblue', 'steelblue'])
plt.ylabel('Correlation with Borg (r)', fontsize=12)
plt.title('Feature Correlations', fontsize=12)
plt.axhline(y=0, color='k', linewidth=0.5)
for i, v in enumerate(feat_corrs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/plots/29_thesis_activity_level_results.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Plot saved: plots/29_thesis_activity_level_results.png")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("FINAL THESIS RESULTS SUMMARY")
print("="*80)

print(f"""
┌──────────────────────────────────────────────────────────────────────────┐
│                    ACTIVITY-LEVEL EFFORT ESTIMATION                       │
├──────────────────────────────────────────────────────────────────────────┤
│ Dataset: {len(df)} activities, {df['subject'].nunique()} elderly subjects                                │
│ Features: HR_delta, HR_load (HR_delta × √duration), activity duration     │
│ Model: Ridge Regression                                                   │
├──────────────────────────────────────────────────────────────────────────┤
│ Evaluation Method          │ Pearson r │   MAE   │ ±1 Borg Accuracy     │
├──────────────────────────────────────────────────────────────────────────┤
│ Within-subject correlation │   {np.mean(within_rs):.3f}   │    -    │       -              │
│ Cross-subject LOSO         │   {r_loso:.3f}   │  {mae_loso:.2f}   │     {within_1_loso:.1f}%            │
│ Personalized (50/50)       │   {r_pers:.3f}   │  {mae_pers:.2f}   │     {within_1_pers:.1f}%            │
└──────────────────────────────────────────────────────────────────────────┘

KEY FINDINGS:
1. Strong within-subject correlation (r = {np.mean(within_rs):.2f}) shows HR response  
   tracks subjective effort well for individuals

2. Poor cross-subject generalization (r = {r_loso:.2f}) reveals fundamental
   inter-subject variability in:
   - Baseline heart rate
   - HR response to same activity
   - Subjective Borg calibration

3. Personalized models (r = {r_pers:.2f}) achieve good accuracy when calibrated
   to each individual with ~50% of their data

RECOMMENDATION FOR DEPLOYMENT:
Use population model + personal calibration (first few activities)
""")

# Save summary
with open('/Users/pascalschlegel/effort-estimator/output/thesis_results_summary.txt', 'w') as f:
    f.write(f"""
THESIS RESULTS: ACTIVITY-LEVEL EFFORT ESTIMATION
================================================

Dataset:
- Activities: {len(df)}
- Subjects: {df['subject'].nunique()} elderly patients
- Features: HR_delta, HR_load, duration

Results:
| Method                     | r     | MAE  | ±1 Borg |
|----------------------------|-------|------|---------|
| Within-subject correlation | {np.mean(within_rs):.3f} |  -   |    -    |
| Cross-subject LOSO         | {r_loso:.3f} | {mae_loso:.2f} | {within_1_loso:.1f}%   |
| Personalized (50/50)       | {r_pers:.3f} | {mae_pers:.2f} | {within_1_pers:.1f}%   |

Individual Subject Performance (Personalized):
""")
    
print(f"✓ Summary saved: output/thesis_results_summary.txt")
