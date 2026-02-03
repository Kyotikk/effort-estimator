#!/usr/bin/env python3
"""
Improved Thesis Plots
=====================
1. Feature importance - bluish colors, full names visible
2. Scatter plots - WITHIN patient (shows the model CAN work)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

paths = [
    '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
]

print("Loading data...")
dfs = []
for i, p in enumerate(paths, 1):
    df = pd.read_csv(p)
    df['subject'] = f'P{i}'
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
labeled = combined.dropna(subset=['borg'])

# Feature selection
imu_cols = [c for c in combined.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in combined.columns if 'ppg_' in c]

print(f"  {len(labeled)} samples, IMU={len(imu_cols)}, PPG={len(ppg_cols)}")

out_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
out_dir.mkdir(exist_ok=True)

# ============================================================================
# PLOT 1: Feature Importance - Bluish colors, full names
# ============================================================================
print("\nCreating feature importance plot...")

# Train on all data to get feature importance
X_imu = labeled[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
X_ppg = labeled[ppg_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y = labeled['borg'].values

rf_imu = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_imu.fit(X_imu, y)

rf_ppg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_ppg.fit(X_ppg, y)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# IMU features
ax = axes[0]
idx = np.argsort(rf_imu.feature_importances_)[::-1][:10]
top_imp = rf_imu.feature_importances_[idx]
top_names = [imu_cols[i] for i in idx]

# Clean names - make readable
clean_names = []
for n in top_names:
    n = n.replace('acc_', 'Acc ').replace('_dyn_', ' dyn: ').replace('_', ' ')
    clean_names.append(n)

y_pos = np.arange(len(clean_names))
bars = ax.barh(y_pos, top_imp, color='#4682B4', edgecolor='white', height=0.7)  # Steel blue
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (RF)', fontsize=12)
ax.set_title(f'Top 10 IMU Features (n={len(imu_cols)})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, max(top_imp) * 1.1)

# PPG features
ax = axes[1]
idx = np.argsort(rf_ppg.feature_importances_)[::-1][:10]
top_imp = rf_ppg.feature_importances_[idx]
top_names = [ppg_cols[i] for i in idx]

clean_names = []
for n in top_names:
    n = n.replace('ppg_', 'PPG ').replace('green_', 'green: ').replace('_', ' ')
    clean_names.append(n)

y_pos = np.arange(len(clean_names))
bars = ax.barh(y_pos, top_imp, color='#5F9EA0', edgecolor='white', height=0.7)  # Cadet blue
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (RF)', fontsize=12)
ax.set_title(f'Top 10 PPG Features (n={len(ppg_cols)})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, max(top_imp) * 1.1)

plt.tight_layout()
plt.savefig(out_dir / '58_features_improved.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '58_features_improved.png'}")

# ============================================================================
# PLOT 2: WITHIN-PATIENT Scatter (shows model CAN work)
# ============================================================================
print("\nCreating within-patient scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Collect within-patient predictions
within_true_imu, within_pred_imu = [], []
within_true_ppg, within_pred_ppg = [], []
per_subj_r_imu, per_subj_r_ppg = [], []

for subj in labeled['subject'].unique():
    subj_data = labeled[labeled['subject'] == subj]
    
    X_imu_s = subj_data[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_ppg_s = subj_data[ppg_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_s = subj_data['borg'].values
    
    if len(y_s) < 20:
        continue
    
    # 70/30 split within patient
    X_train_imu, X_test_imu, y_train, y_test = train_test_split(
        X_imu_s, y_s, test_size=0.3, random_state=42
    )
    X_train_ppg, X_test_ppg, _, _ = train_test_split(
        X_ppg_s, y_s, test_size=0.3, random_state=42
    )
    
    # IMU
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_imu, y_train)
    y_pred_imu = rf.predict(X_test_imu)
    within_true_imu.extend(y_test)
    within_pred_imu.extend(y_pred_imu)
    r_imu, _ = pearsonr(y_test, y_pred_imu)
    per_subj_r_imu.append(r_imu)
    
    # PPG
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_ppg, y_train)
    y_pred_ppg = rf.predict(X_test_ppg)
    within_true_ppg.extend(y_test)
    within_pred_ppg.extend(y_pred_ppg)
    r_ppg, _ = pearsonr(y_test, y_pred_ppg)
    per_subj_r_ppg.append(r_ppg)

within_true_imu, within_pred_imu = np.array(within_true_imu), np.array(within_pred_imu)
within_true_ppg, within_pred_ppg = np.array(within_true_ppg), np.array(within_pred_ppg)

mean_r_imu = np.mean(per_subj_r_imu)
mean_r_ppg = np.mean(per_subj_r_ppg)
mae_imu = np.mean(np.abs(within_true_imu - within_pred_imu))
mae_ppg = np.mean(np.abs(within_true_ppg - within_pred_ppg))

# IMU scatter
ax = axes[0]
ax.scatter(within_true_imu, within_pred_imu, alpha=0.5, s=30, c='#4682B4', edgecolor='none')
ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2, label='Perfect')
z = np.polyfit(within_true_imu, within_pred_imu, 1)
p = np.poly1d(z)
ax.plot([0, 7], [p(0), p(7)], color='#4682B4', linewidth=2, alpha=0.8)
ax.set_xlabel('Actual Borg', fontsize=12)
ax.set_ylabel('Predicted Borg', fontsize=12)
ax.set_title(f'IMU Within-Patient: r = {mean_r_imu:.2f}, MAE = {mae_imu:.2f}', fontsize=14, fontweight='bold')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# PPG scatter
ax = axes[1]
ax.scatter(within_true_ppg, within_pred_ppg, alpha=0.5, s=30, c='#5F9EA0', edgecolor='none')
ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2, label='Perfect')
z = np.polyfit(within_true_ppg, within_pred_ppg, 1)
p = np.poly1d(z)
ax.plot([0, 7], [p(0), p(7)], color='#5F9EA0', linewidth=2, alpha=0.8)
ax.set_xlabel('Actual Borg', fontsize=12)
ax.set_ylabel('Predicted Borg', fontsize=12)
ax.set_title(f'PPG Within-Patient: r = {mean_r_ppg:.2f}, MAE = {mae_ppg:.2f}', fontsize=14, fontweight='bold')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / '59_scatter_within_patient.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '59_scatter_within_patient.png'}")

# ============================================================================
# PLOT 3: Comparison - Within vs Cross patient
# ============================================================================
print("\nCreating comparison plot (within vs cross patient)...")

# Get LOSO (cross-patient) results
loso_true_imu, loso_pred_imu = [], []
loso_per_subj_r = []

for test_subj in labeled['subject'].unique():
    train = labeled[labeled['subject'] != test_subj]
    test = labeled[labeled['subject'] == test_subj]
    
    X_train = train[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train['borg'].values
    X_test = test[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    loso_true_imu.extend(y_test)
    loso_pred_imu.extend(y_pred)
    r, _ = pearsonr(y_test, y_pred)
    loso_per_subj_r.append(r)

loso_true_imu, loso_pred_imu = np.array(loso_true_imu), np.array(loso_pred_imu)
loso_mean_r = np.mean(loso_per_subj_r)
loso_mae = np.mean(np.abs(loso_true_imu - loso_pred_imu))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Within-patient
ax = axes[0]
ax.scatter(within_true_imu, within_pred_imu, alpha=0.5, s=30, c='#2E8B57', edgecolor='none')
ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2)
z = np.polyfit(within_true_imu, within_pred_imu, 1)
p = np.poly1d(z)
ax.plot([0, 7], [p(0), p(7)], color='#2E8B57', linewidth=2)
ax.set_xlabel('Actual Borg', fontsize=12)
ax.set_ylabel('Predicted Borg', fontsize=12)
ax.set_title(f'Within-Patient: r = {mean_r_imu:.2f}', fontsize=14, fontweight='bold', color='#2E8B57')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.05, f'MAE = {mae_imu:.2f}', transform=ax.transAxes, ha='right', fontsize=11)

# Cross-patient (LOSO)
ax = axes[1]
ax.scatter(loso_true_imu, loso_pred_imu, alpha=0.5, s=30, c='#4682B4', edgecolor='none')
ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2)
z = np.polyfit(loso_true_imu, loso_pred_imu, 1)
p = np.poly1d(z)
ax.plot([0, 7], [p(0), p(7)], color='#4682B4', linewidth=2)
ax.set_xlabel('Actual Borg', fontsize=12)
ax.set_ylabel('Predicted Borg', fontsize=12)
ax.set_title(f'Cross-Patient (LOSO): r = {loso_mean_r:.2f}', fontsize=14, fontweight='bold', color='#4682B4')
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.05, f'MAE = {loso_mae:.2f}', transform=ax.transAxes, ha='right', fontsize=11)

plt.tight_layout()
plt.savefig(out_dir / '60_within_vs_cross_patient.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '60_within_vs_cross_patient.png'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
WITHIN-PATIENT (personalized model):
  IMU: r = {mean_r_imu:.2f}, MAE = {mae_imu:.2f}
  PPG: r = {mean_r_ppg:.2f}, MAE = {mae_ppg:.2f}

CROSS-PATIENT (LOSO - generalized model):
  IMU: r = {loso_mean_r:.2f}, MAE = {loso_mae:.2f}

This shows:
- Within-patient works MUCH better (model learns person-specific patterns)
- Cross-patient is harder (effort expression varies between people)
""")
