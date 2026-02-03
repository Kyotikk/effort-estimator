#!/usr/bin/env python3
"""
2x2 Scatter Plot Comparison
===========================
Top row: Within-patient (overfit due to random split!)
Bottom row: Cross-patient (LOSO - honest metric)

Columns: IMU | PPG
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

imu_cols = [c for c in combined.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in combined.columns if 'ppg_' in c]

print(f"  {len(labeled)} samples, IMU={len(imu_cols)}, PPG={len(ppg_cols)}")

# ============================================================================
# Collect all predictions
# ============================================================================

# Within-patient (random split - OVERFIT!)
within_imu_true, within_imu_pred = [], []
within_ppg_true, within_ppg_pred = [], []
within_imu_r, within_ppg_r = [], []

# Cross-patient (LOSO - honest)
loso_imu_true, loso_imu_pred = [], []
loso_ppg_true, loso_ppg_pred = [], []
loso_imu_r, loso_ppg_r = [], []

print("\nRunning evaluations...")

for subj in labeled['subject'].unique():
    subj_data = labeled[labeled['subject'] == subj]
    other_data = labeled[labeled['subject'] != subj]
    
    X_imu = subj_data[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_ppg = subj_data[ppg_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = subj_data['borg'].values
    
    # ---- WITHIN-PATIENT (random split) ----
    X_train_imu, X_test_imu, y_train, y_test = train_test_split(X_imu, y, test_size=0.3, random_state=42)
    X_train_ppg, X_test_ppg, _, _ = train_test_split(X_ppg, y, test_size=0.3, random_state=42)
    
    # IMU within
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_imu, y_train)
    pred = rf.predict(X_test_imu)
    within_imu_true.extend(y_test)
    within_imu_pred.extend(pred)
    r, _ = pearsonr(y_test, pred)
    within_imu_r.append(r)
    
    # PPG within
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_ppg, y_train)
    pred = rf.predict(X_test_ppg)
    within_ppg_true.extend(y_test)
    within_ppg_pred.extend(pred)
    r, _ = pearsonr(y_test, pred)
    within_ppg_r.append(r)
    
    # ---- CROSS-PATIENT (LOSO) ----
    X_train_imu_loso = other_data[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_train_ppg_loso = other_data[ppg_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train_loso = other_data['borg'].values
    
    # IMU LOSO
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_imu_loso, y_train_loso)
    pred = rf.predict(X_imu)
    loso_imu_true.extend(y)
    loso_imu_pred.extend(pred)
    r, _ = pearsonr(y, pred)
    loso_imu_r.append(r)
    
    # PPG LOSO
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train_ppg_loso, y_train_loso)
    pred = rf.predict(X_ppg)
    loso_ppg_true.extend(y)
    loso_ppg_pred.extend(pred)
    r, _ = pearsonr(y, pred)
    loso_ppg_r.append(r)

# Convert to arrays
within_imu_true, within_imu_pred = np.array(within_imu_true), np.array(within_imu_pred)
within_ppg_true, within_ppg_pred = np.array(within_ppg_true), np.array(within_ppg_pred)
loso_imu_true, loso_imu_pred = np.array(loso_imu_true), np.array(loso_imu_pred)
loso_ppg_true, loso_ppg_pred = np.array(loso_ppg_true), np.array(loso_ppg_pred)

# ============================================================================
# Create 2x2 Plot
# ============================================================================

print("\nCreating 2x2 scatter plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Color scheme - all bluish
color_imu = '#4169E1'  # Royal blue
color_ppg = '#5F9EA0'  # Cadet blue

# Helper function
def plot_scatter(ax, y_true, y_pred, mean_r, title, color):
    ax.scatter(y_true, y_pred, alpha=0.4, s=25, c=color, edgecolor='none')
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2, label='Perfect')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot([0, 7], [p(0), p(7)], color=color, linewidth=2.5, alpha=0.9)
    
    mae = np.mean(np.abs(y_true - y_pred))
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.grid(True, alpha=0.3)
    
    # Stats box
    ax.text(0.95, 0.05, f'r = {mean_r:.2f}\nMAE = {mae:.2f}', 
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Top row: Within-patient (OVERFIT)
plot_scatter(axes[0, 0], within_imu_true, within_imu_pred, 
             np.mean(within_imu_r), 
             'Within-Patient IMU (OVERFIT - random split)', color_imu)

plot_scatter(axes[0, 1], within_ppg_true, within_ppg_pred,
             np.mean(within_ppg_r),
             'Within-Patient PPG (OVERFIT - random split)', color_ppg)

# Bottom row: Cross-patient (LOSO)
plot_scatter(axes[1, 0], loso_imu_true, loso_imu_pred,
             np.mean(loso_imu_r),
             'Cross-Patient IMU (LOSO - honest)', color_imu)

plot_scatter(axes[1, 1], loso_ppg_true, loso_ppg_pred,
             np.mean(loso_ppg_r),
             'Cross-Patient PPG (LOSO - honest)', color_ppg)

# Add row labels
fig.text(0.02, 0.75, 'Within\nPatient', ha='center', va='center', fontsize=12, 
         fontweight='bold', rotation=90)
fig.text(0.02, 0.25, 'Cross\nPatient', ha='center', va='center', fontsize=12,
         fontweight='bold', rotation=90)

plt.tight_layout()
plt.subplots_adjust(left=0.08)

out_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / '61_scatter_2x2_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / '61_scatter_2x2_comparison.png'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
                    IMU         PPG
Within-Patient:   r = {np.mean(within_imu_r):.2f}    r = {np.mean(within_ppg_r):.2f}   (OVERFIT!)
Cross-Patient:    r = {np.mean(loso_imu_r):.2f}    r = {np.mean(loso_ppg_r):.2f}   (honest LOSO)

Note: Within-patient uses random 70/30 split which causes 
temporal leakage with overlapping windows → inflated metrics!

The LOSO numbers are the honest cross-patient generalization.
""")
