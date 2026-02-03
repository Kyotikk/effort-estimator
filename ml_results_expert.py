#!/usr/bin/env python3
"""
ML Expert-Level Results Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# Load data
print("Loading data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        tmp = pd.read_csv(path)
        tmp['subject'] = f'P{i}'
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
print(f"Data: {len(df)} windows, {df['subject'].nunique()} subjects, {len(imu_cols)} IMU features")

# Run LOSO and collect all predictions
subjects = sorted(df['subject'].unique())
all_results = []

for test_subj in subjects:
    train = df[df['subject'] != test_subj]
    test = df[df['subject'] == test_subj].copy()
    
    X_train = train[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train['borg'].values
    X_test = test[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    test['y_pred'] = y_pred
    test['y_true'] = y_test
    test['error'] = y_pred - y_test
    all_results.append(test)

results = pd.concat(all_results, ignore_index=True)

# Calculate metrics
per_subj = {}
for subj in subjects:
    mask = results['subject'] == subj
    y_t = results.loc[mask, 'y_true'].values
    y_p = results.loc[mask, 'y_pred'].values
    r, _ = pearsonr(y_t, y_p)
    mae = np.mean(np.abs(y_t - y_p))
    per_subj[subj] = {'r': r, 'mae': mae, 'n': len(y_t)}

mean_r = np.mean([v['r'] for v in per_subj.values()])
mean_mae = np.mean([v['mae'] for v in per_subj.values()])

print(f"\nLOSO Results: Mean r = {mean_r:.2f}, Mean MAE = {mean_mae:.2f}")

# =============================================================================
# CREATE FIGURE: Clean 2x2 visualization
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Consistent colors
COLOR_ACTUAL = '#2c3e50'    # Dark blue-gray for actual
COLOR_PRED = '#e74c3c'      # Red for predicted
COLOR_BAR = '#3498db'       # Blue for bars

# -----------------------------------------------------------------------------
# Plot A: Per-Subject Performance (Bar Chart)
# -----------------------------------------------------------------------------
ax = axes[0, 0]
x = np.arange(len(subjects))
rs = [per_subj[s]['r'] for s in subjects]

bars = ax.bar(x, rs, color=COLOR_BAR, edgecolor='black', linewidth=1.5, width=0.6)
ax.axhline(y=mean_r, color=COLOR_ACTUAL, linestyle='--', linewidth=2, label=f'Mean r = {mean_r:.2f}')

for bar, r in zip(bars, rs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{r:.2f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Subject (Leave-One-Out)', fontsize=11)
ax.set_ylabel('Pearson Correlation (r)', fontsize=11)
ax.set_title('A) Cross-Patient Generalization (LOSO)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=11)
ax.set_ylim(0, 0.75)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# -----------------------------------------------------------------------------
# Plot B: Best Subject Time Series
# -----------------------------------------------------------------------------
ax = axes[0, 1]
best_subj = max(per_subj.keys(), key=lambda x: per_subj[x]['r'])
subj_data = results[results['subject'] == best_subj].sort_values('t_center')
t = np.arange(len(subj_data))

ax.plot(t, subj_data['y_true'].values, color=COLOR_ACTUAL, linewidth=2, label='Actual', alpha=0.9)
ax.plot(t, subj_data['y_pred'].values, color=COLOR_PRED, linewidth=2, label='Predicted', linestyle='--', alpha=0.9)

ax.set_xlabel('Time (window index)', fontsize=11)
ax.set_ylabel('Borg CR10', fontsize=11)
ax.set_title(f'B) Prediction Over Time ({best_subj}, r={per_subj[best_subj]["r"]:.2f})', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(-0.5, 7)
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot C: Scatter - Predicted vs Actual (colored by subject with trend lines)
# -----------------------------------------------------------------------------
ax = axes[1, 0]

# Color palette for subjects
subj_colors = {'P1': '#e74c3c', 'P2': '#3498db', 'P3': '#2ecc71', 'P4': '#9b59b6', 'P5': '#f39c12'}

# Plot per subject with different colors AND trend lines
for subj in subjects:
    mask = results['subject'] == subj
    x_data = results.loc[mask, 'y_true'].values
    y_data = results.loc[mask, 'y_pred'].values
    r_subj = per_subj[subj]['r']
    
    # Scatter
    ax.scatter(x_data, y_data, alpha=0.4, s=20, color=subj_colors[subj], edgecolors='none')
    
    # Add trend line for each subject
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_line = np.array([x_data.min(), x_data.max()])
    ax.plot(x_line, p(x_line), color=subj_colors[subj], linewidth=2.5, 
            label=f'{subj} (r={r_subj:.2f})', alpha=0.9)

# Add diagonal line (perfect prediction)
ax.plot([0, 7], [0, 7], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')

ax.set_xlabel('Actual Borg CR10', fontsize=11)
ax.set_ylabel('Predicted Borg CR10', fontsize=11)
ax.set_title('C) Predicted vs Actual (with Trend Lines)', fontsize=12, fontweight='bold')
ax.set_xlim(-0.5, 7)
ax.set_ylim(-0.5, 7)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot D: Error Distribution
# -----------------------------------------------------------------------------
ax = axes[1, 1]

errors = results['y_pred'] - results['y_true']
ax.hist(errors, bins=30, color=COLOR_BAR, edgecolor='black', alpha=0.8)

# Add vertical lines for mean and std
mean_err = errors.mean()
std_err = errors.std()
ax.axvline(x=mean_err, color=COLOR_PRED, linewidth=2, linestyle='-', label=f'Mean = {mean_err:.2f}')
ax.axvline(x=mean_err - std_err, color=COLOR_ACTUAL, linewidth=1.5, linestyle='--', label=f'±1 SD = {std_err:.2f}')
ax.axvline(x=mean_err + std_err, color=COLOR_ACTUAL, linewidth=1.5, linestyle='--')
ax.axvline(x=0, color='black', linewidth=1, linestyle=':', alpha=0.5)

ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('D) Error Distribution (Near Zero Bias)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# -----------------------------------------------------------------------------
# Final touches
# -----------------------------------------------------------------------------
plt.tight_layout(pad=2.0)
plt.savefig(f"{OUT}/30_ml_results_expert.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT}/30_ml_results_expert.png")
plt.close()

# =============================================================================
# BONUS: All 5 subjects time series - SAME COLOR SCHEME
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, subj in enumerate(subjects):
    ax = axes[idx]
    subj_data = results[results['subject'] == subj].sort_values('t_center')
    t = np.arange(len(subj_data))
    
    ax.plot(t, subj_data['y_true'].values, color=COLOR_ACTUAL, linewidth=1.5, label='Actual', alpha=0.9)
    ax.plot(t, subj_data['y_pred'].values, color=COLOR_PRED, linewidth=1.5, label='Predicted', linestyle='--', alpha=0.8)
    
    r = per_subj[subj]['r']
    mae = per_subj[subj]['mae']
    ax.set_title(f'{subj}: r = {r:.2f}', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.5, 7)
    ax.grid(True, alpha=0.3)
    if idx >= 3:
        ax.set_xlabel('Time (window index)', fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel('Borg CR10', fontsize=10)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

axes[5].axis('off')

fig.suptitle('IMU-Based Effort Prediction: All 5 Subjects (LOSO Cross-Validation)', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/31_all_subjects_timeseries.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/31_all_subjects_timeseries.png")
plt.close()

print("\nDone!")
