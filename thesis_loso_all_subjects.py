#!/usr/bin/env python3
"""
LOSO visualization for all 5 subjects using actual pipeline data.
Shows actual vs predicted Borg with activity boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import glob
from pathlib import Path

# ============================================
# Load actual pipeline data (fused_aligned with labels)
# ============================================
base = Path('/Users/pascalschlegel/data/interim')

# Find all elderly subjects' fused_aligned files
patterns = [
    'parsingsim*/sim_elderly*/effort_estimation_output/*/fused_aligned_5.0s.csv'
]

all_dfs = []
for pattern in patterns:
    files = sorted(glob.glob(str(base / pattern)))
    for f in files:
        df = pd.read_csv(f)
        # Extract subject name from path
        parts = Path(f).parts
        for p in parts:
            if 'elderly' in p and 'sim_' in p:
                df['subject'] = p
                break
        all_dfs.append(df)
        print(f"Loaded {f}: {len(df)} rows, {df['borg'].notna().sum()} labeled")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(df_all)} rows, {df_all['borg'].notna().sum()} labeled")
print(f"Subjects: {sorted(df_all['subject'].unique())}")

# ============================================
# Select IMU features (acc_* without _r suffix)
# ============================================
imu_cols = [c for c in df_all.columns if c.startswith('acc_') and not c.endswith('_r')]
print(f"\nIMU features: {len(imu_cols)}")

# Get labeled data only
df_labeled = df_all.dropna(subset=['borg']).copy()
df_labeled = df_labeled.dropna(subset=imu_cols)  # Need all features
print(f"Labeled with complete IMU: {len(df_labeled)}")

subjects = sorted(df_labeled['subject'].unique())
print(f"Subjects with data: {subjects}")

# ============================================
# LOSO Cross-Validation
# ============================================
all_results = []
per_subject_r = {}

for test_subj in subjects:
    train = df_labeled[df_labeled['subject'] != test_subj]
    test = df_labeled[df_labeled['subject'] == test_subj].copy()
    
    if len(test) < 5:
        print(f"  {test_subj}: skipping (only {len(test)} samples)")
        continue
    
    X_train = train[imu_cols].values
    y_train = train['borg'].values
    X_test = test[imu_cols].values
    
    # Random Forest matching your pipeline
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    test['predicted'] = rf.predict(X_test)
    
    r, _ = pearsonr(test['borg'], test['predicted'])
    per_subject_r[test_subj] = r
    print(f"  {test_subj}: n={len(test)}, r={r:.3f}")
    
    all_results.append(test)

results = pd.concat(all_results, ignore_index=True)
overall_r, _ = pearsonr(results['borg'], results['predicted'])
mean_r = np.mean(list(per_subject_r.values()))
print(f"\nOverall pooled r: {overall_r:.3f}")
print(f"Mean per-subject r: {mean_r:.3f}")

# ============================================
# Create visualization (one subplot per subject)
# ============================================
n_subjects = len(subjects)
fig, axes = plt.subplots(n_subjects, 1, figsize=(14, 4*n_subjects))
if n_subjects == 1:
    axes = [axes]

fig.suptitle(f'LOSO Predictions: Actual vs Predicted Borg (IMU, {len(imu_cols)} features)\nMean r = {mean_r:.2f}', 
             fontsize=14, fontweight='bold')

for idx, subj in enumerate(subjects):
    ax = axes[idx]
    subj_data = results[results['subject'] == subj].sort_values('t_center').reset_index(drop=True)
    
    if len(subj_data) == 0:
        ax.text(0.5, 0.5, f'{subj}: No data', ha='center', va='center', transform=ax.transAxes)
        continue
    
    r = per_subject_r.get(subj, np.nan)
    
    x = range(len(subj_data))
    
    # Plot actual and predicted
    ax.plot(x, subj_data['borg'], 'k-', linewidth=2.5, label='Actual Borg', 
            marker='o', markersize=5, zorder=3)
    ax.plot(x, subj_data['predicted'], 'steelblue', linewidth=1.5, label='Predicted', 
            marker='s', markersize=4, alpha=0.8, zorder=2)
    
    # Fill between
    ax.fill_between(x, subj_data['borg'], subj_data['predicted'], 
                    alpha=0.2, color='steelblue', zorder=1)
    
    # Add activity labels if available (based on Borg level changes)
    borg_vals = subj_data['borg'].values
    prev_borg = None
    start_idx = 0
    activity_num = 1
    colors = plt.cm.Pastel1(np.linspace(0, 1, 20))
    
    for i, b in enumerate(borg_vals):
        if b != prev_borg:
            if prev_borg is not None and i > start_idx:
                color_idx = activity_num % 20
                ax.axvspan(start_idx - 0.5, i - 0.5, alpha=0.2, color=colors[color_idx])
                activity_num += 1
            start_idx = i
            prev_borg = b
    # Last segment
    if prev_borg is not None:
        color_idx = activity_num % 20
        ax.axvspan(start_idx - 0.5, len(borg_vals) - 0.5, alpha=0.2, color=colors[color_idx])
    
    ax.set_ylabel('Borg CR10', fontsize=10)
    ax.set_ylim(-0.5, 10)
    ax.set_xlim(-1, len(subj_data))
    ax.set_title(f'{subj} (r = {r:.2f}, n = {len(subj_data)})', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

axes[-1].set_xlabel('Window Index (Time →)', fontsize=11)

plt.tight_layout()
out_path = Path('/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/63_loso_all_subjects_pipeline.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {out_path}")

# Also save PDF
plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {out_path.with_suffix('.pdf')}")

plt.show()
