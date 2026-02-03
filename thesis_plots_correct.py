#!/usr/bin/env python3
"""
Thesis Final Plots - Matching Your Pipeline Results
====================================================
IMU: 30 features (acc only, no '_r')  → r = 0.52
PPG: 183 features                      → r = 0.26

Plots:
1. Scatter plot (Actual vs Predicted)
2. Top 10 features for IMU and PPG
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading (same as your pipeline)
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
print(f"  {len(labeled)} labeled windows from 5 subjects")

# Feature selection MATCHING YOUR PIPELINE
imu_cols = [c for c in combined.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in combined.columns if 'ppg_' in c]
eda_cols = [c for c in combined.columns if 'eda_' in c]

print(f"  IMU: {len(imu_cols)} features")
print(f"  PPG: {len(ppg_cols)} features")
print(f"  EDA: {len(eda_cols)} features")

# ============================================================================
# LOSO Evaluation
# ============================================================================

def run_loso(df, feature_cols, name):
    """Run LOSO with proper per-subject r calculation."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    per_subj_r = []
    feature_imp_sum = np.zeros(len(feature_cols))
    n_folds = 0
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        # Get valid columns (present in data)
        valid_cols = [c for c in feature_cols if c in train.columns]
        
        X_train = train[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_train = train['borg'].values
        X_test = test[valid_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_test = test['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Per-subject correlation
        if np.std(y_test) > 0:
            r, _ = pearsonr(y_test, y_pred)
            per_subj_r.append(r)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        feature_imp_sum += rf.feature_importances_
        n_folds += 1
    
    return {
        'true': np.array(all_true),
        'pred': np.array(all_pred),
        'per_subj_r': per_subj_r,
        'mean_r': np.mean(per_subj_r),
        'feature_imp': feature_imp_sum / n_folds,
        'feature_names': [c for c in feature_cols if c in train.columns]
    }

print("\n" + "=" * 60)
print("LOSO RESULTS (matching your table)")
print("=" * 60)

print("\nRunning IMU LOSO...")
imu_results = run_loso(labeled, imu_cols, "IMU")
print(f"  Per-subject r: {[f'{r:.2f}' for r in imu_results['per_subj_r']]}")
print(f"  MEAN r = {imu_results['mean_r']:.2f}")

print("\nRunning PPG LOSO...")
ppg_results = run_loso(labeled, ppg_cols, "PPG")
print(f"  Per-subject r: {[f'{r:.2f}' for r in ppg_results['per_subj_r']]}")
print(f"  MEAN r = {ppg_results['mean_r']:.2f}")

print("\nRunning EDA LOSO...")
eda_results = run_loso(labeled, eda_cols, "EDA")
print(f"  Per-subject r: {[f'{r:.2f}' for r in eda_results['per_subj_r']]}")
print(f"  MEAN r = {eda_results['mean_r']:.2f}")

# ============================================================================
# Create Plots
# ============================================================================

out_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
out_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# PLOT 1: Scatter plots (IMU and PPG)
# --------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Creating scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, results, name, color in [(axes[0], imu_results, 'IMU', 'steelblue'),
                                   (axes[1], ppg_results, 'PPG', 'coral')]:
    y_true, y_pred = results['true'], results['pred']
    mean_r = results['mean_r']
    mae = np.mean(np.abs(y_true - y_pred))
    
    ax.scatter(y_true, y_pred, alpha=0.4, s=25, c=color, edgecolor='none')
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2, label='Perfect prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(y_true), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(f'{name}: Mean r = {mean_r:.2f}, MAE = {mae:.2f}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Show prediction range
    ax.annotate(f'Pred range: {y_pred.min():.1f} - {y_pred.max():.1f}',
                xy=(0.95, 0.05), xycoords='axes fraction', ha='right',
                fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(out_dir / '55_scatter_correct.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '55_scatter_correct.png'}")

# --------------------------------------------------------------------------
# PLOT 2: Feature Importance (IMU and PPG)
# --------------------------------------------------------------------------
print("Creating feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, name, color in [(axes[0], imu_results, 'IMU', 'steelblue'),
                                   (axes[1], ppg_results, 'PPG', 'coral')]:
    imp = results['feature_imp']
    names = results['feature_names']
    
    # Top 10
    idx = np.argsort(imp)[::-1][:10]
    top_imp = imp[idx]
    top_names = [names[i] for i in idx]
    
    # Clean up names
    clean_names = []
    for n in top_names:
        n = n.replace('_', ' ').replace('acc ', 'Acc ').replace('ppg ', 'PPG ')
        if len(n) > 30:
            n = n[:27] + '...'
        clean_names.append(n)
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_imp, color=color, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (RF)', fontsize=12)
    ax.set_title(f'Top 10 {name} Features (n={len(names)})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(out_dir / '56_features_correct.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '56_features_correct.png'}")

# --------------------------------------------------------------------------
# PLOT 3: Per-subject bar chart (matching your existing plot)
# --------------------------------------------------------------------------
print("Creating per-subject bar chart...")

fig, ax = plt.subplots(figsize=(10, 6))

subjects = ['P1', 'P2', 'P3', 'P4', 'P5']
x = np.arange(len(subjects))
width = 0.25

bars1 = ax.bar(x - width, imu_results['per_subj_r'], width, label=f"IMU (n={len(imu_cols)})", color='steelblue')
bars2 = ax.bar(x, ppg_results['per_subj_r'], width, label=f"PPG (n={len(ppg_cols)})", color='coral')
bars3 = ax.bar(x + width, eda_results['per_subj_r'], width, label=f"EDA (n={len(eda_cols)})", color='seagreen')

# Mean lines
ax.axhline(y=imu_results['mean_r'], color='steelblue', linestyle='--', alpha=0.7, label=f"IMU mean={imu_results['mean_r']:.2f}")
ax.axhline(y=ppg_results['mean_r'], color='coral', linestyle='--', alpha=0.7, label=f"PPG mean={ppg_results['mean_r']:.2f}")

ax.set_ylabel('Pearson r', fontsize=12)
ax.set_xlabel('Subject (Leave-One-Out)', fontsize=12)
ax.set_title('Cross-Patient Generalization (LOSO)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.legend(loc='upper right')
ax.set_ylim(-0.2, 0.8)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (v1, v2, v3) in enumerate(zip(imu_results['per_subj_r'], ppg_results['per_subj_r'], eda_results['per_subj_r'])):
    ax.text(i - width, v1 + 0.02, f'{v1:.2f}', ha='center', fontsize=9)
    ax.text(i, v2 + 0.02, f'{v2:.2f}', ha='center', fontsize=9)
    ax.text(i + width, v3 + 0.02, f'{v3:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(out_dir / '57_per_subject_correct.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '57_per_subject_correct.png'}")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE (should match your screenshot)")
print("=" * 60)

print(f"""
┌──────────┬──────┬───────────┬─────────┐
│ Modality │  n   │ All LOSO r│   MAE   │
├──────────┼──────┼───────────┼─────────┤
│ IMU      │  {len(imu_cols):3d} │   {imu_results['mean_r']:.2f}    │  {np.mean(np.abs(imu_results['true'] - imu_results['pred'])):.2f}   │
│ PPG      │  {len(ppg_cols):3d} │   {ppg_results['mean_r']:.2f}    │  {np.mean(np.abs(ppg_results['true'] - ppg_results['pred'])):.2f}   │
│ EDA      │  {len(eda_cols):3d} │   {eda_results['mean_r']:.2f}    │  {np.mean(np.abs(eda_results['true'] - eda_results['pred'])):.2f}   │
└──────────┴──────┴───────────┴─────────┘

Your table shows: IMU=30 (r=0.52), PPG=183 (r=0.26), EDA=47 (r=0.02)
""")
