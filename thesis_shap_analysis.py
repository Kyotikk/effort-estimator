#!/usr/bin/env python3
"""
SHAP Analysis for Thesis - Professional Feature Importance Visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_pro")
OUT_DIR.mkdir(exist_ok=True)

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("="*70)
print("SHAP ANALYSIS FOR THESIS")
print("="*70)

# Load data
print("\nLoading data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'P{i}'
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_labeled = df_all.dropna(subset=['borg'])

# Define modality columns
imu_cols = [c for c in df_all.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df_all.columns if 'ppg_' in c]
eda_cols = [c for c in df_all.columns if 'eda_' in c]

print(f"  Loaded {len(df_labeled)} labeled windows from 5 subjects")

# =============================================================================
# TRAIN MODEL FOR SHAP
# =============================================================================
print("\nTraining model for SHAP analysis (IMU features)...")

X_imu = df_labeled[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
y = df_labeled['borg'].values

# Clean column names for display
clean_names = {c: c.replace('acc_', '').replace('_dyn__', ': ').replace('__', ' ') for c in imu_cols}
X_imu.columns = [clean_names[c] for c in X_imu.columns]

rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_imu, y)

# SHAP
print("Computing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_imu)

# =============================================================================
# SHAP SUMMARY PLOT (BEESWARM)
# =============================================================================
print("\nCreating SHAP Summary Plot...")

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_imu, max_display=15, show=False, plot_size=(10, 8))
plt.title('SHAP Feature Importance (IMU Features)\nImpact on Borg Effort Prediction', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '7_shap_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '7_shap_summary.png'}")

# =============================================================================
# SHAP BAR PLOT (MEAN |SHAP|)
# =============================================================================
print("\nCreating SHAP Bar Plot...")

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_imu, plot_type="bar", max_display=15, show=False)
plt.title('Mean |SHAP Value| - Feature Importance\n(Average Impact on Model Output)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '8_shap_bar.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '8_shap_bar.png'}")

# =============================================================================
# SHAP FOR PPG (to show it's different)
# =============================================================================
print("\nSHAP for PPG features (comparison)...")

X_ppg = df_labeled[ppg_cols].fillna(0).replace([np.inf, -np.inf], 0)
clean_ppg = {c: c.replace('ppg_', '').replace('__', ' ')[:35] for c in ppg_cols}
X_ppg.columns = [clean_ppg[c] for c in X_ppg.columns]

rf_ppg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_ppg.fit(X_ppg, y)

explainer_ppg = shap.TreeExplainer(rf_ppg)
shap_values_ppg = explainer_ppg.shap_values(X_ppg)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values_ppg, X_ppg, plot_type="bar", max_display=15, show=False)
plt.title('Mean |SHAP Value| - PPG Feature Importance\n(Note: High importance does NOT mean cross-subject generalization)', 
          fontsize=12, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '9_shap_ppg_bar.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '9_shap_ppg_bar.png'}")

# =============================================================================
# COMBINED SHAP COMPARISON
# =============================================================================
print("\nCreating Combined SHAP Comparison...")

# Get mean absolute SHAP values for each modality
mean_shap_imu = np.abs(shap_values).mean(axis=0)
mean_shap_ppg = np.abs(shap_values_ppg).mean(axis=0)

# Top 10 for each
top10_imu_idx = np.argsort(mean_shap_imu)[::-1][:10]
top10_ppg_idx = np.argsort(mean_shap_ppg)[::-1][:10]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# IMU
ax1 = axes[0]
names_imu = [X_imu.columns[i][:25] for i in top10_imu_idx]
vals_imu = [mean_shap_imu[i] for i in top10_imu_idx]
colors_imu = plt.cm.Blues(np.linspace(0.8, 0.4, 10))
bars1 = ax1.barh(range(10), vals_imu, color=colors_imu, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(10))
ax1.set_yticklabels(names_imu, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Mean |SHAP Value|', fontweight='bold')
ax1.set_title('A) IMU Features (LOSO r=0.52)\n✓ Generalizes across subjects', fontweight='bold', loc='left', color='#28A745')
for bar, val in zip(bars1, vals_imu):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)

# PPG
ax2 = axes[1]
names_ppg = [X_ppg.columns[i][:25] for i in top10_ppg_idx]
vals_ppg = [mean_shap_ppg[i] for i in top10_ppg_idx]
colors_ppg = plt.cm.Reds(np.linspace(0.8, 0.4, 10))
bars2 = ax2.barh(range(10), vals_ppg, color=colors_ppg, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(10))
ax2.set_yticklabels(names_ppg, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Mean |SHAP Value|', fontweight='bold')
ax2.set_title('B) PPG Features (LOSO r=0.26)\n✗ Poor cross-subject generalization', fontweight='bold', loc='left', color='#DC3545')
for bar, val in zip(bars2, vals_ppg):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)

plt.suptitle('SHAP Feature Importance Comparison\nHigh Pooled Importance ≠ Cross-Subject Generalization', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / '10_shap_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '10_shap_comparison.png'}")

# =============================================================================
# SINGLE FEATURE DEPENDENCE PLOT
# =============================================================================
print("\nCreating SHAP Dependence Plot...")

top_feature_idx = top10_imu_idx[0]
top_feature_name = X_imu.columns[top_feature_idx]

fig, ax = plt.subplots(figsize=(8, 6))
shap.dependence_plot(top_feature_idx, shap_values, X_imu, show=False, ax=ax)
ax.set_title(f'SHAP Dependence: {top_feature_name}\nRelationship with Borg Prediction', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '11_shap_dependence.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '11_shap_dependence.png'}")

print("\n" + "="*70)
print("SHAP ANALYSIS COMPLETE!")
print("="*70)
print(f"""
Additional SHAP plots saved:
  7_shap_summary.png      - Beeswarm plot (IMU)
  8_shap_bar.png          - Bar chart mean |SHAP| (IMU)
  9_shap_ppg_bar.png      - Bar chart mean |SHAP| (PPG)
  10_shap_comparison.png  - IMU vs PPG comparison
  11_shap_dependence.png  - Dependence plot for top feature
""")
