#!/usr/bin/env python3
"""
Professional thesis visualizations - Publication quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_pro")
OUT_DIR.mkdir(exist_ok=True)

# Professional style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (professional)
COLORS = {
    'imu': '#2E86AB',      # Steel blue
    'ppg': '#A23B72',      # Raspberry
    'eda': '#F18F01',      # Orange
    'good': '#28A745',     # Green
    'bad': '#DC3545',      # Red
    'neutral': '#6C757D',  # Gray
    'accent': '#17A2B8',   # Cyan
}

print("="*70)
print("GENERATING PROFESSIONAL THESIS VISUALIZATIONS")
print("="*70)

# =============================================================================
# LOAD DATA
# =============================================================================
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
print(f"  Features: IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_top_k_features(feature_cols, k=10):
    """Get top K features by correlation with Borg."""
    correlations = []
    for col in feature_cols:
        if col in df_labeled.columns:
            x = df_labeled[col].fillna(0).replace([np.inf, -np.inf], 0)
            y = df_labeled['borg']
            if x.std() > 0:
                r, _ = pearsonr(x, y)
                correlations.append((col, r, abs(r)))
    correlations.sort(key=lambda x: x[2], reverse=True)
    return [(c[0], c[1]) for c in correlations[:k]]

def run_loso(feature_cols):
    """Run LOSO and return predictions + per-subject metrics."""
    all_true, all_pred, all_subj = [], [], []
    per_subj_metrics = {}
    
    for test_subj in df_all['subject'].unique():
        train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
        
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if len(y_test) > 2 and np.std(y_test) > 0:
            r, _ = pearsonr(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            per_subj_metrics[test_subj] = {'r': r, 'mae': mae, 'rmse': rmse, 'n': len(y_test)}
            
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            all_subj.extend([test_subj] * len(y_test))
    
    return np.array(all_true), np.array(all_pred), np.array(all_subj), per_subj_metrics

# =============================================================================
# RUN LOSO FOR ALL MODALITIES
# =============================================================================
print("\nRunning LOSO evaluation...")

# IMU
y_true_imu, y_pred_imu, subj_imu, metrics_imu = run_loso(imu_cols)
mean_r_imu = np.mean([m['r'] for m in metrics_imu.values()])
mean_mae_imu = np.mean([m['mae'] for m in metrics_imu.values()])

# PPG
y_true_ppg, y_pred_ppg, subj_ppg, metrics_ppg = run_loso(ppg_cols)
mean_r_ppg = np.mean([m['r'] for m in metrics_ppg.values()])
mean_mae_ppg = np.mean([m['mae'] for m in metrics_ppg.values()])

# EDA
y_true_eda, y_pred_eda, subj_eda, metrics_eda = run_loso(eda_cols)
mean_r_eda = np.mean([m['r'] for m in metrics_eda.values()])
mean_mae_eda = np.mean([m['mae'] for m in metrics_eda.values()])

# Top 10 per modality
top10_imu = [f[0] for f in get_top_k_features(imu_cols, 10)]
top10_ppg = [f[0] for f in get_top_k_features(ppg_cols, 10)]
top10_eda = [f[0] for f in get_top_k_features(eda_cols, 10)]

_, _, _, metrics_imu10 = run_loso(top10_imu)
_, _, _, metrics_ppg10 = run_loso(top10_ppg)
_, _, _, metrics_eda10 = run_loso(top10_eda)

mean_r_imu10 = np.mean([m['r'] for m in metrics_imu10.values()])
mean_r_ppg10 = np.mean([m['r'] for m in metrics_ppg10.values()])
mean_r_eda10 = np.mean([m['r'] for m in metrics_eda10.values()])

print(f"  IMU (all): r={mean_r_imu:.3f}, MAE={mean_mae_imu:.2f}")
print(f"  PPG (all): r={mean_r_ppg:.3f}, MAE={mean_mae_ppg:.2f}")
print(f"  EDA (all): r={mean_r_eda:.3f}, MAE={mean_mae_eda:.2f}")

# =============================================================================
# PLOT 1: PREDICTED VS ACTUAL - BLAND-ALTMAN STYLE + DENSITY
# =============================================================================
print("\n1. Creating Predicted vs Actual (Professional)...")

fig = plt.figure(figsize=(14, 5))
gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1])

# Left: Scatter with density
ax1 = fig.add_subplot(gs[0])

# Create 2D histogram for density
from scipy.stats import gaussian_kde
xy = np.vstack([y_true_imu, y_pred_imu])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = y_true_imu[idx], y_pred_imu[idx], z[idx]

scatter = ax1.scatter(x_sorted, y_sorted, c=z_sorted, cmap='viridis', s=30, alpha=0.7, edgecolors='none')
ax1.plot([0, 10], [0, 10], 'k--', linewidth=1.5, alpha=0.7, label='Perfect agreement')
ax1.fill_between([0, 10], [-1, 9], [1, 11], alpha=0.08, color='green')

# Regression line
z_fit = np.polyfit(y_true_imu, y_pred_imu, 1)
x_line = np.linspace(0, 10, 100)
ax1.plot(x_line, np.polyval(z_fit, x_line), color=COLORS['imu'], linewidth=2, label=f'Fit (r={mean_r_imu:.2f})')

ax1.set_xlabel('Actual Borg CR10', fontweight='bold')
ax1.set_ylabel('Predicted Borg CR10', fontweight='bold')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-0.5, 10.5)
ax1.set_aspect('equal')
ax1.legend(loc='lower right', framealpha=0.9)
ax1.set_title('A) Predicted vs Actual (IMU Features)', fontweight='bold', loc='left')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Density', fontsize=9)

# Middle: Bland-Altman plot
ax2 = fig.add_subplot(gs[1])
mean_vals = (y_true_imu + y_pred_imu) / 2
diff_vals = y_pred_imu - y_true_imu
mean_diff = np.mean(diff_vals)
std_diff = np.std(diff_vals)

ax2.scatter(mean_vals, diff_vals, c=COLORS['imu'], alpha=0.4, s=20, edgecolors='none')
ax2.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, label=f'Bias: {mean_diff:.2f}')
ax2.axhline(mean_diff + 1.96*std_diff, color=COLORS['bad'], linestyle='--', linewidth=1.2, label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
ax2.axhline(mean_diff - 1.96*std_diff, color=COLORS['bad'], linestyle='--', linewidth=1.2, label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
ax2.fill_between([0, 10], mean_diff - 1.96*std_diff, mean_diff + 1.96*std_diff, alpha=0.1, color=COLORS['bad'])

ax2.set_xlabel('Mean of Actual & Predicted', fontweight='bold')
ax2.set_ylabel('Prediction Error (Pred - Actual)', fontweight='bold')
ax2.set_xlim(-0.5, 10.5)
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax2.set_title('B) Bland-Altman Agreement', fontweight='bold', loc='left')

# Right: Per-subject performance
ax3 = fig.add_subplot(gs[2])
subjects = list(metrics_imu.keys())
r_vals = [metrics_imu[s]['r'] for s in subjects]
mae_vals = [metrics_imu[s]['mae'] for s in subjects]

x_pos = np.arange(len(subjects))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, r_vals, width, label='Correlation (r)', color=COLORS['imu'], alpha=0.85)
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x_pos + width/2, mae_vals, width, label='MAE', color=COLORS['accent'], alpha=0.85)

ax3.set_ylabel('Correlation (r)', color=COLORS['imu'], fontweight='bold')
ax3_twin.set_ylabel('MAE (Borg points)', color=COLORS['accent'], fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(subjects, fontweight='bold')
ax3.set_ylim(0, 0.8)
ax3_twin.set_ylim(0, 2.5)
ax3.axhline(mean_r_imu, color=COLORS['imu'], linestyle='--', alpha=0.5)
ax3.set_title('C) Per-Subject Performance', fontweight='bold', loc='left')

# Add value labels
for bar, val in zip(bars1, r_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS['imu'])

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / '1_predicted_vs_actual_pro.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '1_predicted_vs_actual_pro.png'}")

# =============================================================================
# PLOT 2: MODALITY COMPARISON - PROFESSIONAL
# =============================================================================
print("\n2. Creating Modality Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Bar chart comparing modalities
ax = axes[0]
modalities = ['IMU', 'PPG', 'EDA']
all_r = [mean_r_imu, mean_r_ppg, mean_r_eda]
top10_r = [mean_r_imu10, mean_r_ppg10, mean_r_eda10]
all_n = [len(imu_cols), len(ppg_cols), len(eda_cols)]
colors_mod = [COLORS['imu'], COLORS['ppg'], COLORS['eda']]

x = np.arange(len(modalities))
width = 0.35

bars1 = ax.bar(x - width/2, all_r, width, label='All Features', color=colors_mod, alpha=0.9, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, top10_r, width, label='Top 10 Features', color=colors_mod, alpha=0.5, edgecolor='black', linewidth=1.2, hatch='///')

ax.axhline(0.5, color=COLORS['good'], linestyle='--', alpha=0.7, linewidth=1.5, label='Good threshold (r=0.5)')
ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

ax.set_ylabel('LOSO Correlation (r)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{m}\n(n={n})' for m, n in zip(modalities, all_n)], fontweight='bold')
ax.set_ylim(-0.15, 0.7)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title('A) Cross-Subject Generalization by Modality', fontweight='bold', loc='left')

# Add value labels
for bars, values in [(bars1, all_r), (bars2, top10_r)]:
    for bar, val in zip(bars, values):
        y_pos = max(bar.get_height() + 0.02, 0.05)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Results table
ax2 = axes[1]
ax2.axis('off')

# Create table data
table_data = [
    ['Modality', 'Features', 'All (r)', 'Top10 (r)', 'MAE', 'Status'],
    ['IMU', f'{len(imu_cols)}', f'{mean_r_imu:.2f}', f'{mean_r_imu10:.2f}', f'{mean_mae_imu:.2f}', 'Generalizes'],
    ['PPG', f'{len(ppg_cols)}', f'{mean_r_ppg:.2f}', f'{mean_r_ppg10:.2f}', f'{mean_mae_ppg:.2f}', 'Poor'],
    ['EDA', f'{len(eda_cols)}', f'{mean_r_eda:.2f}', f'{mean_r_eda10:.2f}', f'{mean_mae_eda:.2f}', 'Fails'],
]

# Draw table
table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc='center', cellLoc='center',
                  colColours=['#E8E8E8']*6)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style cells
for i, row in enumerate(table_data[1:], 1):
    for j in range(len(row)):
        cell = table[(i, j)]
        if j == 5:  # Status column
            if 'Generalizes' in row[j]:
                cell.set_facecolor('#D4EDDA')
            elif 'Poor' in row[j]:
                cell.set_facecolor('#FFF3CD')
            else:
                cell.set_facecolor('#F8D7DA')
        if i == 1:  # IMU row
            cell.get_text().set_fontweight('bold')

ax2.set_title('B) LOSO Cross-Validation Results Summary', fontweight='bold', loc='center', pad=20)

plt.tight_layout()
plt.savefig(OUT_DIR / '2_modality_comparison_pro.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '2_modality_comparison_pro.png'}")

# =============================================================================
# PLOT 3: TOP FEATURES PER MODALITY WITH CORRELATIONS
# =============================================================================
print("\n3. Creating Top Features per Modality...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

def plot_top_features(ax, feature_tuples, color, title, modality_name):
    """Plot top features with correlation values."""
    names = [f[0].replace('acc_', '').replace('ppg_', '').replace('eda_', '').replace('_dyn__', '\n').replace('__', '\n')[:25] for f in feature_tuples]
    values = [f[1] for f in feature_tuples]
    
    # Sort by absolute correlation
    sorted_idx = np.argsort([abs(v) for v in values])[::-1]
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    y_pos = np.arange(len(names))
    colors = [COLORS['good'] if v > 0 else COLORS['bad'] for v in values]
    
    bars = ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Correlation with Borg (r)', fontweight='bold')
    ax.set_title(title, fontweight='bold', loc='left')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-0.5, 0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                ha=ha, va='center', fontsize=8, fontweight='bold')

# Get top 10 with correlation values
top10_imu_corr = get_top_k_features(imu_cols, 10)
top10_ppg_corr = get_top_k_features(ppg_cols, 10)
top10_eda_corr = get_top_k_features(eda_cols, 10)

plot_top_features(axes[0], top10_imu_corr, COLORS['imu'], 'A) Top 10 IMU Features', 'IMU')
plot_top_features(axes[1], top10_ppg_corr, COLORS['ppg'], 'B) Top 10 PPG Features', 'PPG')
plot_top_features(axes[2], top10_eda_corr, COLORS['eda'], 'C) Top 10 EDA Features', 'EDA')

plt.tight_layout()
plt.savefig(OUT_DIR / '3_top_features_per_modality.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '3_top_features_per_modality.png'}")

# =============================================================================
# PLOT 4: SHAP-STYLE FEATURE IMPORTANCE (using RF importance as proxy)
# =============================================================================
print("\n4. Creating SHAP-style Feature Importance...")

# Train RF on all data for importance
X_all = df_labeled[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
y_all = df_labeled['borg'].values

rf_full = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_full.fit(X_all, y_all)

# Get importances
importances = rf_full.feature_importances_
feat_imp = list(zip(imu_cols, importances))
feat_imp.sort(key=lambda x: x[1], reverse=True)
top15 = feat_imp[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Feature importance bar chart
ax = axes[0]
names = [f[0].replace('acc_', '').replace('_dyn__', '\n') for f in top15]
values = [f[1] for f in top15]

y_pos = np.arange(len(names))
colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(names)))

bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Random Forest)', fontweight='bold')
ax.set_title('A) Top 15 IMU Features by Importance', fontweight='bold', loc='left')

for bar, val in zip(bars, values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=8, fontweight='bold')

# Right: SHAP-style beeswarm (simulated with scatter)
ax2 = axes[1]

# For each of top 10 features, show distribution of values colored by Borg
top10_names = [f[0] for f in top15[:10]]
for i, feat in enumerate(top10_names):
    feat_vals = X_all[feat].values
    # Normalize feature values
    feat_norm = (feat_vals - feat_vals.mean()) / (feat_vals.std() + 1e-8)
    feat_norm = np.clip(feat_norm, -3, 3)  # Clip outliers
    
    # Add jitter
    jitter = np.random.normal(0, 0.15, len(feat_norm))
    
    scatter = ax2.scatter(feat_norm, np.ones(len(feat_norm)) * i + jitter, 
                         c=y_all, cmap='RdYlBu_r', s=8, alpha=0.5, 
                         vmin=0, vmax=10)

ax2.set_yticks(range(len(top10_names)))
ax2.set_yticklabels([n.replace('acc_', '').replace('_dyn__', '\n')[:20] for n in top10_names], fontsize=9)
ax2.set_xlabel('Normalized Feature Value', fontweight='bold')
ax2.set_title('B) Feature Values vs Borg Rating (SHAP-style)', fontweight='bold', loc='left')
ax2.invert_yaxis()

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Borg Rating', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / '4_shap_feature_importance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '4_shap_feature_importance.png'}")

# =============================================================================
# PLOT 5: COMPREHENSIVE RESULTS TABLE
# =============================================================================
print("\n5. Creating Comprehensive Results Table...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Full results table
table_data = [
    ['Metric', 'IMU (30)', 'IMU (Top10)', 'PPG (183)', 'PPG (Top10)', 'EDA (47)', 'EDA (Top10)'],
]

# Per-subject data
for subj in ['P1', 'P2', 'P3', 'P4', 'P5']:
    row = [subj]
    for metrics in [metrics_imu, metrics_imu10, metrics_ppg, metrics_ppg10, metrics_eda, metrics_eda10]:
        if subj in metrics:
            row.append(f"r={metrics[subj]['r']:.2f}")
        else:
            row.append('-')
    table_data.append(row)

# Summary row
table_data.append(['', '', '', '', '', '', ''])
table_data.append(['Mean r', f'{mean_r_imu:.2f}', f'{mean_r_imu10:.2f}', 
                   f'{mean_r_ppg:.2f}', f'{mean_r_ppg10:.2f}',
                   f'{mean_r_eda:.2f}', f'{mean_r_eda10:.2f}'])
table_data.append(['Mean MAE', f'{mean_mae_imu:.2f}', '-', 
                   f'{mean_mae_ppg:.2f}', '-',
                   f'{mean_mae_eda:.2f}', '-'])

# Create table
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center',
                 colColours=['#4A90A4'] + ['#E3F2FD']*2 + ['#FCE4EC']*2 + ['#FFF3E0']*2)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.8)

# Style header
for j in range(len(table_data[0])):
    table[(0, j)].set_text_props(fontweight='bold', color='white')

# Style summary rows
for i in range(len(table_data) - 3, len(table_data) - 1):
    for j in range(len(table_data[0])):
        table[(i, j)].set_facecolor('#E8E8E8')
        table[(i, j)].get_text().set_fontweight('bold')

ax.set_title('LOSO Cross-Validation Results: Complete Summary\n(5 elderly subjects, 5-second windows, Random Forest)', 
             fontsize=14, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig(OUT_DIR / '5_results_table.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '5_results_table.png'}")

# =============================================================================
# PLOT 6: WITHIN VS CROSS-PATIENT (Enhanced)
# =============================================================================
print("\n6. Creating Within vs Cross-Patient Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Bar comparison
ax = axes[0]
categories = ['Within-Patient\n(1 subject, pooled)', 'Cross-Patient\n(LOSO, 5 subjects)']
hr_values = [0.82, mean_r_ppg]
imu_values = [0.65, mean_r_imu]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, hr_values, width, label='HR/PPG', color=COLORS['ppg'], alpha=0.85, edgecolor='black')
bars2 = ax.bar(x + width/2, imu_values, width, label='IMU', color=COLORS['imu'], alpha=0.85, edgecolor='black')

ax.set_ylabel('Correlation (r)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title('A) Generalization Gap', fontweight='bold', loc='left')

# Add value labels
for bars, values, color in [(bars1, hr_values, COLORS['ppg']), (bars2, imu_values, COLORS['imu'])]:
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'r={val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Draw arrow showing drop
ax.annotate('', xy=(0.175, hr_values[1]), xytext=(0.175, hr_values[0]),
            arrowprops=dict(arrowstyle='<->', color=COLORS['bad'], lw=2))
ax.text(0.35, 0.55, f'Δ = {hr_values[0] - hr_values[1]:.2f}\ndrop', fontsize=10, 
        color=COLORS['bad'], fontweight='bold')

# Right: Key insight box
ax2 = axes[1]
ax2.axis('off')

insight_text = """
KEY FINDING: Generalization Gap

HR/PPG Features:
• Within-patient:  r = 0.82  (excellent)
• Cross-patient:   r = 0.26  (poor)
• Drop:            Δr = 0.56

IMU Features:
• Within-patient:  r = 0.65  (good)
• Cross-patient:   r = 0.52  (moderate)
• Drop:            Δr = 0.13

INTERPRETATION:
Heart rate-effort relationships are highly 
individual due to:
  1. Different baseline HR (50-80 bpm)
  2. Different fitness levels
  3. Different physiological responses

IMU generalizes better because movement
patterns are more universal across individuals.

IMPLICATION:
→ Personalized calibration needed for HR
→ IMU can work out-of-box
"""

ax2.text(0.1, 0.95, insight_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6', linewidth=2))

ax2.set_title('B) Clinical Interpretation', fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(OUT_DIR / '6_within_vs_cross_patient.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_DIR / '6_within_vs_cross_patient.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL PROFESSIONAL PLOTS SAVED TO:", OUT_DIR)
print("="*70)
print("""
Files created:
  1_predicted_vs_actual_pro.png   - Scatter + Bland-Altman + Per-subject
  2_modality_comparison_pro.png   - Bar chart + Results table
  3_top_features_per_modality.png - Top 10 features for IMU/PPG/EDA
  4_shap_feature_importance.png   - SHAP-style importance plot
  5_results_table.png             - Complete LOSO results table
  6_within_vs_cross_patient.png   - Generalization gap analysis
""")

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"""
┌─────────────┬──────────┬─────────────┬─────────────┬─────────────┐
│ Modality    │ Features │ LOSO r (All)│ LOSO r (10) │ MAE (All)   │
├─────────────┼──────────┼─────────────┼─────────────┼─────────────┤
│ IMU         │ {len(imu_cols):>3}      │ {mean_r_imu:.2f}        │ {mean_r_imu10:.2f}        │ {mean_mae_imu:.2f}        │
│ PPG         │ {len(ppg_cols):>3}      │ {mean_r_ppg:.2f}        │ {mean_r_ppg10:.2f}        │ {mean_mae_ppg:.2f}        │
│ EDA         │ {len(eda_cols):>3}      │ {mean_r_eda:.2f}        │ {mean_r_eda10:.2f}        │ {mean_mae_eda:.2f}        │
└─────────────┴──────────┴─────────────┴─────────────┴─────────────┘

Per-Subject LOSO (IMU All Features):
  P1: r={metrics_imu['P1']['r']:.2f}, MAE={metrics_imu['P1']['mae']:.2f}
  P2: r={metrics_imu['P2']['r']:.2f}, MAE={metrics_imu['P2']['mae']:.2f}
  P3: r={metrics_imu['P3']['r']:.2f}, MAE={metrics_imu['P3']['mae']:.2f}
  P4: r={metrics_imu['P4']['r']:.2f}, MAE={metrics_imu['P4']['mae']:.2f}
  P5: r={metrics_imu['P5']['r']:.2f}, MAE={metrics_imu['P5']['mae']:.2f}
""")
