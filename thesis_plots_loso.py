#!/usr/bin/env python3
"""
Clean thesis visualizations - ALL metrics from LOSO, NO pooled data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_final")
OUT_DIR.mkdir(exist_ok=True)

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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'imu': '#2E86AB',
    'ppg': '#A23B72',
    'eda': '#F18F01',
}

print("="*70)
print("THESIS VISUALIZATIONS - LOSO ONLY (NO POOLED)")
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

imu_cols = [c for c in df_all.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df_all.columns if 'ppg_' in c]
eda_cols = [c for c in df_all.columns if 'eda_' in c]

print(f"  {len(df_labeled)} windows, 5 subjects")
print(f"  IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)} features")

# =============================================================================
# LOSO FUNCTION - RETURNS PREDICTIONS + FEATURE IMPORTANCES
# =============================================================================
def run_loso_full(feature_cols):
    """Run LOSO, return predictions and AVERAGED feature importances."""
    all_true, all_pred, all_subj = [], [], []
    per_subj_metrics = {}
    all_importances = []
    
    for test_subj in df_all['subject'].unique():
        train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Store feature importances from this fold
        all_importances.append(dict(zip(valid_cols, rf.feature_importances_)))
        
        if len(y_test) > 2 and np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            per_subj_metrics[test_subj] = {'r': r, 'mae': mae, 'rmse': rmse, 'n': len(y_test)}
            
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            all_subj.extend([test_subj] * len(y_test))
    
    # Average feature importances across folds
    avg_importance = {}
    for col in feature_cols:
        vals = [imp.get(col, 0) for imp in all_importances]
        avg_importance[col] = np.mean(vals)
    
    return np.array(all_true), np.array(all_pred), np.array(all_subj), per_subj_metrics, avg_importance

# =============================================================================
# RUN LOSO
# =============================================================================
print("\nRunning LOSO...")

y_true_imu, y_pred_imu, subj_imu, metrics_imu, importance_imu = run_loso_full(imu_cols)
y_true_ppg, y_pred_ppg, subj_ppg, metrics_ppg, importance_ppg = run_loso_full(ppg_cols)
y_true_eda, y_pred_eda, subj_eda, metrics_eda, importance_eda = run_loso_full(eda_cols)

mean_r_imu = np.mean([m['r'] for m in metrics_imu.values()])
mean_r_ppg = np.mean([m['r'] for m in metrics_ppg.values()])
mean_r_eda = np.mean([m['r'] for m in metrics_eda.values()])

mean_mae_imu = np.mean([m['mae'] for m in metrics_imu.values()])
mean_mae_ppg = np.mean([m['mae'] for m in metrics_ppg.values()])
mean_mae_eda = np.mean([m['mae'] for m in metrics_eda.values()])

mean_rmse_imu = np.mean([m['rmse'] for m in metrics_imu.values()])
mean_rmse_ppg = np.mean([m['rmse'] for m in metrics_ppg.values()])
mean_rmse_eda = np.mean([m['rmse'] for m in metrics_eda.values()])

# Top 10 by LOSO importance
top10_imu = sorted(importance_imu.items(), key=lambda x: x[1], reverse=True)[:10]
top10_ppg = sorted(importance_ppg.items(), key=lambda x: x[1], reverse=True)[:10]
top10_eda = sorted(importance_eda.items(), key=lambda x: x[1], reverse=True)[:10]

# Run LOSO with Top 10 features only
print("  Running Top 10 LOSO...")
_, _, _, metrics_imu10, _ = run_loso_full([f[0] for f in top10_imu])
_, _, _, metrics_ppg10, _ = run_loso_full([f[0] for f in top10_ppg])
_, _, _, metrics_eda10, _ = run_loso_full([f[0] for f in top10_eda])

mean_r_imu10 = np.mean([m['r'] for m in metrics_imu10.values()])
mean_r_ppg10 = np.mean([m['r'] for m in metrics_ppg10.values()])
mean_r_eda10 = np.mean([m['r'] for m in metrics_eda10.values()])

mean_mae_imu10 = np.mean([m['mae'] for m in metrics_imu10.values()])
mean_mae_ppg10 = np.mean([m['mae'] for m in metrics_ppg10.values()])
mean_mae_eda10 = np.mean([m['mae'] for m in metrics_eda10.values()])

# Greedy forward feature selection
print("  Running Greedy Feature Selection...")

def greedy_forward_loso(feature_pool, max_features=10):
    """Greedy forward selection using LOSO r as criterion."""
    selected = []
    best_r_history = []
    remaining = list(feature_pool)
    
    for _ in range(min(max_features, len(remaining))):
        best_r = -999
        best_feat = None
        
        for feat in remaining:
            test_feats = selected + [feat]
            _, _, _, metrics, _ = run_loso_full(test_feats)
            r = np.mean([m['r'] for m in metrics.values()])
            
            if r > best_r:
                best_r = r
                best_feat = feat
        
        if best_feat is None:
            break
        
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_r_history.append(best_r)
    
    return selected, best_r_history

# Greedy on IMU (fastest)
greedy_imu, greedy_r_imu = greedy_forward_loso(imu_cols, max_features=10)
_, _, _, metrics_greedy_imu, _ = run_loso_full(greedy_imu)
mean_r_greedy_imu = np.mean([m['r'] for m in metrics_greedy_imu.values()])
mean_mae_greedy_imu = np.mean([m['mae'] for m in metrics_greedy_imu.values()])

# Greedy on PPG (subset for speed)
ppg_subset = [f[0] for f in sorted(importance_ppg.items(), key=lambda x: x[1], reverse=True)[:30]]
greedy_ppg, greedy_r_ppg = greedy_forward_loso(ppg_subset, max_features=10)
_, _, _, metrics_greedy_ppg, _ = run_loso_full(greedy_ppg)
mean_r_greedy_ppg = np.mean([m['r'] for m in metrics_greedy_ppg.values()])
mean_mae_greedy_ppg = np.mean([m['mae'] for m in metrics_greedy_ppg.values()])

# Greedy on EDA
greedy_eda, greedy_r_eda = greedy_forward_loso(eda_cols, max_features=10)
_, _, _, metrics_greedy_eda, _ = run_loso_full(greedy_eda)
mean_r_greedy_eda = np.mean([m['r'] for m in metrics_greedy_eda.values()])
mean_mae_greedy_eda = np.mean([m['mae'] for m in metrics_greedy_eda.values()])

print(f"  Greedy IMU ({len(greedy_imu)} feats): r={mean_r_greedy_imu:.2f}")
print(f"  Greedy PPG ({len(greedy_ppg)} feats): r={mean_r_greedy_ppg:.2f}")
print(f"  Greedy EDA ({len(greedy_eda)} feats): r={mean_r_greedy_eda:.2f}")

# Within-patient for ALL modalities
print("  Computing within-patient for all modalities...")
from sklearn.model_selection import train_test_split

def compute_within_patient(feature_cols):
    within_r = []
    for subj in df_all['subject'].unique():
        subj_df = df_all[df_all['subject'] == subj].dropna(subset=['borg'])
        if len(subj_df) > 20:
            X = subj_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = subj_df['borg'].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            y_p = rf.predict(X_te)
            if np.std(y_te) > 0 and np.std(y_p) > 0:
                r, _ = pearsonr(y_te, y_p)
                within_r.append(r)
    return np.mean(within_r) if within_r else 0

mean_within_imu = compute_within_patient(imu_cols)
mean_within_ppg = compute_within_patient(ppg_cols)
mean_within_eda = compute_within_patient(eda_cols)

print(f"  Within-patient: IMU={mean_within_imu:.2f}, PPG={mean_within_ppg:.2f}, EDA={mean_within_eda:.2f}")

print(f"  IMU: LOSO r={mean_r_imu:.2f}, MAE={mean_mae_imu:.2f}")
print(f"  PPG: LOSO r={mean_r_ppg:.2f}, MAE={mean_mae_ppg:.2f}")
print(f"  EDA: LOSO r={mean_r_eda:.2f}, MAE={mean_mae_eda:.2f}")

# =============================================================================
# PLOT 1: PREDICTED VS ACTUAL (LOSO)
# =============================================================================
print("\n1. Predicted vs Actual...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter with density
ax1 = axes[0]
from scipy.stats import gaussian_kde
xy = np.vstack([y_true_imu, y_pred_imu])
z = gaussian_kde(xy)(xy)
idx = z.argsort()

scatter = ax1.scatter(y_true_imu[idx], y_pred_imu[idx], c=z[idx], cmap='viridis', s=25, alpha=0.7, edgecolors='none')
ax1.plot([0, 10], [0, 10], 'k--', linewidth=1.5, alpha=0.5)
z_fit = np.polyfit(y_true_imu, y_pred_imu, 1)
ax1.plot(np.linspace(0, 10, 100), np.polyval(z_fit, np.linspace(0, 10, 100)), color=COLORS['imu'], linewidth=2)
ax1.set_xlabel('Actual Borg CR10')
ax1.set_ylabel('Predicted Borg CR10')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-0.5, 10.5)
ax1.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Density')

# Bland-Altman
ax2 = axes[1]
mean_vals = (y_true_imu + y_pred_imu) / 2
diff_vals = y_pred_imu - y_true_imu
mean_diff = np.mean(diff_vals)
std_diff = np.std(diff_vals)

ax2.scatter(mean_vals, diff_vals, c=COLORS['imu'], alpha=0.3, s=15, edgecolors='none')
ax2.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(mean_diff + 1.96*std_diff, color='#DC3545', linestyle='--', linewidth=1.2)
ax2.axhline(mean_diff - 1.96*std_diff, color='#DC3545', linestyle='--', linewidth=1.2)
ax2.set_xlabel('Mean (Actual + Predicted) / 2')
ax2.set_ylabel('Difference (Predicted - Actual)')
ax2.set_xlim(-0.5, 10.5)

plt.tight_layout()
plt.savefig(OUT_DIR / '1_predicted_vs_actual.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 2: PER-SUBJECT PERFORMANCE
# =============================================================================
print("2. Per-Subject Performance...")

fig, ax = plt.subplots(figsize=(8, 5))

subjects = list(metrics_imu.keys())
r_vals = [metrics_imu[s]['r'] for s in subjects]
mae_vals = [metrics_imu[s]['mae'] for s in subjects]

x_pos = np.arange(len(subjects))
width = 0.35

bars1 = ax.bar(x_pos - width/2, r_vals, width, label='Pearson r', color=COLORS['imu'], alpha=0.85)
ax_twin = ax.twinx()
bars2 = ax_twin.bar(x_pos + width/2, mae_vals, width, label='MAE', color='#17A2B8', alpha=0.85)

ax.set_ylabel('Pearson r', color=COLORS['imu'])
ax_twin.set_ylabel('MAE (Borg)', color='#17A2B8')
ax.set_xticks(x_pos)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 0.8)
ax_twin.set_ylim(0, 3.0)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(OUT_DIR / '2_per_subject.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 3: MODALITY COMPARISON (LOSO r)
# =============================================================================
print("3. Modality Comparison...")

fig, ax = plt.subplots(figsize=(7, 5))

modalities = ['IMU', 'PPG', 'EDA']
loso_r = [mean_r_imu, mean_r_ppg, mean_r_eda]
colors_mod = [COLORS['imu'], COLORS['ppg'], COLORS['eda']]

bars = ax.bar(modalities, loso_r, color=colors_mod, alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_ylabel('LOSO Pearson r')
ax.set_ylim(-0.1, 0.65)

for bar, val in zip(bars, loso_r):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '3_modality_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 4: TOP FEATURES BY LOSO IMPORTANCE (NOT POOLED CORRELATION!)
# =============================================================================
print("4. Top Features (LOSO Importance)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

def clean_name(name, prefix):
    name = name.replace(f'{prefix}_', '').replace('_dyn__', ': ').replace('__', ': ')
    if len(name) > 28:
        name = name[:25] + '...'
    return name

def plot_importance(ax, top_features, color, prefix):
    names = [clean_name(f[0], prefix) for f in top_features]
    values = [f[1] for f in top_features]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('LOSO Feature Importance')
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=8)

plot_importance(axes[0], top10_imu, COLORS['imu'], 'acc')
axes[0].set_title('IMU')

plot_importance(axes[1], top10_ppg, COLORS['ppg'], 'ppg')
axes[1].set_title('PPG')

plot_importance(axes[2], top10_eda, COLORS['eda'], 'eda')
axes[2].set_title('EDA')

plt.tight_layout()
plt.savefig(OUT_DIR / '4_top_features_loso.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 5: RESULTS TABLE (WITH TOP10 AND GREEDY)
# =============================================================================
print("5. Results Table...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

table_data = [
    ['Modality', 'n', 'All\nLOSO r', 'All\nMAE', 'Top10\nLOSO r', 'Top10\nMAE', 'Greedy\nLOSO r', 'Greedy\nMAE'],
    ['IMU', f'{len(imu_cols)}', f'{mean_r_imu:.2f}', f'{mean_mae_imu:.2f}', f'{mean_r_imu10:.2f}', f'{mean_mae_imu10:.2f}', f'{mean_r_greedy_imu:.2f}', f'{mean_mae_greedy_imu:.2f}'],
    ['PPG', f'{len(ppg_cols)}', f'{mean_r_ppg:.2f}', f'{mean_mae_ppg:.2f}', f'{mean_r_ppg10:.2f}', f'{mean_mae_ppg10:.2f}', f'{mean_r_greedy_ppg:.2f}', f'{mean_mae_greedy_ppg:.2f}'],
    ['EDA', f'{len(eda_cols)}', f'{mean_r_eda:.2f}', f'{mean_mae_eda:.2f}', f'{mean_r_eda10:.2f}', f'{mean_mae_eda10:.2f}', f'{mean_r_greedy_eda:.2f}', f'{mean_mae_greedy_eda:.2f}'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center', colColours=['#E8E8E8']*8)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

for j in range(8):
    table[(0, j)].get_text().set_fontweight('bold')
for j in range(8):
    table[(1, j)].set_facecolor('#E3F2FD')

plt.tight_layout()
plt.savefig(OUT_DIR / '5_results_table.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 6: PER-SUBJECT TABLE
# =============================================================================
print("6. Per-Subject Table...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

header = ['Subject', 'IMU r', 'IMU MAE', 'PPG r', 'PPG MAE', 'EDA r', 'EDA MAE']
rows = []
for subj in ['P1', 'P2', 'P3', 'P4', 'P5']:
    row = [subj]
    for metrics in [metrics_imu, metrics_ppg, metrics_eda]:
        row.append(f'{metrics[subj]["r"]:.2f}')
        row.append(f'{metrics[subj]["mae"]:.2f}')
    rows.append(row)
rows.append(['Mean', f'{mean_r_imu:.2f}', f'{mean_mae_imu:.2f}', 
             f'{mean_r_ppg:.2f}', f'{mean_mae_ppg:.2f}',
             f'{mean_r_eda:.2f}', f'{mean_mae_eda:.2f}'])

table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center', colColours=['#E8E8E8']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

for j in range(7):
    table[(0, j)].get_text().set_fontweight('bold')
    table[(6, j)].get_text().set_fontweight('bold')
    table[(6, j)].set_facecolor('#D4D4D4')

plt.tight_layout()
plt.savefig(OUT_DIR / '6_per_subject_table.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 7: WITHIN VS CROSS PATIENT (ALL MODALITIES)
# =============================================================================
print("7. Within vs Cross Patient (All Modalities)...")

fig, ax = plt.subplots(figsize=(10, 6))

modalities = ['IMU', 'PPG', 'EDA']
within_vals = [mean_within_imu, mean_within_ppg, mean_within_eda]
cross_vals = [mean_r_imu, mean_r_ppg, mean_r_eda]
colors_mod = [COLORS['imu'], COLORS['ppg'], COLORS['eda']]

x = np.arange(len(modalities))
width = 0.35

bars1 = ax.bar(x - width/2, within_vals, width, label='Within-Patient', color=colors_mod, alpha=0.9, edgecolor='black')
bars2 = ax.bar(x + width/2, cross_vals, width, label='Cross-Patient (LOSO)', color=colors_mod, alpha=0.4, edgecolor='black', hatch='///')

ax.set_ylabel('Pearson r')
ax.set_xticks(x)
ax.set_xticklabels(modalities)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')

for bar, val in zip(bars1, within_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, cross_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '7_within_vs_cross.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 8: PREDICTIONS OVER TIME (SINGLE SUBJECT EXAMPLE)
# =============================================================================
print("8. Predictions Over Time (Example Subject)...")

# Use P4 as example (good performance)
example_subj = 'P4'
mask = subj_imu == example_subj
y_true_ex = y_true_imu[mask]
y_pred_ex = y_pred_imu[mask]

fig, ax = plt.subplots(figsize=(14, 5))

time_idx = np.arange(len(y_true_ex))
ax.plot(time_idx, y_true_ex, 'o-', color='black', alpha=0.7, markersize=4, label='Actual Borg')
ax.plot(time_idx, y_pred_ex, 's-', color=COLORS['imu'], alpha=0.7, markersize=4, label='Predicted Borg')

ax.fill_between(time_idx, y_true_ex, y_pred_ex, alpha=0.2, color=COLORS['imu'])
ax.set_xlabel('Window Index (Time)')
ax.set_ylabel('Borg CR10')
ax.set_ylim(-0.5, 10.5)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUT_DIR / '8_predictions_over_time.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 9: PER-SUBJECT SCATTER GRID
# =============================================================================
print("9. Per-Subject Scatter Grid...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, subj in enumerate(['P1', 'P2', 'P3', 'P4', 'P5']):
    ax = axes[i]
    mask = subj_imu == subj
    y_t = y_true_imu[mask]
    y_p = y_pred_imu[mask]
    
    ax.scatter(y_t, y_p, c=COLORS['imu'], alpha=0.5, s=30, edgecolors='none')
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5)
    
    if len(y_t) > 2:
        r, _ = pearsonr(y_t, y_p)
        mae = np.mean(np.abs(y_t - y_p))
        ax.text(0.05, 0.95, f'r={r:.2f}\nMAE={mae:.2f}', transform=ax.transAxes, 
                va='top', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.set_title(subj)

# Hide unused subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR / '9_per_subject_scatter.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 10: RESIDUAL DISTRIBUTION
# =============================================================================
print("10. Residual Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

residuals = [y_pred_imu - y_true_imu, y_pred_ppg - y_true_ppg, y_pred_eda - y_true_eda]
titles = ['IMU', 'PPG', 'EDA']
colors = [COLORS['imu'], COLORS['ppg'], COLORS['eda']]

for ax, resid, title, color in zip(axes, residuals, titles, colors):
    ax.hist(resid, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(resid), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(resid):.2f}')
    ax.set_xlabel('Residual (Predicted - Actual)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUT_DIR / '10_residual_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 11: GREEDY LEARNING CURVE
# =============================================================================
print("11. Greedy Feature Selection Curve...")

fig, ax = plt.subplots(figsize=(10, 5))

# Plot learning curves
ax.plot(range(1, len(greedy_r_imu)+1), greedy_r_imu, 'o-', color=COLORS['imu'], linewidth=2, markersize=8, label='IMU')
ax.plot(range(1, len(greedy_r_ppg)+1), greedy_r_ppg, 's-', color=COLORS['ppg'], linewidth=2, markersize=8, label='PPG')
ax.plot(range(1, len(greedy_r_eda)+1), greedy_r_eda, '^-', color=COLORS['eda'], linewidth=2, markersize=8, label='EDA')

ax.set_xlabel('Number of Features (Greedy Selection)')
ax.set_ylabel('LOSO Pearson r')
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.1, 0.7)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '11_greedy_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 12: ACTUAL BORG DISTRIBUTION PER SUBJECT
# =============================================================================
print("12. Borg Distribution per Subject...")

fig, ax = plt.subplots(figsize=(10, 5))

subjects = ['P1', 'P2', 'P3', 'P4', 'P5']
borg_data = [df_labeled[df_labeled['subject'] == s]['borg'].values for s in subjects]

bp = ax.boxplot(borg_data, labels=subjects, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(COLORS['imu'])
    patch.set_alpha(0.6)

ax.set_ylabel('Borg CR10')
ax.set_ylim(-0.5, 10.5)

plt.tight_layout()
plt.savefig(OUT_DIR / '12_borg_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print(f"ALL PLOTS SAVED TO: {OUT_DIR}")
print("="*70)

print(f"""
FINAL LOSO RESULTS:

┌───────────┬──────┬────────┬────────┬─────────┬─────────┬─────────┬─────────┐
│ Modality  │ n    │ All r  │All MAE │ Top10 r │Top10 MAE│ Greedy r│Grdy MAE │
├───────────┼──────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┤
│ IMU       │ {len(imu_cols):>3}  │ {mean_r_imu:>5.2f}  │ {mean_mae_imu:>5.2f}  │ {mean_r_imu10:>6.2f}  │ {mean_mae_imu10:>6.2f}  │ {mean_r_greedy_imu:>6.2f}  │ {mean_mae_greedy_imu:>5.2f}   │
│ PPG       │ {len(ppg_cols):>3}  │ {mean_r_ppg:>5.2f}  │ {mean_mae_ppg:>5.2f}  │ {mean_r_ppg10:>6.2f}  │ {mean_mae_ppg10:>6.2f}  │ {mean_r_greedy_ppg:>6.2f}  │ {mean_mae_greedy_ppg:>5.2f}   │
│ EDA       │ {len(eda_cols):>3}  │ {mean_r_eda:>5.2f}  │ {mean_mae_eda:>5.2f}  │ {mean_r_eda10:>6.2f}  │ {mean_mae_eda10:>6.2f}  │ {mean_r_greedy_eda:>6.2f}  │ {mean_mae_greedy_eda:>5.2f}   │
└───────────┴──────┴────────┴────────┴─────────┴─────────┴─────────┴─────────┘

Within-Patient vs Cross-Patient:
┌───────────┬─────────────┬─────────────┬───────────┐
│ Modality  │ Within r    │ Cross r     │ Gap (Δr)  │
├───────────┼─────────────┼─────────────┼───────────┤
│ IMU       │ {mean_within_imu:>6.2f}      │ {mean_r_imu:>6.2f}      │ {mean_within_imu - mean_r_imu:>6.2f}    │
│ PPG       │ {mean_within_ppg:>6.2f}      │ {mean_r_ppg:>6.2f}      │ {mean_within_ppg - mean_r_ppg:>6.2f}    │
│ EDA       │ {mean_within_eda:>6.2f}      │ {mean_r_eda:>6.2f}      │ {mean_within_eda - mean_r_eda:>6.2f}    │
└───────────┴─────────────┴─────────────┴───────────┘

Greedy selected features:
  IMU: {greedy_imu}
  PPG: {greedy_ppg}
  EDA: {greedy_eda}
""")
