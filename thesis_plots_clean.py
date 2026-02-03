#!/usr/bin/env python3
"""
Clean professional thesis visualizations - NO annotations, NO text boxes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_clean")
OUT_DIR.mkdir(exist_ok=True)

# Professional style
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
print("GENERATING CLEAN THESIS VISUALIZATIONS")
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

print(f"  Loaded {len(df_labeled)} windows from 5 subjects")
print(f"  Features: IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_top_k_features(feature_cols, k=10):
    """Get top K features by ABSOLUTE correlation with Borg."""
    correlations = []
    for col in feature_cols:
        if col in df_labeled.columns:
            x = df_labeled[col].fillna(0).replace([np.inf, -np.inf], 0)
            y = df_labeled['borg']
            if x.std() > 0:
                r, _ = pearsonr(x, y)
                if not np.isnan(r):
                    correlations.append((col, r, abs(r)))
    correlations.sort(key=lambda x: x[2], reverse=True)
    return [(c[0], c[1]) for c in correlations[:k]]

def run_loso(feature_cols):
    """Run LOSO and return per-subject metrics."""
    per_subj_metrics = {}
    all_true, all_pred = [], []
    
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
        
        if len(y_test) > 2 and np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            per_subj_metrics[test_subj] = {'r': r, 'mae': mae, 'rmse': rmse, 'n': len(y_test)}
            all_true.extend(y_test)
            all_pred.extend(y_pred)
    
    return np.array(all_true), np.array(all_pred), per_subj_metrics

# =============================================================================
# RUN LOSO FOR ALL MODALITIES
# =============================================================================
print("\nRunning LOSO evaluation...")

y_true_imu, y_pred_imu, metrics_imu = run_loso(imu_cols)
y_true_ppg, y_pred_ppg, metrics_ppg = run_loso(ppg_cols)
y_true_eda, y_pred_eda, metrics_eda = run_loso(eda_cols)

mean_r_imu = np.mean([m['r'] for m in metrics_imu.values()])
mean_r_ppg = np.mean([m['r'] for m in metrics_ppg.values()])
mean_r_eda = np.mean([m['r'] for m in metrics_eda.values()])

mean_mae_imu = np.mean([m['mae'] for m in metrics_imu.values()])
mean_mae_ppg = np.mean([m['mae'] for m in metrics_ppg.values()])
mean_mae_eda = np.mean([m['mae'] for m in metrics_eda.values()])

# Top 10 per modality
top10_imu = [f[0] for f in get_top_k_features(imu_cols, 10)]
top10_ppg = [f[0] for f in get_top_k_features(ppg_cols, 10)]
top10_eda = [f[0] for f in get_top_k_features(eda_cols, 10)]

_, _, metrics_imu10 = run_loso(top10_imu)
_, _, metrics_ppg10 = run_loso(top10_ppg)
_, _, metrics_eda10 = run_loso(top10_eda)

mean_r_imu10 = np.mean([m['r'] for m in metrics_imu10.values()])
mean_r_ppg10 = np.mean([m['r'] for m in metrics_ppg10.values()])
mean_r_eda10 = np.mean([m['r'] for m in metrics_eda10.values()])

print(f"  IMU: r={mean_r_imu:.2f}, MAE={mean_mae_imu:.2f}")
print(f"  PPG: r={mean_r_ppg:.2f}, MAE={mean_mae_ppg:.2f}")
print(f"  EDA: r={mean_r_eda:.2f}, MAE={mean_mae_eda:.2f}")

# =============================================================================
# PLOT 1: PREDICTED VS ACTUAL - CLEAN SCATTER + BLAND-ALTMAN
# =============================================================================
print("\n1. Predicted vs Actual...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Scatter with density
ax1 = axes[0]
from scipy.stats import gaussian_kde
xy = np.vstack([y_true_imu, y_pred_imu])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = y_true_imu[idx], y_pred_imu[idx], z[idx]

scatter = ax1.scatter(x_sorted, y_sorted, c=z_sorted, cmap='viridis', s=25, alpha=0.7, edgecolors='none')
ax1.plot([0, 10], [0, 10], 'k--', linewidth=1.5, alpha=0.5)
z_fit = np.polyfit(y_true_imu, y_pred_imu, 1)
x_line = np.linspace(0, 10, 100)
ax1.plot(x_line, np.polyval(z_fit, x_line), color=COLORS['imu'], linewidth=2)
ax1.set_xlabel('Actual Borg CR10')
ax1.set_ylabel('Predicted Borg CR10')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-0.5, 10.5)
ax1.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Density')

# Right: Bland-Altman
ax2 = axes[1]
mean_vals = (y_true_imu + y_pred_imu) / 2
diff_vals = y_pred_imu - y_true_imu
mean_diff = np.mean(diff_vals)
std_diff = np.std(diff_vals)

ax2.scatter(mean_vals, diff_vals, c=COLORS['imu'], alpha=0.3, s=15, edgecolors='none')
ax2.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5)
ax2.axhline(mean_diff + 1.96*std_diff, color='#DC3545', linestyle='--', linewidth=1.2)
ax2.axhline(mean_diff - 1.96*std_diff, color='#DC3545', linestyle='--', linewidth=1.2)
ax2.set_xlabel('Mean of Actual & Predicted')
ax2.set_ylabel('Difference (Predicted - Actual)')
ax2.set_xlim(-0.5, 10.5)

plt.tight_layout()
plt.savefig(OUT_DIR / '1_predicted_vs_actual.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 1_predicted_vs_actual.png")

# =============================================================================
# PLOT 2: PER-SUBJECT PERFORMANCE (CLEAN)
# =============================================================================
print("\n2. Per-Subject Performance...")

fig, ax = plt.subplots(figsize=(8, 5))

subjects = list(metrics_imu.keys())
r_vals = [metrics_imu[s]['r'] for s in subjects]
mae_vals = [metrics_imu[s]['mae'] for s in subjects]

x_pos = np.arange(len(subjects))
width = 0.35

bars1 = ax.bar(x_pos - width/2, r_vals, width, label='Pearson r', color=COLORS['imu'], alpha=0.85)
ax_twin = ax.twinx()
bars2 = ax_twin.bar(x_pos + width/2, mae_vals, width, label='MAE', color='#17A2B8', alpha=0.85)

ax.set_ylabel('Pearson Correlation (r)', color=COLORS['imu'])
ax_twin.set_ylabel('MAE (Borg points)', color='#17A2B8')
ax.set_xticks(x_pos)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 0.8)
ax_twin.set_ylim(0, 3.0)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(OUT_DIR / '2_per_subject_performance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 2_per_subject_performance.png")

# =============================================================================
# PLOT 3: MODALITY COMPARISON (CLEAN - NO THRESHOLD LINE)
# =============================================================================
print("\n3. Modality Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))

modalities = ['IMU', 'PPG', 'EDA']
all_r = [mean_r_imu, mean_r_ppg, mean_r_eda]
top10_r = [mean_r_imu10, mean_r_ppg10, mean_r_eda10]
colors_mod = [COLORS['imu'], COLORS['ppg'], COLORS['eda']]

x = np.arange(len(modalities))
width = 0.35

bars1 = ax.bar(x - width/2, all_r, width, label='All Features', color=colors_mod, alpha=0.9, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, top10_r, width, label='Top 10 Features', color=colors_mod, alpha=0.5, edgecolor='black', linewidth=1, hatch='///')

ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_ylabel('LOSO Pearson Correlation (r)')
ax.set_xticks(x)
ax.set_xticklabels([f'{m}\n(n={n})' for m, n in zip(modalities, [len(imu_cols), len(ppg_cols), len(eda_cols)])])
ax.set_ylim(-0.1, 0.65)
ax.legend(loc='upper right')

# Value labels
for bars, values in [(bars1, all_r), (bars2, top10_r)]:
    for bar, val in zip(bars, values):
        y_pos = max(bar.get_height() + 0.02, 0.03)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / '3_modality_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 3_modality_comparison.png")

# =============================================================================
# PLOT 4: TOP FEATURES PER MODALITY (FIXED - PROPER CORRELATIONS)
# =============================================================================
print("\n4. Top Features per Modality (Fixed)...")

# Get actual correlations
top10_imu_corr = get_top_k_features(imu_cols, 10)
top10_ppg_corr = get_top_k_features(ppg_cols, 10)
top10_eda_corr = get_top_k_features(eda_cols, 10)

print("\n  DEBUG - Top 10 IMU correlations:")
for name, r in top10_imu_corr:
    print(f"    {name}: r={r:.3f}")

print("\n  DEBUG - Top 10 PPG correlations:")
for name, r in top10_ppg_corr:
    print(f"    {name}: r={r:.3f}")

print("\n  DEBUG - Top 10 EDA correlations:")
for name, r in top10_eda_corr:
    print(f"    {name}: r={r:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

def clean_name(name, modality):
    """Clean feature name for display."""
    name = name.replace(f'{modality}_', '').replace('_dyn__', ': ').replace('__', ': ')
    # Shorten if too long
    if len(name) > 30:
        name = name[:27] + '...'
    return name

def plot_features(ax, feature_tuples, color, modality_prefix):
    names = [clean_name(f[0], modality_prefix) for f in feature_tuples]
    values = [f[1] for f in feature_tuples]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Pearson Correlation (r)')
    ax.axvline(0, color='black', linewidth=1)
    
    # Dynamic x-limits based on data
    min_val = min(values) - 0.1
    max_val = max(values) + 0.1
    ax.set_xlim(min(min_val, -0.1), max(max_val, 0.1))
    
    # Value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.01 if val > 0 else val - 0.01
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', ha=ha, va='center', fontsize=8)

plot_features(axes[0], top10_imu_corr, COLORS['imu'], 'acc')
axes[0].set_title('IMU Features')

plot_features(axes[1], top10_ppg_corr, COLORS['ppg'], 'ppg')
axes[1].set_title('PPG Features')

plot_features(axes[2], top10_eda_corr, COLORS['eda'], 'eda')
axes[2].set_title('EDA Features')

plt.tight_layout()
plt.savefig(OUT_DIR / '4_top_features_per_modality.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 4_top_features_per_modality.png")

# =============================================================================
# PLOT 5: RESULTS TABLE (CLEAR)
# =============================================================================
print("\n5. Results Table...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Clear table with proper headers
table_data = [
    ['Modality', 'n Features', 'LOSO r\n(All)', 'LOSO r\n(Top 10)', 'MAE\n(Borg pts)', 'RMSE'],
    ['IMU', f'{len(imu_cols)}', f'{mean_r_imu:.2f}', f'{mean_r_imu10:.2f}', f'{mean_mae_imu:.2f}', f'{np.mean([m["rmse"] for m in metrics_imu.values()]):.2f}'],
    ['PPG', f'{len(ppg_cols)}', f'{mean_r_ppg:.2f}', f'{mean_r_ppg10:.2f}', f'{mean_mae_ppg:.2f}', f'{np.mean([m["rmse"] for m in metrics_ppg.values()]):.2f}'],
    ['EDA', f'{len(eda_cols)}', f'{mean_r_eda:.2f}', f'{mean_r_eda10:.2f}', f'{mean_mae_eda:.2f}', f'{np.mean([m["rmse"] for m in metrics_eda.values()]):.2f}'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center',
                 colColours=['#E8E8E8']*6)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.2)

# Bold header
for j in range(len(table_data[0])):
    table[(0, j)].get_text().set_fontweight('bold')

# Color code best results
for i in range(1, 4):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 1:  # IMU row
            cell.set_facecolor('#E3F2FD')  # Light blue

plt.tight_layout()
plt.savefig(OUT_DIR / '5_results_table.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 5_results_table.png")

# =============================================================================
# PLOT 6: PER-SUBJECT DETAILED TABLE
# =============================================================================
print("\n6. Per-Subject Results Table...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

# Per-subject table
header = ['Subject', 'IMU r', 'IMU MAE', 'PPG r', 'PPG MAE', 'EDA r', 'EDA MAE']
rows = []
for subj in ['P1', 'P2', 'P3', 'P4', 'P5']:
    row = [subj]
    for metrics in [metrics_imu, metrics_ppg, metrics_eda]:
        if subj in metrics:
            row.append(f'{metrics[subj]["r"]:.2f}')
            row.append(f'{metrics[subj]["mae"]:.2f}')
        else:
            row.extend(['-', '-'])
    rows.append(row)

# Add mean row
rows.append(['Mean', 
             f'{mean_r_imu:.2f}', f'{mean_mae_imu:.2f}',
             f'{mean_r_ppg:.2f}', f'{mean_mae_ppg:.2f}',
             f'{mean_r_eda:.2f}', f'{mean_mae_eda:.2f}'])

table = ax.table(cellText=rows, colLabels=header,
                 loc='center', cellLoc='center',
                 colColours=['#E8E8E8']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

# Bold header and mean row
for j in range(len(header)):
    table[(0, j)].get_text().set_fontweight('bold')
    table[(6, j)].get_text().set_fontweight('bold')
    table[(6, j)].set_facecolor('#D4D4D4')

plt.tight_layout()
plt.savefig(OUT_DIR / '6_per_subject_table.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 6_per_subject_table.png")

# =============================================================================
# PLOT 7: WITHIN VS CROSS PATIENT (SAME FEATURES - IMU ONLY)
# =============================================================================
print("\n7. Within vs Cross Patient (IMU features)...")

# Calculate within-patient performance using IMU features
within_patient_r = []
for subj in df_all['subject'].unique():
    subj_df = df_all[df_all['subject'] == subj].dropna(subset=['borg'])
    if len(subj_df) > 20:
        X = subj_df[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = subj_df['borg'].values
        
        # Simple train/test split within subject
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            within_patient_r.append(r)

mean_within = np.mean(within_patient_r)

fig, ax = plt.subplots(figsize=(7, 5))

categories = ['Within-Patient', 'Cross-Patient\n(LOSO)']
values = [mean_within, mean_r_imu]
alphas = [0.9, 0.5]

bars = ax.bar(categories, values, color=COLORS['imu'], edgecolor='black', linewidth=1.5)
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)

ax.set_ylabel('Pearson Correlation (r)')
ax.set_ylim(0, 1.0)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '7_within_vs_cross.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 7_within_vs_cross.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL PLOTS SAVED TO:", OUT_DIR)
print("="*70)

print(f"""
FINAL RESULTS:

┌───────────┬──────────┬──────────┬──────────┬─────────┬─────────┐
│ Modality  │ Features │ LOSO r   │ Top10 r  │ MAE     │ RMSE    │
├───────────┼──────────┼──────────┼──────────┼─────────┼─────────┤
│ IMU       │ {len(imu_cols):>3}      │ {mean_r_imu:>6.2f}   │ {mean_r_imu10:>6.2f}   │ {mean_mae_imu:>5.2f}   │ {np.mean([m['rmse'] for m in metrics_imu.values()]):>5.2f}   │
│ PPG       │ {len(ppg_cols):>3}      │ {mean_r_ppg:>6.2f}   │ {mean_r_ppg10:>6.2f}   │ {mean_mae_ppg:>5.2f}   │ {np.mean([m['rmse'] for m in metrics_ppg.values()]):>5.2f}   │
│ EDA       │ {len(eda_cols):>3}      │ {mean_r_eda:>6.2f}   │ {mean_r_eda10:>6.2f}   │ {mean_mae_eda:>5.2f}   │ {np.mean([m['rmse'] for m in metrics_eda.values()]):>5.2f}   │
└───────────┴──────────┴──────────┴──────────┴─────────┴─────────┘

Per-Subject (IMU):
""")

for subj in ['P1', 'P2', 'P3', 'P4', 'P5']:
    m = metrics_imu[subj]
    print(f"  {subj}: r={m['r']:.2f}, MAE={m['mae']:.2f}, n={m['n']}")

print(f"\nWithin-patient (IMU): r={mean_within:.2f}")
print(f"Cross-patient (IMU LOSO): r={mean_r_imu:.2f}")
print(f"Generalization gap: Δr={mean_within - mean_r_imu:.2f}")
