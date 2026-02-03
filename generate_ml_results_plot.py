#!/usr/bin/env python3
"""
Comprehensive ML Results Plot - All Important Metrics
Professional thesis-ready visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
SELECTED_FEATURES_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined/ml_expert_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['borg'])
selected_features = pd.read_csv(SELECTED_FEATURES_PATH, header=None)[0].tolist()
feat_cols = [c for c in selected_features if c in df.columns]

# Get subject info
if 'subject' in df.columns:
    df['subject_id'] = df['subject']
elif 'subject_id' not in df.columns:
    df['subject_id'] = 'P1'

subjects = df['subject_id'].unique()
print(f"Subjects: {subjects}")
print(f"Total samples: {len(df)}, Features: {len(feat_cols)}")

# =============================================================================
# RUN ALL METHODS WITH PROPER LOSO
# =============================================================================
print("\nRunning LOSO evaluation...")

def run_loso(df, feat_cols, method='raw'):
    """LOSO with different methods"""
    all_preds = []
    all_true = []
    all_subj = []
    
    subjects = df['subject_id'].unique()
    
    for test_subj in subjects:
        train_df = df[df['subject_id'] != test_subj]
        test_df = df[df['subject_id'] == test_subj]
        
        X_train = train_df[feat_cols].values
        y_train = train_df['borg'].values
        X_test = test_df[feat_cols].values
        y_test = test_df['borg'].values
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        if method == 'calibrated':
            # Use first 10% for calibration
            n_cal = max(10, int(0.1 * len(y_test)))
            offset = np.mean(y_test[:n_cal]) - np.mean(y_pred[:n_cal])
            y_pred = y_pred + offset
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_subj.extend([test_subj] * len(y_test))
    
    return np.array(all_preds), np.array(all_true), np.array(all_subj)

def run_within_subject(df, feat_cols, train_ratio=0.5):
    """Within-subject: train and test on same subject"""
    all_preds = []
    all_true = []
    all_subj = []
    
    for subj in df['subject_id'].unique():
        subj_df = df[df['subject_id'] == subj].reset_index(drop=True)
        n = len(subj_df)
        n_train = int(train_ratio * n)
        
        train_df = subj_df.iloc[:n_train]
        test_df = subj_df.iloc[n_train:]
        
        if len(test_df) < 10:
            continue
            
        X_train = train_df[feat_cols].values
        y_train = train_df['borg'].values
        X_test = test_df[feat_cols].values
        y_test = test_df['borg'].values
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_subj.extend([subj] * len(y_test))
    
    return np.array(all_preds), np.array(all_true), np.array(all_subj)

# Run all methods
print("  Method 1: Raw LOSO...")
pred_raw, true_raw, subj_raw = run_loso(df, feat_cols, method='raw')

print("  Method 3: LOSO + Calibration...")
pred_cal, true_cal, subj_cal = run_loso(df, feat_cols, method='calibrated')

print("  Method 4: Within-Subject (50/50)...")
pred_within, true_within, subj_within = run_within_subject(df, feat_cols, train_ratio=0.5)

# =============================================================================
# CALCULATE ALL METRICS
# =============================================================================
def calc_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Within ±1 Borg point accuracy
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    within_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
    
    # Category accuracy (LOW=0-2, MOD=3-4, HIGH=5+)
    def to_cat(x):
        if x <= 2: return 0
        elif x <= 4: return 1
        else: return 2
    
    true_cat = np.array([to_cat(y) for y in y_true])
    pred_cat = np.array([to_cat(y) for y in y_pred])
    cat_exact = np.mean(true_cat == pred_cat) * 100
    cat_within1 = np.mean(np.abs(true_cat - pred_cat) <= 1) * 100
    
    return {
        'r': r, 'rho': rho, 'mae': mae, 'rmse': rmse, 'r2': r2,
        'within_1': within_1, 'within_2': within_2,
        'cat_exact': cat_exact, 'cat_within1': cat_within1,
        'n': len(y_true)
    }

metrics_raw = calc_metrics(true_raw, pred_raw)
metrics_cal = calc_metrics(true_cal, pred_cal)
metrics_within = calc_metrics(true_within, pred_within)

print(f"\nMethod 1 (Raw LOSO):        r={metrics_raw['r']:.3f}, MAE={metrics_raw['mae']:.2f}")
print(f"Method 3 (Calibrated):      r={metrics_cal['r']:.3f}, MAE={metrics_cal['mae']:.2f}")
print(f"Method 4 (Within-Subject):  r={metrics_within['r']:.3f}, MAE={metrics_within['mae']:.2f}")

# =============================================================================
# CREATE COMPREHENSIVE FIGURE
# =============================================================================
print("\nGenerating comprehensive results plot...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Colors
colors = {'raw': '#3498db', 'cal': '#e67e22', 'within': '#27ae60'}
method_names = ['Raw LOSO', 'LOSO + Calibration', 'Within-Subject']

# =============================================================================
# ROW 1: SCATTER PLOTS
# =============================================================================
for i, (pred, true, subj, metrics, name, color) in enumerate([
    (pred_raw, true_raw, subj_raw, metrics_raw, 'Method 1: Raw LOSO', colors['raw']),
    (pred_cal, true_cal, subj_cal, metrics_cal, 'Method 3: LOSO + Calibration', colors['cal']),
    (pred_within, true_within, subj_within, metrics_within, 'Method 4: Within-Subject', colors['within'])
]):
    ax = fig.add_subplot(gs[0, i])
    
    # Hexbin for density
    hb = ax.hexbin(true, pred, gridsize=15, cmap='Blues', mincnt=1, alpha=0.7)
    
    # Perfect prediction line
    ax.plot([0, 7], [0, 7], 'k--', linewidth=2, label='Perfect', alpha=0.7)
    
    # ±1 bounds
    ax.fill_between([0, 7], [0-1, 7-1], [0+1, 7+1], alpha=0.15, color='green', label='±1 Borg')
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xlabel('Actual Borg CR-10')
    ax.set_ylabel('Predicted Borg CR-10')
    ax.set_title(f'{name}\nr = {metrics["r"]:.2f}, MAE = {metrics["mae"]:.2f}', fontweight='bold', color=color)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')

# =============================================================================
# ROW 2A: BAR CHART - KEY METRICS COMPARISON
# =============================================================================
ax_bar = fig.add_subplot(gs[1, 0:2])

x = np.arange(5)
width = 0.25

metrics_to_plot = ['r', 'rho', 'within_1', 'within_2', 'cat_exact']
labels = ['Pearson r', 'Spearman ρ', '±1 Borg (%)', '±2 Borg (%)', 'Category Exact (%)']

vals_raw = [metrics_raw['r'], metrics_raw['rho'], metrics_raw['within_1'], metrics_raw['within_2'], metrics_raw['cat_exact']]
vals_cal = [metrics_cal['r'], metrics_cal['rho'], metrics_cal['within_1'], metrics_cal['within_2'], metrics_cal['cat_exact']]
vals_within = [metrics_within['r'], metrics_within['rho'], metrics_within['within_1'], metrics_within['within_2'], metrics_within['cat_exact']]

# Scale correlations to percentage for visualization
vals_raw_scaled = [v*100 if i < 2 else v for i, v in enumerate(vals_raw)]
vals_cal_scaled = [v*100 if i < 2 else v for i, v in enumerate(vals_cal)]
vals_within_scaled = [v*100 if i < 2 else v for i, v in enumerate(vals_within)]

bars1 = ax_bar.bar(x - width, vals_raw_scaled, width, label='Raw LOSO', color=colors['raw'], edgecolor='black')
bars2 = ax_bar.bar(x, vals_cal_scaled, width, label='Calibrated', color=colors['cal'], edgecolor='black')
bars3 = ax_bar.bar(x + width, vals_within_scaled, width, label='Within-Subject', color=colors['within'], edgecolor='black')

ax_bar.set_ylabel('Value (×100 for correlations)')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=15, ha='right')
ax_bar.legend(loc='upper left')
ax_bar.set_title('Performance Metrics Comparison', fontweight='bold')
ax_bar.set_ylim(0, 100)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height:.0f}' if height > 10 else f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

# =============================================================================
# ROW 2B: ERROR METRICS
# =============================================================================
ax_err = fig.add_subplot(gs[1, 2])

x = np.arange(2)
width = 0.25

mae_vals = [metrics_raw['mae'], metrics_cal['mae'], metrics_within['mae']]
rmse_vals = [metrics_raw['rmse'], metrics_cal['rmse'], metrics_within['rmse']]

x2 = np.arange(3)
width2 = 0.35

bars_mae = ax_err.bar(x2 - width2/2, mae_vals, width2, label='MAE', color='#e74c3c', edgecolor='black')
bars_rmse = ax_err.bar(x2 + width2/2, rmse_vals, width2, label='RMSE', color='#9b59b6', edgecolor='black')

ax_err.set_ylabel('Error (Borg points)')
ax_err.set_xticks(x2)
ax_err.set_xticklabels(['Raw LOSO', 'Calibrated', 'Within-Subj'], rotation=15, ha='right')
ax_err.legend()
ax_err.set_title('Prediction Errors', fontweight='bold')

for bars in [bars_mae, bars_rmse]:
    for bar in bars:
        height = bar.get_height()
        ax_err.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

# =============================================================================
# ROW 3A: ERROR DISTRIBUTION
# =============================================================================
ax_hist = fig.add_subplot(gs[2, 0])

errors_raw = pred_raw - true_raw
errors_cal = pred_cal - true_cal
errors_within = pred_within - true_within

bins = np.arange(-5, 6, 0.5)
ax_hist.hist(errors_raw, bins=bins, alpha=0.5, label=f'Raw (μ={np.mean(errors_raw):.2f})', color=colors['raw'], edgecolor='black')
ax_hist.hist(errors_cal, bins=bins, alpha=0.5, label=f'Cal (μ={np.mean(errors_cal):.2f})', color=colors['cal'], edgecolor='black')
ax_hist.hist(errors_within, bins=bins, alpha=0.5, label=f'Within (μ={np.mean(errors_within):.2f})', color=colors['within'], edgecolor='black')

ax_hist.axvline(0, color='black', linestyle='--', linewidth=2)
ax_hist.axvline(-1, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax_hist.axvline(1, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax_hist.set_xlabel('Prediction Error (Predicted - Actual)')
ax_hist.set_ylabel('Count')
ax_hist.set_title('Error Distribution', fontweight='bold')
ax_hist.legend(fontsize=8)

# =============================================================================
# ROW 3B: PER-SUBJECT PERFORMANCE (Calibrated)
# =============================================================================
ax_subj = fig.add_subplot(gs[2, 1])

subj_metrics = []
for subj in np.unique(subj_cal):
    mask = subj_cal == subj
    r_subj, _ = pearsonr(true_cal[mask], pred_cal[mask])
    mae_subj = mean_absolute_error(true_cal[mask], pred_cal[mask])
    subj_metrics.append({'subject': subj, 'r': r_subj, 'mae': mae_subj, 'n': np.sum(mask)})

subj_df = pd.DataFrame(subj_metrics)
subj_df = subj_df.sort_values('r', ascending=True)

colors_subj = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(subj_df)))
bars = ax_subj.barh(range(len(subj_df)), subj_df['r'], color=colors_subj, edgecolor='black')

ax_subj.set_yticks(range(len(subj_df)))
ax_subj.set_yticklabels([f"{s} (n={n})" for s, n in zip(subj_df['subject'], subj_df['n'])])
ax_subj.set_xlabel('Pearson r')
ax_subj.set_title('Per-Subject Performance\n(Calibrated Method)', fontweight='bold')
ax_subj.axvline(metrics_cal['r'], color='red', linestyle='--', linewidth=2, label=f'Mean r={metrics_cal["r"]:.2f}')
ax_subj.legend(loc='lower right')

for i, (_, row) in enumerate(subj_df.iterrows()):
    ax_subj.annotate(f'r={row["r"]:.2f}', xy=(row['r'] + 0.02, i), va='center', fontsize=9)

# =============================================================================
# ROW 3C: METRICS TABLE
# =============================================================================
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('off')

table_data = [
    ['Metric', 'Raw LOSO', 'Calibrated', 'Within-Subj'],
    ['Pearson r', f'{metrics_raw["r"]:.3f}', f'{metrics_cal["r"]:.3f}', f'{metrics_within["r"]:.3f}'],
    ['Spearman ρ', f'{metrics_raw["rho"]:.3f}', f'{metrics_cal["rho"]:.3f}', f'{metrics_within["rho"]:.3f}'],
    ['R²', f'{metrics_raw["r2"]:.3f}', f'{metrics_cal["r2"]:.3f}', f'{metrics_within["r2"]:.3f}'],
    ['MAE', f'{metrics_raw["mae"]:.2f}', f'{metrics_cal["mae"]:.2f}', f'{metrics_within["mae"]:.2f}'],
    ['RMSE', f'{metrics_raw["rmse"]:.2f}', f'{metrics_cal["rmse"]:.2f}', f'{metrics_within["rmse"]:.2f}'],
    ['±1 Borg (%)', f'{metrics_raw["within_1"]:.1f}', f'{metrics_cal["within_1"]:.1f}', f'{metrics_within["within_1"]:.1f}'],
    ['±2 Borg (%)', f'{metrics_raw["within_2"]:.1f}', f'{metrics_cal["within_2"]:.1f}', f'{metrics_within["within_2"]:.1f}'],
    ['Cat. Exact (%)', f'{metrics_raw["cat_exact"]:.1f}', f'{metrics_cal["cat_exact"]:.1f}', f'{metrics_within["cat_exact"]:.1f}'],
    ['N samples', f'{metrics_raw["n"]}', f'{metrics_cal["n"]}', f'{metrics_within["n"]}'],
]

table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Header row styling
for j in range(4):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Best value highlighting (green)
for i in range(1, 10):
    vals = []
    for j in range(1, 4):
        try:
            vals.append(float(table_data[i][j].replace('%', '')))
        except:
            vals.append(0)
    
    if i in [4, 5]:  # MAE, RMSE - lower is better
        best_idx = np.argmin(vals) + 1
    else:  # Others - higher is better
        best_idx = np.argmax(vals) + 1
    
    table[(i, best_idx)].set_facecolor('#d5f5e3')

ax_table.set_title('Complete Metrics Summary', fontweight='bold', pad=20)

# =============================================================================
# MAIN TITLE
# =============================================================================
fig.suptitle('Effort Estimation Results: Ridge Regression on 5 Elderly Subjects\n34 Selected Features (EDA + IMU + PPG)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / '30_ml_results_comprehensive.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✓ Saved: {OUTPUT_DIR / '30_ml_results_comprehensive.png'}")

# =============================================================================
# PRINT HONEST SUMMARY
# =============================================================================
print("\n" + "="*70)
print("HONEST RESULTS SUMMARY")
print("="*70)
print(f"\n{'Method':<25} {'r':>8} {'MAE':>8} {'±1 Borg':>10} {'±2 Borg':>10}")
print("-"*65)
print(f"{'Raw LOSO':<25} {metrics_raw['r']:>8.3f} {metrics_raw['mae']:>8.2f} {metrics_raw['within_1']:>9.1f}% {metrics_raw['within_2']:>9.1f}%")
print(f"{'LOSO + Calibration':<25} {metrics_cal['r']:>8.3f} {metrics_cal['mae']:>8.2f} {metrics_cal['within_1']:>9.1f}% {metrics_cal['within_2']:>9.1f}%")
print(f"{'Within-Subject (50/50)':<25} {metrics_within['r']:>8.3f} {metrics_within['mae']:>8.2f} {metrics_within['within_1']:>9.1f}% {metrics_within['within_2']:>9.1f}%")
print("="*70)
print("\nKey takeaways:")
print(f"  • Best generalization (new subjects): r={metrics_cal['r']:.2f}, ±1 Borg = {metrics_cal['within_1']:.0f}%")
print(f"  • Best personalized: r={metrics_within['r']:.2f}, ±1 Borg = {metrics_within['within_1']:.0f}%")
print(f"  • Average error: {metrics_cal['mae']:.1f} Borg points (calibrated)")
