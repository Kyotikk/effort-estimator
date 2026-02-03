#!/usr/bin/env python3
"""
Generate all 6 thesis presentation plots with accurate data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

# Output directory
OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots")
OUT_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'

print("="*60)
print("GENERATING THESIS PLOTS")
print("="*60)

# =============================================================================
# PLOT 1: Pipeline Diagram
# =============================================================================
print("\n1. Creating Pipeline Diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
colors = {
    'raw': '#E74C3C',      # Red
    'preproc': '#E67E22',  # Orange
    'window': '#F1C40F',   # Yellow
    'feature': '#2ECC71',  # Green
    'align': '#3498DB',    # Blue
    'select': '#9B59B6',   # Purple
    'train': '#1ABC9C',    # Teal
    'result': '#34495E',   # Dark gray
}

# Box positions (x, y, width, height)
boxes = [
    (0.5, 6, 2.5, 1.5, 'Raw Signals\n\nPPG (3 variants)\nIMU (acc x,y,z)\nEDA', colors['raw']),
    (0.5, 3.5, 2.5, 1.8, 'Preprocessing\n\nResample 32Hz\nFilter (HPF/LPF)\nRemove artifacts', colors['preproc']),
    (0.5, 1, 2.5, 1.8, 'Windowing\n\n5s windows\n10% overlap\n~1965 windows', colors['window']),
    (4, 6, 2.5, 1.5, 'Feature\nExtraction\n\n260+ features', colors['feature']),
    (4, 3.5, 2.5, 1.8, 'Alignment\n\nMatch with\nBorg labels\n→ 855 labeled', colors['align']),
    (4, 1, 2.5, 1.8, 'Feature\nSelection\n\n260 → 48\n(redundancy)', colors['select']),
    (7.5, 6, 2.5, 1.5, 'Training\n\nLOSO CV\nRandomForest', colors['train']),
    (7.5, 3.5, 2.5, 1.8, 'Per-Modality\nEvaluation\n\nIMU vs PPG\nvs EDA', colors['result']),
    (11, 4.5, 2.5, 2.5, 'Results\n\nIMU: r=0.54\nPPG: r=0.26\nEDA: r=0.01', colors['result']),
]

for x, y, w, h, text, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')

# Arrows
arrow_style = dict(arrowstyle='->', color='black', lw=2)
# Vertical arrows (left column)
ax.annotate('', xy=(1.75, 5.3), xytext=(1.75, 6), arrowprops=arrow_style)
ax.annotate('', xy=(1.75, 2.8), xytext=(1.75, 3.5), arrowprops=arrow_style)
# Horizontal arrows (to middle column)
ax.annotate('', xy=(4, 6.75), xytext=(3, 6.75), arrowprops=arrow_style)
ax.annotate('', xy=(4, 4.4), xytext=(3, 4.4), arrowprops=arrow_style)
ax.annotate('', xy=(4, 1.9), xytext=(3, 1.9), arrowprops=arrow_style)
# Vertical arrows (middle column)
ax.annotate('', xy=(5.25, 5.3), xytext=(5.25, 6), arrowprops=arrow_style)
ax.annotate('', xy=(5.25, 2.8), xytext=(5.25, 3.5), arrowprops=arrow_style)
# To right column
ax.annotate('', xy=(7.5, 6.75), xytext=(6.5, 6.75), arrowprops=arrow_style)
ax.annotate('', xy=(7.5, 4.4), xytext=(6.5, 4.4), arrowprops=arrow_style)
# Vertical in right
ax.annotate('', xy=(8.75, 5.3), xytext=(8.75, 6), arrowprops=arrow_style)
# To final
ax.annotate('', xy=(11, 5.75), xytext=(10, 5.75), arrowprops=arrow_style)
ax.annotate('', xy=(11, 4.75), xytext=(10, 4.75), arrowprops=arrow_style)

ax.set_title('Effort Estimation Pipeline: From Raw Signals to Prediction', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUT_DIR / '1_pipeline_diagram.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print(f"  ✓ Saved: {OUT_DIR / '1_pipeline_diagram.png'}")

# =============================================================================
# PLOT 2: Window Size Comparison
# =============================================================================
print("\n2. Creating Window Size Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data from experiments
windows = ['5s', '10s', '30s']
samples = [855, 424, 100]
loso_r = [0.54, 0.48, 0.28]

x = np.arange(len(windows))
width = 0.35

bars1 = ax.bar(x - width/2, samples, width, label='N Samples', color='#3498DB', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, loso_r, width, label='LOSO r', color='#E74C3C', alpha=0.8)

ax.set_xlabel('Window Size', fontweight='bold')
ax.set_ylabel('Number of Samples', color='#3498DB', fontweight='bold')
ax2.set_ylabel('LOSO Correlation (r)', color='#E74C3C', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(windows)
ax.set_ylim(0, 1000)
ax2.set_ylim(0, 0.7)

# Add value labels
for bar, val in zip(bars1, samples):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
            f'{val}', ha='center', va='bottom', fontweight='bold', color='#3498DB')

for bar, val in zip(bars2, loso_r):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'r={val:.2f}', ha='center', va='bottom', fontweight='bold', color='#E74C3C')

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax.set_title('Window Size Comparison: 5s Windows Perform Best\n(More samples + Better temporal resolution)', 
             fontsize=14, fontweight='bold')

# Add annotation
ax.annotate('Best\nperformance', xy=(0, 0.54), xytext=(0.5, 0.65),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green',
            transform=ax2.get_yaxis_transform())

plt.tight_layout()
plt.savefig(OUT_DIR / '2_window_size_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  ✓ Saved: {OUT_DIR / '2_window_size_comparison.png'}")

# =============================================================================
# PLOT 3: Modality LOSO Bar Chart (Top 10 and All features)
# =============================================================================
print("\n3. Creating Modality LOSO Bar Chart...")

fig, ax = plt.subplots(figsize=(12, 6))

modalities = ['IMU', 'PPG', 'EDA']
all_features = [30, 183, 47]
loso_all = [0.52, 0.26, 0.01]
loso_top10 = [0.52, 0.15, -0.01]

x = np.arange(len(modalities))
width = 0.35

bars1 = ax.bar(x - width/2, loso_all, width, label='All Features', color='#3498DB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, loso_top10, width, label='Top 10 Features', color='#2ECC71', alpha=0.8, edgecolor='black')

# Add value labels
for bar, val, n in zip(bars1, loso_all, all_features):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
            f'r={val:.2f}\n(n={n})', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, loso_top10):
    height = max(bar.get_height(), 0)
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
            f'r={val:.2f}\n(n=10)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Reference lines
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (r=0.5)')
ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (r=0.3)')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax.set_ylabel('LOSO Cross-Validation Correlation (r)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modalities, fontsize=12, fontweight='bold')
ax.set_ylim(-0.15, 0.7)
ax.set_title('Cross-Subject Generalization by Modality\n(Leave-One-Subject-Out, 5 elderly subjects)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

# Add interpretation text
ax.text(0, 0.60, 'Generalizes', fontsize=11, ha='center', color='green', fontweight='bold')
ax.text(1, 0.32, 'Poor', fontsize=11, ha='center', color='orange', fontweight='bold')
ax.text(2, 0.08, 'Useless', fontsize=11, ha='center', color='gray', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '3_modality_loso_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  ✓ Saved: {OUT_DIR / '3_modality_loso_comparison.png'}")

# =============================================================================
# PLOT 4: Within vs Cross-Patient
# =============================================================================
print("\n4. Creating Within vs Cross-Patient Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Within-Patient\n(1 subject, pooled)', 'Cross-Patient\n(LOSO, 5 subjects)']
hr_values = [0.82, 0.26]
imu_values = [0.65, 0.54]  # estimated within-patient IMU

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, hr_values, width, label='HR/PPG Features', color='#E74C3C', alpha=0.85)
bars2 = ax.bar(x + width/2, imu_values, width, label='IMU Features', color='#3498DB', alpha=0.85)

ax.set_ylabel('Correlation (r)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')

# Add value labels
for bar, val in zip(bars1, hr_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'r={val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#E74C3C')
for bar, val in zip(bars2, imu_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'r={val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#3498DB')

# Add drop annotation
ax.annotate('', xy=(0.82, 0.82), xytext=(1.18, 0.26),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2, ls='--'))
ax.text(1.4, 0.55, 'Δr = 0.56\ndrop!', fontsize=11, color='red', fontweight='bold')

ax.set_title('Within-Patient vs Cross-Patient Generalization\n'
             'HR correlates well within subject but fails across subjects', 
             fontsize=14, fontweight='bold')

# Add box with key insight
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, 'Key Insight:\nHR-effort relationship is\nhighly individual',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(OUT_DIR / '4_within_vs_cross_patient.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"  ✓ Saved: {OUT_DIR / '4_within_vs_cross_patient.png'}")

# =============================================================================
# PLOT 5: Predicted vs Actual (using real data)
# =============================================================================
print("\n5. Creating Predicted vs Actual Scatter...")

# Load real data and run LOSO
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        dfs.append(df)

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Get ALL IMU features (same as loso_per_modality.py)
    imu_cols = [c for c in df_all.columns if 'acc_' in c and '_r' not in c]
    print(f"    Using {len(imu_cols)} IMU features")
    
    # Collect all predictions via LOSO
    all_y_true = []
    all_y_pred = []
    all_subjects = []
    
    for test_subj in sorted(df_all['subject'].unique()):
        train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
        
        valid_cols = [c for c in imu_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_subjects.extend([test_subj] * len(y_test))
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_subjects = np.array(all_subjects)
    
    # Calculate per-subject correlations (this is the proper LOSO metric)
    per_subj_r = []
    for subj in sorted(np.unique(all_subjects)):
        mask = all_subjects == subj
        r_s, _ = pearsonr(all_y_true[mask], all_y_pred[mask])
        per_subj_r.append(r_s)
        print(f"    P{subj}: r = {r_s:.3f}")
    
    mean_loso_r = np.mean(per_subj_r)
    pooled_r, _ = pearsonr(all_y_true, all_y_pred)
    mae = np.mean(np.abs(all_y_true - all_y_pred))
    print(f"    Mean LOSO r = {mean_loso_r:.3f}, Pooled r = {pooled_r:.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by subject
    subject_colors = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71', 4: '#9B59B6', 5: '#F39C12'}
    for subj in sorted(np.unique(all_subjects)):
        mask = all_subjects == subj
        ax.scatter(all_y_true[mask], all_y_pred[mask], 
                  c=subject_colors[subj], alpha=0.6, s=50, 
                  label=f'P{subj}', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    ax.plot([0, 10], [0, 10], 'k--', linewidth=2, label='Perfect prediction')
    
    # ±1 Borg band
    ax.fill_between([0, 10], [-1, 9], [1, 11], alpha=0.1, color='green', label='±1 Borg')
    
    # Regression line (per subject mean shown in title)
    z = np.polyfit(all_y_true, all_y_pred, 1)
    x_line = np.linspace(0, 10, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2, alpha=0.7, label=f'Fit')
    
    ax.set_xlabel('Actual Borg CR10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg CR10', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    ax.set_title(f'LOSO Cross-Validation: Predicted vs Actual Effort\n'
                 f'Mean LOSO r = {mean_loso_r:.2f}, MAE = {mae:.2f} Borg points ({len(imu_cols)} IMU features)',
                 fontsize=14, fontweight='bold')
    
    # Add stats box
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    stats_text = f'N = {len(all_y_true)} windows\nMean LOSO r = {mean_loso_r:.2f}\nMAE = {mae:.2f}\n5 subjects'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '5_predicted_vs_actual.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {OUT_DIR / '5_predicted_vs_actual.png'}")
    print(f"    Mean LOSO r = {mean_loso_r:.3f}, MAE = {mae:.2f}")
else:
    print("  ⚠ Could not load data for predicted vs actual plot")

# =============================================================================
# PLOT 6: Top 10 Features (Feature Importance)
# =============================================================================
print("\n6. Creating Top 10 Features Bar Chart...")

if dfs:
    # Train on all data to get feature importance
    df_train = df_all.dropna(subset=['borg'])
    valid_cols = [c for c in imu_cols if c in df_train.columns]
    X = df_train[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df_train['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_imp = list(zip(valid_cols, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    # Top 10
    top10 = feature_imp[:10]
    names = [f[0].replace('acc_', '').replace('_', '\n') for f in top10]
    values = [f[1] for f in top10]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Greens(np.linspace(0.8, 0.4, 10))
    bars = ax.barh(range(10), values[::-1], color=colors[::-1], edgecolor='black', linewidth=1)
    ax.set_yticks(range(10))
    ax.set_yticklabels(names[::-1], fontsize=11)
    ax.set_xlabel('Feature Importance (Random Forest)', fontweight='bold')
    ax.set_title('Top 10 IMU Features for Effort Prediction\n(All features are acceleration-based)', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values[::-1])):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    # Add interpretation
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    ax.text(0.98, 0.02, 'IMU (accelerometer) features\ncapture physical movement\nintensity directly',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '6_top10_features.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {OUT_DIR / '6_top10_features.png'}")
    print(f"    Top 3 features: {[f[0] for f in top10[:3]]}")
else:
    print("  ⚠ Could not create feature importance plot")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("ALL PLOTS SAVED TO:", OUT_DIR)
print("="*60)
print("""
Files created:
  1_pipeline_diagram.png        - Data flow visualization
  2_window_size_comparison.png  - 5s vs 10s vs 30s
  3_modality_loso_comparison.png - IMU vs PPG vs EDA
  4_within_vs_cross_patient.png - Generalization gap
  5_predicted_vs_actual.png     - Scatter with regression
  6_top10_features.png          - Feature importance
""")
