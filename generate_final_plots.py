#!/usr/bin/env python3
"""
Final Thesis Plots Generator - Accurate & Up-to-Date Results
Uses the 32 auto-selected features with proper methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use the new pipeline output
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
SELECTED_FEATURES_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined/ml_expert_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

# Subject colors (colorblind-friendly)
SUBJECT_COLORS = {
    'sim_elderly1': '#E69F00',  # Orange
    'sim_elderly2': '#56B4E9',  # Sky blue
    'sim_elderly3': '#009E73',  # Green
    'sim_elderly4': '#CC79A7',  # Pink
    'sim_elderly5': '#F0E442',  # Yellow
}

SUBJECT_LABELS = {
    'sim_elderly1': 'P1',
    'sim_elderly2': 'P2', 
    'sim_elderly3': 'P3',
    'sim_elderly4': 'P4',
    'sim_elderly5': 'P5',
}

# Category colors
CAT_COLORS = {'LOW': '#2ecc71', 'MOD': '#f39c12', 'HIGH': '#e74c3c'}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*70)
print("LOADING DATA...")
print("="*70)

# Load the full aligned dataset
df_full = pd.read_csv(DATA_PATH)
df_full = df_full.dropna(subset=['borg'])  # Only labeled samples
print(f"Full dataset: {len(df_full)} samples")

# Load the selected features list
selected_features = pd.read_csv(SELECTED_FEATURES_PATH, header=None)[0].tolist()
print(f"Selected features: {len(selected_features)}")

# Get feature columns that exist in the data
feat_cols = [c for c in selected_features if c in df_full.columns]
print(f"Features found in data: {len(feat_cols)}")

# Create filtered dataframe with only selected features
meta_cols = ['subject', 'borg', 't_center']
keep_cols = [c for c in meta_cols if c in df_full.columns] + feat_cols
df = df_full[keep_cols].copy()

# Categorize features
eda_feats = [c for c in feat_cols if c.startswith('eda_')]
imu_feats = [c for c in feat_cols if c.startswith('acc_')]
ppg_feats = [c for c in feat_cols if c.startswith('ppg_')]

print(f"  EDA: {len(eda_feats)}, IMU: {len(imu_feats)}, PPG: {len(ppg_feats)}")

# Category mapping
def to_cat(borg):
    if borg <= 2: return 0  # LOW
    elif borg <= 4: return 1  # MOD
    else: return 2  # HIGH

def to_cat_label(borg):
    if borg <= 2: return 'LOW'
    elif borg <= 4: return 'MOD'
    else: return 'HIGH'

# =============================================================================
# COMPUTE ALL PREDICTIONS
# =============================================================================
print("\n" + "="*70)
print("COMPUTING PREDICTIONS...")
print("="*70)

X = df[feat_cols].values
y = df['borg'].values
subjects = df['subject'].values

# Method 1: Raw LOSO
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

method1_preds = np.full(len(df), np.nan)
for subj in df['subject'].unique():
    test_mask = df['subject'] == subj
    train_mask = ~test_mask
    model = Ridge(alpha=1.0).fit(X_scaled[train_mask], y[train_mask])
    method1_preds[test_mask] = model.predict(X_scaled[test_mask])

# Method 3: LOSO + Calibration
method3_preds = method1_preds.copy()
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    bias = np.mean(y[mask]) - np.mean(method1_preds[mask])
    method3_preds[mask] += bias

# Method 4: Within-Subject (20% train, 80% test)
np.random.seed(42)
method4_preds = np.full(len(df), np.nan)
method4_test_mask = np.zeros(len(df), dtype=bool)
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    subj_idx = np.where(mask)[0]
    np.random.shuffle(subj_idx)
    train_idx = subj_idx[:int(len(subj_idx)*0.2)]
    test_idx = subj_idx[int(len(subj_idx)*0.2):]
    
    scaler_s = StandardScaler().fit(X[train_idx])
    X_train = scaler_s.transform(X[train_idx])
    X_test = scaler_s.transform(X[test_idx])
    
    model = Ridge(alpha=1.0).fit(X_train, y[train_idx])
    method4_preds[test_idx] = model.predict(X_test)
    method4_preds[train_idx] = y[train_idx]
    method4_test_mask[test_idx] = True

# Compute per-subject correlations for Method 1
per_subj_r_m1 = {}
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    r, _ = pearsonr(y[mask], method1_preds[mask])
    per_subj_r_m1[subj] = r

# Compute per-subject correlations for Method 3
per_subj_r_m3 = {}
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    r, _ = pearsonr(y[mask], method3_preds[mask])
    per_subj_r_m3[subj] = r

print("✓ All predictions computed")

# =============================================================================
# PLOT 01: DATA OVERVIEW
# =============================================================================
print("\nGenerating 01_data_overview.png...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Samples per subject
ax = axes[0]
subj_counts = df.groupby('subject').size()
colors = [SUBJECT_COLORS[s] for s in subj_counts.index]
bars = ax.bar([SUBJECT_LABELS[s] for s in subj_counts.index], subj_counts.values, color=colors, edgecolor='black')
ax.set_ylabel('Number of Windows')
ax.set_xlabel('Subject')
ax.set_title('A. Samples per Subject', fontweight='bold')
for bar, count in zip(bars, subj_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), 
            ha='center', va='bottom', fontsize=10)

# Borg distribution
ax = axes[1]
ax.hist(df['borg'], bins=np.arange(-0.5, 11.5, 1), color='steelblue', edgecolor='black', alpha=0.8)
ax.set_xlabel('Borg CR-10 Rating')
ax.set_ylabel('Count')
ax.set_title('B. Borg Distribution', fontweight='bold')
ax.axvline(2.5, color='green', linestyle='--', linewidth=2, label='LOW/MOD')
ax.axvline(4.5, color='orange', linestyle='--', linewidth=2, label='MOD/HIGH')
ax.legend()

# Category distribution
ax = axes[2]
df['category'] = df['borg'].apply(to_cat_label)
cat_counts = df['category'].value_counts()[['LOW', 'MOD', 'HIGH']]
colors = [CAT_COLORS[c] for c in cat_counts.index]
wedges, texts, autotexts = ax.pie(cat_counts.values, labels=cat_counts.index, colors=colors,
                                   autopct='%1.1f%%', startangle=90, explode=[0.02]*3)
ax.set_title('C. Category Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_data_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 01_data_overview.png")

# =============================================================================
# PLOT 02: BORG DISTRIBUTION BY SUBJECT
# =============================================================================
print("Generating 02_borg_by_subject.png...")

fig, ax = plt.subplots(figsize=(10, 6))

positions = []
for i, subj in enumerate(sorted(df['subject'].unique())):
    subj_data = df[df['subject'] == subj]['borg']
    bp = ax.boxplot([subj_data], positions=[i], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(SUBJECT_COLORS[subj])
    bp['boxes'][0].set_alpha(0.7)
    positions.append(i)

ax.set_xticks(positions)
ax.set_xticklabels([SUBJECT_LABELS[s] for s in sorted(df['subject'].unique())])
ax.set_ylabel('Borg CR-10 Rating')
ax.set_xlabel('Subject')
ax.set_title('Borg Distribution by Subject', fontweight='bold', fontsize=14)
ax.axhline(2.5, color='green', linestyle='--', alpha=0.5, label='LOW/MOD threshold')
ax.axhline(4.5, color='orange', linestyle='--', alpha=0.5, label='MOD/HIGH threshold')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_borg_by_subject.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 02_borg_by_subject.png")

# =============================================================================
# PLOT 03: FEATURE SELECTION SUMMARY
# =============================================================================
print("Generating 03_feature_selection.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature counts by modality
ax = axes[0]
modalities = ['EDA', 'IMU', 'PPG']
counts = [len(eda_feats), len(imu_feats), len(ppg_feats)]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(modalities, counts, color=colors, edgecolor='black')
ax.set_ylabel('Number of Features')
ax.set_title('A. Selected Features by Modality', fontweight='bold')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(count), 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(counts) + 5)

# Feature correlation with Borg
ax = axes[1]
correlations = []
for feat in feat_cols:
    r, _ = pearsonr(df[feat].fillna(0), df['borg'])
    correlations.append(abs(r))
    
top_n = 15
sorted_idx = np.argsort(correlations)[-top_n:][::-1]
top_feats = [feat_cols[i] for i in sorted_idx]
top_corrs = [correlations[i] for i in sorted_idx]

# Color by modality
colors = []
for f in top_feats:
    if f.startswith('eda_'): colors.append('#3498db')
    elif f.startswith('acc_'): colors.append('#e74c3c')
    else: colors.append('#2ecc71')

y_pos = np.arange(len(top_feats))
ax.barh(y_pos, top_corrs, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f.replace('_dyn__', '\n') for f in top_feats], fontsize=8)
ax.set_xlabel('|Correlation with Borg|')
ax.set_title('B. Top 15 Features by Correlation', fontweight='bold')
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='EDA'),
                   Patch(facecolor='#e74c3c', label='IMU'),
                   Patch(facecolor='#2ecc71', label='PPG')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_feature_selection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 03_feature_selection.png")

# =============================================================================
# PLOT 04: DATA QUALITY - MISSINGNESS
# =============================================================================
print("Generating 04_quality_missingness.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get full feature set missingness
feat_full = [c for c in df_full.columns if c not in ['subject', 'borg', 't_center']]
missingness = df_full[feat_full].isna().mean().sort_values(ascending=False)
top_missing = missingness.head(20)

# Color by missingness level
colors = []
for val in top_missing.values:
    if val > 0.5: colors.append('#e74c3c')  # Red
    elif val > 0.3: colors.append('#f39c12')  # Orange
    else: colors.append('#2ecc71')  # Green

ax = axes[0]
y_pos = np.arange(len(top_missing))
ax.barh(y_pos, top_missing.values, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_missing.index, fontsize=9)
ax.set_xlabel('Missing Rate')
ax.set_title('A. Feature Missingness (Top 20)', fontweight='bold')
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax.invert_yaxis()
ax.legend()

# Per-subject missingness
ax = axes[1]
subj_miss = df_full.groupby('subject')[feat_full].apply(lambda x: x.isna().mean().mean())
colors = [SUBJECT_COLORS[s] for s in subj_miss.index]
bars = ax.bar([SUBJECT_LABELS[s] for s in subj_miss.index], subj_miss.values * 100, 
              color=colors, edgecolor='black')
ax.set_ylabel('Average Missing Rate (%)')
ax.set_xlabel('Subject')
ax.set_title('B. Average Missingness by Subject', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_quality_missingness.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 04_quality_missingness.png")

# =============================================================================
# PLOT 05: METHOD 1 - LOSO RAW RESULTS
# =============================================================================
print("Generating 05_method1_loso_raw.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall scatter
ax = axes[0]
for subj in sorted(df['subject'].unique()):
    mask = df['subject'] == subj
    ax.scatter(y[mask], method1_preds[mask], c=SUBJECT_COLORS[subj], 
               label=SUBJECT_LABELS[subj], alpha=0.6, s=30, edgecolor='white', linewidth=0.3)

ax.plot([0, 10], [0, 10], 'k--', linewidth=2, label='Perfect prediction')
r_overall, _ = pearsonr(y, method1_preds)
ax.set_xlabel('Actual Borg CR-10')
ax.set_ylabel('Predicted Borg CR-10')
ax.set_title(f'A. LOSO: Predicted vs Actual (r={r_overall:.2f})', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 7.5)

# Per-subject performance
ax = axes[1]
for subj in sorted(df['subject'].unique()):
    mask = df['subject'] == subj
    ax.scatter(y[mask], method1_preds[mask], c=SUBJECT_COLORS[subj], 
               label=f"{SUBJECT_LABELS[subj]}: r={per_subj_r_m1[subj]:.2f}", 
               alpha=0.6, s=30, edgecolor='white', linewidth=0.3)

ax.plot([0, 10], [0, 10], 'k--', linewidth=2)
ax.set_xlabel('Actual Borg CR-10')
ax.set_ylabel('Predicted Borg CR-10')
ax.set_title('B. Per-Subject Performance', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 7.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_method1_loso_raw.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 05_method1_loso_raw.png")

# =============================================================================
# PLOT 06: METHOD 3 - LOSO + CALIBRATION
# =============================================================================
print("Generating 06_method3_calibration.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall scatter
ax = axes[0]
for subj in sorted(df['subject'].unique()):
    mask = df['subject'] == subj
    ax.scatter(y[mask], method3_preds[mask], c=SUBJECT_COLORS[subj], 
               label=SUBJECT_LABELS[subj], alpha=0.6, s=30, edgecolor='white', linewidth=0.3)

ax.plot([0, 10], [0, 10], 'k--', linewidth=2, label='Perfect prediction')
r_overall, _ = pearsonr(y, method3_preds)
ax.set_xlabel('Actual Borg CR-10')
ax.set_ylabel('Predicted Borg CR-10')
ax.set_title(f'A. LOSO + Calibration (r={r_overall:.2f})', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 7.5)

# Per-subject performance
ax = axes[1]
for subj in sorted(df['subject'].unique()):
    mask = df['subject'] == subj
    ax.scatter(y[mask], method3_preds[mask], c=SUBJECT_COLORS[subj], 
               label=f"{SUBJECT_LABELS[subj]}: r={per_subj_r_m3[subj]:.2f}", 
               alpha=0.6, s=30, edgecolor='white', linewidth=0.3)

ax.plot([0, 10], [0, 10], 'k--', linewidth=2)
ax.set_xlabel('Actual Borg CR-10')
ax.set_ylabel('Predicted Borg CR-10')
ax.set_title('B. Per-Subject Performance (Calibrated)', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 7.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_method3_calibration.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 06_method3_calibration.png")

# =============================================================================
# PLOT 07: METHOD COMPARISON BAR CHART
# =============================================================================
print("Generating 07_method_comparison.png...")

# Compute metrics for all methods
methods = ['Method 1\n(Raw LOSO)', 'Method 3\n(Calibration)', 'Method 4\n(Within-Subj)']
predictions = [method1_preds, method3_preds, method4_preds]

correlations = []
maes = []
exact_accs = []
within1_accs = []

for preds in predictions:
    valid = ~np.isnan(preds)
    r, _ = pearsonr(y[valid], preds[valid])
    mae = np.mean(np.abs(y[valid] - preds[valid]))
    
    cat_true = np.array([to_cat(b) for b in y[valid]])
    cat_pred = np.array([to_cat(b) for b in np.round(preds[valid])])
    exact = np.mean(cat_true == cat_pred) * 100
    within1 = np.mean(np.abs(cat_true - cat_pred) <= 1) * 100
    
    correlations.append(r)
    maes.append(mae)
    exact_accs.append(exact)
    within1_accs.append(within1)

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

# Correlation
ax = axes[0]
colors = ['#e74c3c', '#3498db', '#2ecc71']
bars = ax.bar(methods, correlations, color=colors, edgecolor='black')
ax.set_ylabel('Pearson r')
ax.set_title('A. Correlation', fontweight='bold')
ax.set_ylim(0, 1)
for bar, val in zip(bars, correlations):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
            ha='center', va='bottom', fontweight='bold')

# MAE
ax = axes[1]
bars = ax.bar(methods, maes, color=colors, edgecolor='black')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('B. MAE (lower is better)', fontweight='bold')
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
            ha='center', va='bottom', fontweight='bold')

# Exact Category Accuracy
ax = axes[2]
bars = ax.bar(methods, exact_accs, color=colors, edgecolor='black')
ax.set_ylabel('Accuracy (%)')
ax.set_title('C. Exact Category Accuracy', fontweight='bold')
ax.set_ylim(0, 100)
for bar, val in zip(bars, exact_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
            ha='center', va='bottom', fontweight='bold')

# Within ±1 Category Accuracy
ax = axes[3]
bars = ax.bar(methods, within1_accs, color=colors, edgecolor='black')
ax.set_ylabel('Accuracy (%)')
ax.set_title('D. Within ±1 Category Accuracy', fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(90, color='green', linestyle='--', alpha=0.5, label='90% target')
for bar, val in zip(bars, within1_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
            ha='center', va='bottom', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_method_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 07_method_comparison.png")

# =============================================================================
# PLOT 08: CONFUSION MATRICES
# =============================================================================
print("Generating 08_confusion_matrices.png...")

from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

cat_labels = ['LOW', 'MOD', 'HIGH']

for idx, (name, preds) in enumerate([('Method 1 (Raw)', method1_preds), 
                                      ('Method 3 (Calibration)', method3_preds),
                                      ('Method 4 (Within-Subj)', method4_preds)]):
    ax = axes[idx]
    valid = ~np.isnan(preds)
    cat_true = np.array([to_cat(b) for b in y[valid]])
    cat_pred = np.array([to_cat(b) for b in np.round(preds[valid])])
    
    cm = confusion_matrix(cat_true, cat_pred, labels=[0, 1, 2])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=cat_labels, yticklabels=cat_labels,
                cbar=False, annot_kws={'size': 12})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Calculate accuracy
    acc = np.mean(cat_true == cat_pred) * 100
    ax.set_title(f'{name}\nAccuracy: {acc:.1f}%', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 08_confusion_matrices.png")

# =============================================================================
# PLOT 09: PER-SUBJECT IMPROVEMENT
# =============================================================================
print("Generating 09_per_subject_improvement.png...")

fig, ax = plt.subplots(figsize=(10, 6))

subjects = sorted(df['subject'].unique())
x = np.arange(len(subjects))
width = 0.35

# Method 1 correlations
m1_corrs = [per_subj_r_m1[s] for s in subjects]
# Method 3 correlations
m3_corrs = [per_subj_r_m3[s] for s in subjects]

bars1 = ax.bar(x - width/2, m1_corrs, width, label='Method 1 (Raw)', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, m3_corrs, width, label='Method 3 (Calibrated)', color='#3498db', edgecolor='black')

ax.set_ylabel('Pearson r')
ax.set_xlabel('Subject')
ax.set_title('Per-Subject Correlation Improvement with Calibration', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([SUBJECT_LABELS[s] for s in subjects])
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

# Add improvement arrows/text
for i, (m1, m3) in enumerate(zip(m1_corrs, m3_corrs)):
    improvement = m3 - m1
    if improvement > 0:
        ax.annotate(f'+{improvement:.2f}', xy=(i, max(m1, m3) + 0.05), 
                   ha='center', fontsize=9, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_per_subject_improvement.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 09_per_subject_improvement.png")

# =============================================================================
# PLOT 10: RESIDUAL ANALYSIS
# =============================================================================
print("Generating 10_residual_analysis.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (name, preds) in enumerate([('Method 1 (Raw)', method1_preds), 
                                      ('Method 3 (Calibration)', method3_preds),
                                      ('Method 4 (Within-Subj)', method4_preds)]):
    ax = axes[idx]
    valid = ~np.isnan(preds)
    residuals = y[valid] - preds[valid]
    
    for subj in sorted(df['subject'].unique()):
        mask = (df['subject'] == subj).values[valid]
        ax.scatter(preds[valid][mask], residuals[mask], c=SUBJECT_COLORS[subj], 
                   label=SUBJECT_LABELS[subj], alpha=0.5, s=20)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Borg')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(f'{name}', fontweight='bold')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_residual_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 10_residual_analysis.png")

# =============================================================================
# PLOT 11: CATEGORY ACCURACY BY CATEGORY
# =============================================================================
print("Generating 11_category_breakdown.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, preds) in enumerate([('Method 1 (Raw)', method1_preds), 
                                      ('Method 3 (Calibration)', method3_preds),
                                      ('Method 4 (Within-Subj)', method4_preds)]):
    ax = axes[idx]
    valid = ~np.isnan(preds)
    cat_true = np.array([to_cat(b) for b in y[valid]])
    cat_pred = np.array([to_cat(b) for b in np.round(preds[valid])])
    
    # Per-category accuracy
    accs = []
    for cat in [0, 1, 2]:
        mask = cat_true == cat
        if mask.sum() > 0:
            acc = np.mean(cat_true[mask] == cat_pred[mask]) * 100
        else:
            acc = 0
        accs.append(acc)
    
    colors = [CAT_COLORS['LOW'], CAT_COLORS['MOD'], CAT_COLORS['HIGH']]
    bars = ax.bar(['LOW', 'MOD', 'HIGH'], accs, color=colors, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
                ha='center', va='bottom', fontweight='bold')

plt.suptitle('Per-Category Accuracy Breakdown', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_category_breakdown.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 11_category_breakdown.png")

# =============================================================================
# PLOT 12: SUMMARY TABLE AS FIGURE
# =============================================================================
print("Generating 12_results_summary.png...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Create table data
table_data = [
    ['Method', 'Correlation (r)', 'MAE', 'Exact Acc.', '±1 Category', 'Dangerous'],
    ['Method 1 (Raw LOSO)', f'{correlations[0]:.2f}', f'{maes[0]:.2f}', f'{exact_accs[0]:.1f}%', f'{within1_accs[0]:.1f}%', f'{100-within1_accs[0]:.1f}%'],
    ['Method 3 (Calibration)', f'{correlations[1]:.2f}', f'{maes[1]:.2f}', f'{exact_accs[1]:.1f}%', f'{within1_accs[1]:.1f}%', f'{100-within1_accs[1]:.1f}%'],
    ['Method 4 (Within-Subject)', f'{correlations[2]:.2f}', f'{maes[2]:.2f}', f'{exact_accs[2]:.1f}%', f'{within1_accs[2]:.1f}%', f'{100-within1_accs[2]:.1f}%'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center',
                 colColours=['#f0f0f0']*6)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Color the rows
for i in range(1, 4):
    for j in range(6):
        cell = table[(i, j)]
        if i == 1:
            cell.set_facecolor('#ffcccc')  # Light red for Method 1
        elif i == 2:
            cell.set_facecolor('#cce5ff')  # Light blue for Method 3
        else:
            cell.set_facecolor('#ccffcc')  # Light green for Method 4

ax.set_title('Results Summary: 32 Selected Features, 5 Subjects, 1421 Windows\n', 
             fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_results_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 12_results_summary.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nPlots created:")
print("  01_data_overview.png         - Sample counts, Borg distribution, categories")
print("  02_borg_by_subject.png       - Borg boxplots per subject")
print("  03_feature_selection.png     - Selected features by modality + top correlations")
print("  04_quality_missingness.png   - Missing data analysis")
print("  05_method1_loso_raw.png      - Method 1: Raw LOSO scatter plots")
print("  06_method3_calibration.png   - Method 3: Calibrated scatter plots")
print("  07_method_comparison.png     - All methods comparison bar charts")
print("  08_confusion_matrices.png    - Confusion matrices for all methods")
print("  09_per_subject_improvement.png - Correlation improvement with calibration")
print("  10_residual_analysis.png     - Residual plots for all methods")
print("  11_category_breakdown.png    - Per-category accuracy breakdown")
print("  12_results_summary.png       - Summary table")

print("\n" + "="*70)
print("KEY RESULTS:")
print("="*70)
print(f"  Method 1 (Raw LOSO):      r={correlations[0]:.2f}, Exact={exact_accs[0]:.1f}%, ±1={within1_accs[0]:.1f}%")
print(f"  Method 3 (Calibration):   r={correlations[1]:.2f}, Exact={exact_accs[1]:.1f}%, ±1={within1_accs[1]:.1f}%")
print(f"  Method 4 (Within-Subj):   r={correlations[2]:.2f}, Exact={exact_accs[2]:.1f}%, ±1={within1_accs[2]:.1f}%")
print("="*70)
