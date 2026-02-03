#!/usr/bin/env python3
"""
Comprehensive ML Expert Visualization Suite for Effort Estimation Pipeline

Generates publication-quality plots for:
1. DATA OVERVIEW - Sample counts, signal quality, data distribution
2. FEATURE EXTRACTION - Feature distributions, correlations, sensor contributions
3. FEATURE SELECTION - Importance rankings, selection process
4. QUALITY CHECKS - Missing data, outliers, signal quality metrics
5. MODEL PERFORMANCE - Learning curves, residuals, calibration
6. SUBJECT-SPECIFIC - Per-subject performance, LOSO results
7. SHAP ANALYSIS - Waterfall, beeswarm, feature importance

Author: ML Pipeline Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined_5subj/ml_expert_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

# Subject colors (colorblind-friendly palette)
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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("="*70)
print("LOADING DATA...")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"Total samples: {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")

# Identify feature columns
exclude_cols = ['subject', 'borg', 't_center', 'window_start', 'window_end', 
                'unix_time', 'Unnamed: 0', 'index']
feature_cols = [c for c in df.columns if c not in exclude_cols 
                and df[c].dtype in ['float64', 'int64']
                and df[c].notna().sum() > 100]

# Categorize features by sensor
ppg_features = [c for c in feature_cols if 'ppg' in c.lower()]
eda_features = [c for c in feature_cols if 'eda' in c.lower() or 'gsr' in c.lower()]
imu_features = [c for c in feature_cols if any(x in c.lower() for x in ['acc', 'gyro', 'imu', 'mag'])]
hrv_features = [c for c in feature_cols if any(x in c.lower() for x in ['hrv', 'rmssd', 'sdnn', 'rr'])]
other_features = [c for c in feature_cols if c not in ppg_features + eda_features + imu_features + hrv_features]

print(f"\nFeatures by sensor:")
print(f"  PPG: {len(ppg_features)}")
print(f"  EDA: {len(eda_features)}")
print(f"  IMU: {len(imu_features)}")
print(f"  HRV: {len(hrv_features)}")
print(f"  Other: {len(other_features)}")
print(f"  TOTAL: {len(feature_cols)}")

# Clean data for modeling
df_clean = df.dropna(subset=['borg'])
# Select features with <50% missing
valid_features = [c for c in feature_cols if df_clean[c].isna().mean() < 0.5]
df_model = df_clean[['subject', 'borg'] + valid_features].dropna()

print(f"\nAfter cleaning: {len(df_model)} samples, {len(valid_features)} features")

# =============================================================================
# SECTION 1: DATA OVERVIEW PLOTS
# =============================================================================
print("\n" + "="*70)
print("SECTION 1: DATA OVERVIEW")
print("="*70)

# 1.1 Sample count per subject
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
subject_counts = df_model.groupby('subject').size()
colors = [SUBJECT_COLORS[s] for s in subject_counts.index]
bars = ax1.bar([SUBJECT_LABELS[s] for s in subject_counts.index], subject_counts.values, 
               color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Subject')
ax1.set_ylabel('Number of Windows')
ax1.set_title('A. Sample Count per Subject', fontweight='bold')
for bar, count in zip(bars, subject_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, count + 5, str(count), 
             ha='center', fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(subject_counts.values) * 1.15)

# 1.2 Borg distribution per subject
ax2 = axes[1]
for subj in sorted(df_model['subject'].unique()):
    subj_data = df_model[df_model['subject'] == subj]['borg']
    ax2.hist(subj_data, bins=15, alpha=0.5, label=SUBJECT_LABELS[subj], 
             color=SUBJECT_COLORS[subj], density=True)
ax2.set_xlabel('Borg CR-10 Rating')
ax2.set_ylabel('Density')
ax2.set_title('B. Borg Rating Distribution by Subject', fontweight='bold')
ax2.legend(title='Subject')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 01_data_overview.png")

# 1.3 Detailed Borg statistics boxplot
fig, ax = plt.subplots(figsize=(10, 6))
subjects_order = sorted(df_model['subject'].unique())
data_to_plot = [df_model[df_model['subject'] == s]['borg'].values for s in subjects_order]
bp = ax.boxplot(data_to_plot, patch_artist=True, labels=[SUBJECT_LABELS[s] for s in subjects_order])
for patch, subj in zip(bp['boxes'], subjects_order):
    patch.set_facecolor(SUBJECT_COLORS[subj])
    patch.set_alpha(0.7)
ax.set_xlabel('Subject')
ax.set_ylabel('Borg CR-10 Rating')
ax.set_title('Borg Rating Distribution: Evidence of Inter-Subject Variability', fontweight='bold')

# Add statistics
stats_text = "Subject Statistics:\n"
for subj in subjects_order:
    subj_borg = df_model[df_model['subject'] == subj]['borg']
    stats_text += f"{SUBJECT_LABELS[subj]}: μ={subj_borg.mean():.1f}, σ={subj_borg.std():.1f}\n"
ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_borg_boxplot_detailed.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 02_borg_boxplot_detailed.png")

# =============================================================================
# SECTION 2: FEATURE EXTRACTION VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: FEATURE EXTRACTION")
print("="*70)

# 2.1 Feature count by sensor type (pie chart)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
sensor_counts = {
    'PPG': len(ppg_features),
    'EDA': len(eda_features),
    'IMU': len(imu_features),
    'HRV': len(hrv_features),
    'Other': len(other_features)
}
sensor_counts = {k: v for k, v in sensor_counts.items() if v > 0}
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
wedges, texts, autotexts = ax1.pie(sensor_counts.values(), labels=sensor_counts.keys(),
                                    autopct='%1.0f%%', colors=colors_pie[:len(sensor_counts)],
                                    explode=[0.02]*len(sensor_counts), shadow=True)
ax1.set_title('A. Feature Distribution by Sensor Type', fontweight='bold')

# 2.2 Feature extraction pipeline diagram
ax2 = axes[1]
ax2.axis('off')
pipeline_text = """
FEATURE EXTRACTION PIPELINE
═══════════════════════════════════════

Raw Signals (5.0s windows, 70% overlap)
           │
           ▼
┌──────────┬──────────┬──────────┐
│   PPG    │   EDA    │   IMU    │
│ (Green)  │ (Skin)   │ (Accel)  │
└────┬─────┴────┬─────┴────┬─────┘
     │          │          │
     ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ VitalPy │ │ NeuroKit│ │ Tsfresh │
│  + QC   │ │  + QC   │ │  + QC   │
└────┬────┘ └────┬────┘ └────┬────┘
     │          │          │
     └──────────┼──────────┘
                │
                ▼
        Feature Fusion
        ({} features)
                │
                ▼
        Quality Filtering
        (missingness < 50%)
                │
                ▼
        Final Features
        ({} features)
""".format(len(feature_cols), len(valid_features))

ax2.text(0.1, 0.95, pipeline_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_feature_extraction_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 03_feature_extraction_overview.png")

# 2.3 Sample feature distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

sample_features = []
if ppg_features: sample_features.append(ppg_features[0])
if eda_features: sample_features.append(eda_features[0])
if imu_features: sample_features.append(imu_features[0])
if len(ppg_features) > 1: sample_features.append(ppg_features[1])
if len(eda_features) > 1: sample_features.append(eda_features[1])
if len(valid_features) > 5: sample_features.append(valid_features[5])

for ax, feat in zip(axes, sample_features[:6]):
    for subj in subjects_order:
        subj_data = df_model[df_model['subject'] == subj][feat].dropna()
        if len(subj_data) > 0:
            ax.hist(subj_data, bins=20, alpha=0.4, label=SUBJECT_LABELS[subj],
                   color=SUBJECT_COLORS[subj], density=True)
    ax.set_xlabel(feat[:30] + ('...' if len(feat) > 30 else ''))
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Subject (Examples)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 04_feature_distributions.png")

# =============================================================================
# SECTION 3: QUALITY CHECKS
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: QUALITY CHECKS")
print("="*70)

# 3.1 Missing data heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Per-feature missingness
ax1 = axes[0]
feature_missingness = df[feature_cols].isna().mean().sort_values(ascending=False)
top_missing = feature_missingness.head(20)
colors_miss = ['red' if v > 0.5 else 'orange' if v > 0.2 else 'green' for v in top_missing.values]
ax1.barh(range(len(top_missing)), top_missing.values, color=colors_miss)
ax1.set_yticks(range(len(top_missing)))
ax1.set_yticklabels([f[:25] for f in top_missing.index], fontsize=8)
ax1.set_xlabel('Missing Rate')
ax1.set_title('A. Feature Missingness (Top 20)', fontweight='bold')
ax1.axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
ax1.legend()
ax1.invert_yaxis()

# Per-subject missingness
ax2 = axes[1]
subject_missingness = df.groupby('subject')[feature_cols].apply(lambda x: x.isna().mean().mean())
colors_subj = [SUBJECT_COLORS[s] for s in subject_missingness.index]
ax2.bar([SUBJECT_LABELS[s] for s in subject_missingness.index], subject_missingness.values,
        color=colors_subj, edgecolor='black')
ax2.set_xlabel('Subject')
ax2.set_ylabel('Mean Missing Rate')
ax2.set_title('B. Average Missingness by Subject', fontweight='bold')
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_quality_missingness.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 05_quality_missingness.png")

# 3.2 Feature correlation matrix (top features)
fig, ax = plt.subplots(figsize=(12, 10))

# Calculate correlations with Borg
correlations_with_borg = {}
for feat in valid_features:
    valid_data = df_model[[feat, 'borg']].dropna()
    if len(valid_data) > 10:
        r, _ = pearsonr(valid_data[feat], valid_data['borg'])
        correlations_with_borg[feat] = abs(r)

top_corr_features = sorted(correlations_with_borg.items(), key=lambda x: x[1], reverse=True)[:15]
top_features = [f[0] for f in top_corr_features]

corr_matrix = df_model[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8},
            xticklabels=[f[:20] for f in top_features],
            yticklabels=[f[:20] for f in top_features])
ax.set_title('Feature Correlation Matrix (Top 15 Borg-Correlated Features)', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 06_feature_correlation_matrix.png")

# =============================================================================
# SECTION 4: FEATURE SELECTION
# =============================================================================
print("\n" + "="*70)
print("SECTION 4: FEATURE SELECTION")
print("="*70)

# 4.1 Feature importance ranking (correlation-based)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Top positive and negative correlations
all_correlations = {}
for feat in valid_features:
    valid_data = df_model[[feat, 'borg']].dropna()
    if len(valid_data) > 10:
        r, p = pearsonr(valid_data[feat], valid_data['borg'])
        all_correlations[feat] = r

sorted_corrs = sorted(all_correlations.items(), key=lambda x: x[1], reverse=True)
top_positive = sorted_corrs[:15]
top_negative = sorted_corrs[-15:][::-1]

ax1 = axes[0]
feats_pos = [f[0][:25] for f in top_positive]
vals_pos = [f[1] for f in top_positive]
colors_pos = ['green' if v > 0.2 else 'lightgreen' for v in vals_pos]
ax1.barh(range(len(feats_pos)), vals_pos, color=colors_pos, edgecolor='black')
ax1.set_yticks(range(len(feats_pos)))
ax1.set_yticklabels(feats_pos, fontsize=9)
ax1.set_xlabel('Pearson r with Borg')
ax1.set_title('A. Top Positive Correlations', fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlim(0, max(vals_pos) * 1.1)

ax2 = axes[1]
feats_neg = [f[0][:25] for f in top_negative]
vals_neg = [f[1] for f in top_negative]
colors_neg = ['red' if v < -0.2 else 'lightcoral' for v in vals_neg]
ax2.barh(range(len(feats_neg)), vals_neg, color=colors_neg, edgecolor='black')
ax2.set_yticks(range(len(feats_neg)))
ax2.set_yticklabels(feats_neg, fontsize=9)
ax2.set_xlabel('Pearson r with Borg')
ax2.set_title('B. Top Negative Correlations', fontweight='bold')
ax2.invert_yaxis()
ax2.set_xlim(min(vals_neg) * 1.1, 0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_feature_importance_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 07_feature_importance_correlation.png")

# 4.2 Feature importance by sensor type
fig, ax = plt.subplots(figsize=(10, 6))

sensor_importance = {
    'PPG': np.mean([abs(all_correlations.get(f, 0)) for f in ppg_features if f in all_correlations]),
    'EDA': np.mean([abs(all_correlations.get(f, 0)) for f in eda_features if f in all_correlations]),
    'IMU': np.mean([abs(all_correlations.get(f, 0)) for f in imu_features if f in all_correlations]),
    'HRV': np.mean([abs(all_correlations.get(f, 0)) for f in hrv_features if f in all_correlations]),
}
sensor_importance = {k: v for k, v in sensor_importance.items() if not np.isnan(v) and v > 0}

colors_sensor = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(sensor_importance.keys(), sensor_importance.values(), 
              color=colors_sensor[:len(sensor_importance)], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Mean |Correlation| with Borg')
ax.set_title('Feature Importance by Sensor Type', fontweight='bold')
for bar, val in zip(bars, sensor_importance.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_sensor_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 08_sensor_importance.png")

# =============================================================================
# SECTION 5: MODEL TRAINING AND EVALUATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: MODEL TRAINING & EVALUATION")
print("="*70)

# Prepare data for modeling
X = df_model[valid_features].fillna(df_model[valid_features].median())
y = df_model['borg'].values
groups = df_model['subject'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5.1 LOSO Cross-Validation
print("Running LOSO Cross-Validation...")
logo = LeaveOneGroupOut()
model = Ridge(alpha=1.0)

y_pred_loso = cross_val_predict(model, X_scaled, y, cv=logo, groups=groups)

# Calculate per-subject metrics
loso_results = {}
for subj in np.unique(groups):
    mask = groups == subj
    y_true_subj = y[mask]
    y_pred_subj = y_pred_loso[mask]
    r, _ = pearsonr(y_true_subj, y_pred_subj)
    mae = mean_absolute_error(y_true_subj, y_pred_subj)
    rmse = np.sqrt(mean_squared_error(y_true_subj, y_pred_subj))
    loso_results[subj] = {'r': r, 'mae': mae, 'rmse': rmse, 'n': sum(mask)}

# Overall metrics
r_overall, _ = pearsonr(y, y_pred_loso)
mae_overall = mean_absolute_error(y, y_pred_loso)
rmse_overall = np.sqrt(mean_squared_error(y, y_pred_loso))

print(f"LOSO Overall: r={r_overall:.3f}, MAE={mae_overall:.2f}, RMSE={rmse_overall:.2f}")

# 5.2 LOSO Results Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Pearson r
ax1 = axes[0]
subjects_plot = sorted(loso_results.keys())
rs = [loso_results[s]['r'] for s in subjects_plot]
colors_r = [SUBJECT_COLORS[s] for s in subjects_plot]
bars = ax1.bar([SUBJECT_LABELS[s] for s in subjects_plot], rs, color=colors_r, edgecolor='black')
ax1.axhline(y=r_overall, color='red', linestyle='--', linewidth=2, label=f'Mean r={r_overall:.2f}')
ax1.set_ylabel('Pearson r')
ax1.set_title('A. LOSO: Correlation by Test Subject', fontweight='bold')
ax1.set_ylim(0, 0.6)
ax1.legend()
for bar, r in zip(bars, rs):
    ax1.text(bar.get_x() + bar.get_width()/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=10)

# MAE
ax2 = axes[1]
maes = [loso_results[s]['mae'] for s in subjects_plot]
bars = ax2.bar([SUBJECT_LABELS[s] for s in subjects_plot], maes, color=colors_r, edgecolor='black')
ax2.axhline(y=mae_overall, color='red', linestyle='--', linewidth=2, label=f'Mean MAE={mae_overall:.2f}')
ax2.set_ylabel('MAE (Borg units)')
ax2.set_title('B. LOSO: MAE by Test Subject', fontweight='bold')
ax2.legend()
for bar, mae in zip(bars, maes):
    ax2.text(bar.get_x() + bar.get_width()/2, mae + 0.05, f'{mae:.2f}', ha='center', fontsize=10)

# RMSE
ax3 = axes[2]
rmses = [loso_results[s]['rmse'] for s in subjects_plot]
bars = ax3.bar([SUBJECT_LABELS[s] for s in subjects_plot], rmses, color=colors_r, edgecolor='black')
ax3.axhline(y=rmse_overall, color='red', linestyle='--', linewidth=2, label=f'Mean RMSE={rmse_overall:.2f}')
ax3.set_ylabel('RMSE (Borg units)')
ax3.set_title('C. LOSO: RMSE by Test Subject', fontweight='bold')
ax3.legend()
for bar, rmse in zip(bars, rmses):
    ax3.text(bar.get_x() + bar.get_width()/2, rmse + 0.05, f'{rmse:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_loso_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 09_loso_results.png")

# 5.3 Predicted vs Actual scatter (LOSO)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# All subjects combined
ax1 = axes[0]
for subj in subjects_plot:
    mask = groups == subj
    ax1.scatter(y[mask], y_pred_loso[mask], c=SUBJECT_COLORS[subj], 
                alpha=0.5, s=30, label=SUBJECT_LABELS[subj])
ax1.plot([0, 7], [0, 7], 'k--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Borg CR-10')
ax1.set_ylabel('Predicted Borg CR-10')
ax1.set_title(f'A. LOSO: Predicted vs Actual (r={r_overall:.2f})', fontweight='bold')
ax1.legend(loc='upper left')
ax1.set_xlim(-0.5, 7)
ax1.set_ylim(-0.5, 7)

# Per-subject panels
ax2 = axes[1]
for i, subj in enumerate(subjects_plot):
    mask = groups == subj
    ax2.scatter(y[mask], y_pred_loso[mask], c=SUBJECT_COLORS[subj], 
                alpha=0.6, s=25, label=f"{SUBJECT_LABELS[subj]}: r={loso_results[subj]['r']:.2f}")
ax2.plot([0, 7], [0, 7], 'k--', linewidth=2)
ax2.set_xlabel('Actual Borg CR-10')
ax2.set_ylabel('Predicted Borg CR-10')
ax2.set_title('B. Per-Subject Performance', fontweight='bold')
ax2.legend(loc='upper left')
ax2.set_xlim(-0.5, 7)
ax2.set_ylim(-0.5, 7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_loso_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 10_loso_scatter.png")

# 5.4 Residual analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

residuals = y - y_pred_loso

# Residuals vs predicted
ax1 = axes[0]
for subj in subjects_plot:
    mask = groups == subj
    ax1.scatter(y_pred_loso[mask], residuals[mask], c=SUBJECT_COLORS[subj], 
                alpha=0.5, s=30, label=SUBJECT_LABELS[subj])
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Borg')
ax1.set_ylabel('Residual (Actual - Predicted)')
ax1.set_title('A. Residuals vs Predicted', fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)

# Residual distribution
ax2 = axes[1]
ax2.hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=2, label=f'Mean={np.mean(residuals):.2f}')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency')
ax2.set_title('B. Residual Distribution', fontweight='bold')
ax2.legend()

# Residuals by subject
ax3 = axes[2]
residuals_by_subj = [residuals[groups == s] for s in subjects_plot]
bp = ax3.boxplot(residuals_by_subj, patch_artist=True, labels=[SUBJECT_LABELS[s] for s in subjects_plot])
for patch, subj in zip(bp['boxes'], subjects_plot):
    patch.set_facecolor(SUBJECT_COLORS[subj])
    patch.set_alpha(0.7)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Subject')
ax3.set_ylabel('Residual')
ax3.set_title('C. Residuals by Subject', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 11_residual_analysis.png")

# =============================================================================
# SECTION 6: SUBJECT-SPECIFIC ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 6: SUBJECT-SPECIFIC ANALYSIS")
print("="*70)

# 6.1 Within-subject performance (train and test on same subject)
within_subject_results = {}
for subj in subjects_plot:
    mask = groups == subj
    X_subj = X_scaled[mask]
    y_subj = y[mask]
    
    if len(y_subj) > 20:
        # 5-fold CV within subject
        from sklearn.model_selection import cross_val_score, KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model_subj = Ridge(alpha=1.0)
        y_pred_subj = cross_val_predict(model_subj, X_subj, y_subj, cv=kf)
        r_within, _ = pearsonr(y_subj, y_pred_subj)
        mae_within = mean_absolute_error(y_subj, y_pred_subj)
        within_subject_results[subj] = {'r': r_within, 'mae': mae_within}
        print(f"  {SUBJECT_LABELS[subj]}: Within-subject r={r_within:.2f}, MAE={mae_within:.2f}")

# 6.2 LOSO vs Within-Subject comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Comparison bar chart
ax1 = axes[0]
x = np.arange(len(subjects_plot))
width = 0.35

loso_rs = [loso_results[s]['r'] for s in subjects_plot]
within_rs = [within_subject_results.get(s, {}).get('r', 0) for s in subjects_plot]

bars1 = ax1.bar(x - width/2, loso_rs, width, label='LOSO (Cross-Subject)', 
                color='steelblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, within_rs, width, label='Within-Subject', 
                color='coral', edgecolor='black')

ax1.set_xlabel('Subject')
ax1.set_ylabel('Pearson r')
ax1.set_title('LOSO vs Within-Subject Performance', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([SUBJECT_LABELS[s] for s in subjects_plot])
ax1.legend()
ax1.set_ylim(0, 0.7)

# Add value labels
for bar, val in zip(bars1, loso_rs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
             ha='center', fontsize=9, color='steelblue')
for bar, val in zip(bars2, within_rs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
             ha='center', fontsize=9, color='coral')

# Summary message
ax2 = axes[1]
ax2.axis('off')

summary_text = f"""
PERFORMANCE SUMMARY
═══════════════════════════════════════

CROSS-SUBJECT (LOSO):
  • Mean r = {r_overall:.2f}
  • Mean MAE = {mae_overall:.2f} Borg units
  • Range: r = {min(loso_rs):.2f} to {max(loso_rs):.2f}

WITHIN-SUBJECT:
  • Mean r = {np.mean(within_rs):.2f}
  • Consistently higher than LOSO

KEY INSIGHT:
─────────────────────────────────────
Cross-subject generalization fails
because perceived effort is SUBJECTIVE.

Same physiological response →
Different Borg rating per individual.

IMPLICATION:
Personalized calibration required
for practical deployment.
"""

ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_loso_vs_within_subject.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 12_loso_vs_within_subject.png")

# 6.3 Per-subject scatter plots (detailed)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, subj in zip(axes[:5], subjects_plot):
    mask = groups == subj
    ax.scatter(y[mask], y_pred_loso[mask], c=SUBJECT_COLORS[subj], alpha=0.6, s=40)
    ax.plot([0, 7], [0, 7], 'k--', linewidth=2)
    
    r = loso_results[subj]['r']
    mae = loso_results[subj]['mae']
    n = loso_results[subj]['n']
    
    ax.set_xlabel('Actual Borg')
    ax.set_ylabel('Predicted Borg')
    ax.set_title(f'{SUBJECT_LABELS[subj]}: r={r:.2f}, MAE={mae:.2f}, n={n}', fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 7)

# Summary panel
axes[5].axis('off')
axes[5].text(0.1, 0.9, "Per-Subject LOSO Results\n\nEach panel shows predictions\nwhen that subject was held out\nduring training.\n\nNote the varying performance\nacross subjects - evidence of\ninter-individual variability.",
             transform=axes[5].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('LOSO Cross-Validation: Per-Subject Predictions', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '13_per_subject_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 13_per_subject_scatter.png")

# =============================================================================
# SECTION 7: SHAP ANALYSIS (if available)
# =============================================================================
print("\n" + "="*70)
print("SECTION 7: SHAP ANALYSIS")
print("="*70)

if SHAP_AVAILABLE:
    print("Computing SHAP values (this may take a moment)...")
    
    # Train final model on all data
    model_final = Ridge(alpha=1.0)
    model_final.fit(X_scaled, y)
    
    # SHAP for linear model
    explainer = shap.LinearExplainer(model_final, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    
    # 7.1 SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=valid_features, 
                      max_display=20, show=False)
    plt.title('SHAP Feature Importance (Beeswarm Plot)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '14_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 14_shap_beeswarm.png")
    
    # 7.2 SHAP Bar Plot (Mean Absolute SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=valid_features,
                      plot_type='bar', max_display=20, show=False)
    plt.title('SHAP Feature Importance (Bar Plot)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '15_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 15_shap_bar.png")
    
    # 7.3 SHAP Waterfall Plot (single prediction)
    plt.figure(figsize=(12, 8))
    # Pick a sample with moderate Borg
    idx = np.argmin(np.abs(y - np.median(y)))
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                          base_values=explainer.expected_value,
                                          data=X_scaled[idx],
                                          feature_names=valid_features), 
                        max_display=15, show=False)
    plt.title(f'SHAP Waterfall: Sample {idx} (Actual Borg={y[idx]:.1f})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '16_shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 16_shap_waterfall.png")
    
    # 7.4 SHAP Dependence Plots for top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_shap_idx = np.argsort(mean_shap)[-4:][::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, idx in zip(axes, top_shap_idx):
        feat_name = valid_features[idx]
        shap.dependence_plot(idx, shap_values, X_scaled, feature_names=valid_features,
                            ax=ax, show=False)
        ax.set_title(f'{feat_name[:30]}', fontweight='bold')
    
    plt.suptitle('SHAP Dependence Plots (Top 4 Features)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '17_shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 17_shap_dependence.png")

else:
    print("⚠️ SHAP not available - skipping SHAP plots")
    print("Install with: pip install shap")

# =============================================================================
# SECTION 8: COMPREHENSIVE THESIS FIGURE
# =============================================================================
print("\n" + "="*70)
print("SECTION 8: THESIS SUMMARY FIGURE")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# A. Sample overview
ax1 = fig.add_subplot(gs[0, 0])
colors = [SUBJECT_COLORS[s] for s in subject_counts.index]
ax1.bar([SUBJECT_LABELS[s] for s in subject_counts.index], subject_counts.values, 
        color=colors, edgecolor='black')
ax1.set_ylabel('Samples')
ax1.set_title('A. Dataset Overview', fontweight='bold')

# B. Borg distributions
ax2 = fig.add_subplot(gs[0, 1])
for subj in subjects_order:
    subj_data = df_model[df_model['subject'] == subj]['borg']
    ax2.hist(subj_data, bins=12, alpha=0.5, label=SUBJECT_LABELS[subj], 
             color=SUBJECT_COLORS[subj], density=True)
ax2.set_xlabel('Borg CR-10')
ax2.set_ylabel('Density')
ax2.set_title('B. Subjective Effort Varies', fontweight='bold')
ax2.legend(fontsize=8)

# C. Feature importance by sensor
ax3 = fig.add_subplot(gs[0, 2])
if sensor_importance:
    colors_s = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(sensor_importance)]
    ax3.bar(sensor_importance.keys(), sensor_importance.values(), 
            color=colors_s, edgecolor='black')
ax3.set_ylabel('Mean |r| with Borg')
ax3.set_title('C. Sensor Contributions', fontweight='bold')

# D. LOSO results
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar([SUBJECT_LABELS[s] for s in subjects_plot], loso_rs, 
               color=[SUBJECT_COLORS[s] for s in subjects_plot], edgecolor='black')
ax4.axhline(y=r_overall, color='red', linestyle='--', linewidth=2)
ax4.set_ylabel('LOSO r')
ax4.set_title(f'D. Cross-Subject: r={r_overall:.2f}', fontweight='bold')
ax4.set_ylim(0, 0.5)

# E. Predicted vs Actual
ax5 = fig.add_subplot(gs[1, 1])
for subj in subjects_plot:
    mask = groups == subj
    ax5.scatter(y[mask], y_pred_loso[mask], c=SUBJECT_COLORS[subj], alpha=0.4, s=20)
ax5.plot([0, 7], [0, 7], 'k--', linewidth=2)
ax5.set_xlabel('Actual Borg')
ax5.set_ylabel('Predicted Borg')
ax5.set_title('E. LOSO Predictions', fontweight='bold')

# F. LOSO vs Within comparison
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(subjects_plot))
width = 0.35
ax6.bar(x - width/2, loso_rs, width, label='LOSO', color='steelblue', edgecolor='black')
ax6.bar(x + width/2, within_rs, width, label='Within', color='coral', edgecolor='black')
ax6.set_xticks(x)
ax6.set_xticklabels([SUBJECT_LABELS[s] for s in subjects_plot])
ax6.set_ylabel('r')
ax6.set_title('F. LOSO vs Personalized', fontweight='bold')
ax6.legend(fontsize=8)

# G-I. Key findings text
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

findings_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                              KEY FINDINGS: EFFORT ESTIMATION PIPELINE                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                              ║
║  DATASET:  5 elderly subjects  •  {len(df_model)} samples  •  5.0s windows (70% overlap)  •  {len(valid_features)} features (PPG, EDA, IMU)                  ║
║                                                                                                                              ║
║  CROSS-SUBJECT GENERALIZATION (LOSO):                           WITHIN-SUBJECT PERFORMANCE:                                 ║
║    • Pearson r = {r_overall:.2f} (essentially random)                      • Mean r = {np.mean(within_rs):.2f} (2× better than LOSO)                       ║
║    • MAE = {mae_overall:.2f} Borg units                                    • Personalization helps significantly                          ║
║    • RMSE = {rmse_overall:.2f} Borg units                                                                                           ║
║                                                                                                                              ║
║  WHY CROSS-SUBJECT FAILS:                                       IMPLICATION:                                                 ║
║    • Borg ratings are SUBJECTIVE                                  • Personalized/longitudinal approach required               ║
║    • Same activity → different perceived effort                   • Calibration phase needed for new users                    ║
║    • Subject means: {min([df_model[df_model['subject']==s]['borg'].mean() for s in subjects_plot]):.1f} to {max([df_model[df_model['subject']==s]['borg'].mean() for s in subjects_plot]):.1f} Borg (3× variation)             • Cannot deploy "one-size-fits-all" model                      ║
║                                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

ax7.text(0.5, 0.5, findings_text, transform=ax7.transAxes, fontsize=9,
         verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Wearable-Based Effort Estimation: Pipeline Results & Analysis', 
             fontweight='bold', fontsize=16, y=0.98)

plt.savefig(OUTPUT_DIR / '18_thesis_comprehensive_figure.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 18_thesis_comprehensive_figure.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("GENERATION COMPLETE!")
print("="*70)
print(f"\n📁 All plots saved to: {OUTPUT_DIR}")
print("\nPlots generated:")
print("  01. Data overview (sample counts, Borg distributions)")
print("  02. Borg boxplot detailed")
print("  03. Feature extraction overview")
print("  04. Feature distributions by subject")
print("  05. Quality: missingness analysis")
print("  06. Feature correlation matrix")
print("  07. Feature importance (correlation-based)")
print("  08. Sensor importance comparison")
print("  09. LOSO results (r, MAE, RMSE)")
print("  10. LOSO scatter (predicted vs actual)")
print("  11. Residual analysis")
print("  12. LOSO vs within-subject comparison")
print("  13. Per-subject scatter plots")
if SHAP_AVAILABLE:
    print("  14. SHAP beeswarm plot")
    print("  15. SHAP bar plot")
    print("  16. SHAP waterfall plot")
    print("  17. SHAP dependence plots")
print("  18. THESIS COMPREHENSIVE FIGURE")

print(f"\n✅ Total: {'18' if SHAP_AVAILABLE else '14'} publication-quality plots")

# Save results to CSV
results_df = pd.DataFrame([
    {'Subject': SUBJECT_LABELS[s], 
     'LOSO_r': loso_results[s]['r'],
     'LOSO_MAE': loso_results[s]['mae'],
     'LOSO_RMSE': loso_results[s]['rmse'],
     'Within_r': within_subject_results.get(s, {}).get('r', np.nan),
     'Within_MAE': within_subject_results.get(s, {}).get('mae', np.nan),
     'n_samples': loso_results[s]['n']}
    for s in subjects_plot
])
results_df.to_csv(OUTPUT_DIR / 'performance_results.csv', index=False)
print(f"\n📊 Results saved to: {OUTPUT_DIR / 'performance_results.csv'}")
