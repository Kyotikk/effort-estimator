#!/usr/bin/env python3
"""
Advanced Thesis Plots: SHAP Waterfall, Feature Explanations, Confusion Matrices, Train-Test Split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
SELECTED_FEATURES_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined/ml_expert_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

SUBJECT_COLORS = {
    'sim_elderly1': '#E69F00', 'sim_elderly2': '#56B4E9', 'sim_elderly3': '#009E73',
    'sim_elderly4': '#CC79A7', 'sim_elderly5': '#F0E442',
}
SUBJECT_LABELS = {
    'sim_elderly1': 'P1', 'sim_elderly2': 'P2', 'sim_elderly3': 'P3',
    'sim_elderly4': 'P4', 'sim_elderly5': 'P5',
}

# =============================================================================
# FEATURE EXPLANATIONS DICTIONARY
# =============================================================================
FEATURE_EXPLANATIONS = {
    # EDA Features
    'eda_scr_rate': 'Skin Conductance Response rate - frequency of stress/arousal events',
    'eda_scr_count': 'Number of skin conductance responses in window',
    'eda_stress_skin_max': 'Maximum stress-related skin conductance level',
    'eda_stress_skin_mean_abs_diff': 'Mean absolute difference in stress skin conductance - variability',
    'eda_cc_min': 'Minimum continuous conductance - baseline arousal',
    'eda_cc_kurtosis': 'Kurtosis of conductance - peakedness of distribution',
    'eda_phasic_energy': 'Energy in phasic (rapid) EDA component - acute stress',
    'eda_stress_skin_iqr': 'Interquartile range of stress skin conductance',
    
    # IMU Features (Accelerometer)
    'acc_x_dyn__cardinality': 'X-axis movement complexity - number of unique motion patterns',
    'acc_x_dyn__katz_fractal_dimension': 'X-axis fractal dimension - movement irregularity',
    'acc_x_dyn__sample_entropy': 'X-axis entropy - unpredictability of movement',
    'acc_x_dyn__variance_of_absolute_differences': 'X-axis jerkiness - sudden movement changes',
    'acc_x_dyn__max': 'Maximum X-axis acceleration - peak horizontal movement',
    'acc_y_dyn__sample_entropy': 'Y-axis entropy - vertical movement unpredictability',
    'acc_y_dyn__katz_fractal_dimension': 'Y-axis fractal dimension - vertical irregularity',
    'acc_y_dyn__tsallis_entropy': 'Y-axis Tsallis entropy - movement disorder',
    'acc_z_dyn__variance_of_absolute_differences': 'Z-axis jerkiness - forward/backward jerks',
    'acc_z_dyn__lower_complete_moment': 'Z-axis lower moment - sustained low movement',
    'acc_z_dyn__sum_of_absolute_changes': 'Z-axis total movement - cumulative activity',
    'acc_z_dyn__quantile_0.4': 'Z-axis 40th percentile acceleration',
    'acc_z_dyn__quantile_0.6': 'Z-axis 60th percentile acceleration',
    'acc_z_dyn__approximate_entropy_0.1': 'Z-axis approximate entropy - movement regularity',
    
    # PPG Features (Photoplethysmography)
    'ppg_green_max': 'Maximum green PPG signal - peak blood volume',
    'ppg_green_p90': '90th percentile green PPG - typical high blood volume',
    'ppg_green_p10': '10th percentile green PPG - typical low blood volume',
    'ppg_green_mad': 'Median absolute deviation - PPG signal variability',
    'ppg_green_p90_p10': 'PPG pulse amplitude (p90-p10) - circulation strength',
    'ppg_green_range': 'PPG signal range - cardiovascular response magnitude',
    'ppg_green_iqr': 'Interquartile range - middle 50% PPG variation',
    'ppg_green_rms': 'Root mean square - average PPG signal power',
    'ppg_green_n_peaks': 'Number of heartbeat peaks detected',
    'ppg_green_peak_quality': 'Quality/clarity of detected heartbeat peaks',
    'ppg_green_ddx_kurtosis': 'Second derivative kurtosis - waveform sharpness',
    'ppg_green_dx_kurtosis': 'First derivative kurtosis - rate of change peakedness',
    'ppg_green_crest_factor': 'Peak-to-RMS ratio - signal spikiness',
    'ppg_red_max': 'Maximum red PPG - deeper tissue blood volume',
    'ppg_red_zcr': 'Red PPG zero-crossing rate - signal oscillation frequency',
    'ppg_red_ddx_std': 'Red PPG acceleration variability',
    'ppg_red_dx_kurtosis': 'Red PPG velocity kurtosis',
    'ppg_red_n_peaks': 'Red PPG heartbeat count',
    'ppg_red_tke_std': 'Red PPG Teager-Kaiser energy std - signal dynamics',
    'ppg_red_ddx_kurtosis': 'Red PPG second derivative kurtosis',
    'ppg_infra_n_peaks': 'Infrared PPG peak count - deep tissue heartbeats',
    'ppg_infra_max': 'Maximum infrared PPG signal',
    'ppg_infra_dx_kurtosis': 'Infrared PPG velocity kurtosis',
    'ppg_infra_ddx_kurtosis': 'Infrared PPG acceleration kurtosis',
    'ppg_infra_mean_cross_rate': 'Infrared signal mean crossing rate',
}

# Category definitions
def to_cat(borg):
    if borg <= 2: return 0
    elif borg <= 4: return 1
    else: return 2

def to_cat_label(borg):
    if borg <= 2: return 'LOW'
    elif borg <= 4: return 'MOD'
    else: return 'HIGH'

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*70)
print("LOADING DATA...")
print("="*70)

df_full = pd.read_csv(DATA_PATH)
df_full = df_full.dropna(subset=['borg'])
selected_features = pd.read_csv(SELECTED_FEATURES_PATH, header=None)[0].tolist()

feat_cols = [c for c in selected_features if c in df_full.columns]
print(f"Dataset: {len(df_full)} samples, {len(feat_cols)} features")

X = df_full[feat_cols].values
y = df_full['borg'].values
subjects = df_full['subject'].values

# =============================================================================
# PLOT 13: FEATURE CORRELATION MATRIX (HEATMAP)
# =============================================================================
print("\nGenerating 13_feature_correlation_matrix.png...")

fig, ax = plt.subplots(figsize=(16, 14))

corr_matrix = df_full[feat_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Shorten feature names for display
short_names = []
for f in feat_cols:
    name = f.replace('_dyn__', '_')
    if len(name) > 25:
        name = name[:22] + '...'
    short_names.append(name)

sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=short_names, yticklabels=short_names,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            vmin=-1, vmax=1)

ax.set_title('Feature Correlation Matrix (34 Selected Features)\nAfter Redundancy Pruning (threshold=0.90)', 
             fontweight='bold', fontsize=14)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '13_feature_correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 13_feature_correlation_matrix.png")

# Verify no high correlations remain
upper_triangle = corr_matrix.where(mask.T)
max_corr = upper_triangle.abs().max().max()
print(f"  → Maximum remaining correlation: {max_corr:.3f}")

# =============================================================================
# PLOT 14: FEATURE EXPLANATIONS TABLE
# =============================================================================
print("Generating 14_feature_explanations.png...")

# Group features by modality
eda_feats = [(f, FEATURE_EXPLANATIONS.get(f, 'No description')) for f in feat_cols if f.startswith('eda_')]
imu_feats = [(f, FEATURE_EXPLANATIONS.get(f, 'No description')) for f in feat_cols if f.startswith('acc_')]
ppg_feats = [(f, FEATURE_EXPLANATIONS.get(f, 'No description')) for f in feat_cols if f.startswith('ppg_')]

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

def create_feature_table(ax, features, title, color):
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=14, loc='left', color=color)
    
    if not features:
        ax.text(0.5, 0.5, 'No features', ha='center', va='center')
        return
    
    # Create table
    cell_text = [[f[0], f[1][:80] + '...' if len(f[1]) > 80 else f[1]] for f in features]
    table = ax.table(cellText=cell_text,
                     colLabels=['Feature Name', 'Description'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.35, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor(color)
        table[(0, j)].set_text_props(color='white', fontweight='bold')

create_feature_table(axes[0], eda_feats, f'EDA Features ({len(eda_feats)} features) - Electrodermal Activity / Stress Response', '#3498db')
create_feature_table(axes[1], imu_feats, f'IMU Features ({len(imu_feats)} features) - Accelerometer / Physical Activity', '#e74c3c')
create_feature_table(axes[2], ppg_feats[:12], f'PPG Features ({len(ppg_feats)} features) - Photoplethysmography / Cardiovascular', '#2ecc71')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '14_feature_explanations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 14_feature_explanations.png")

# =============================================================================
# PLOT 15: TRAIN-TEST SPLIT VISUALIZATION
# =============================================================================
print("Generating 15_train_test_split.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. LOSO Split Illustration
ax = axes[0, 0]
subjects_list = sorted(df_full['subject'].unique())
colors = ['#2ecc71'] * 5  # All green (training)
colors[2] = '#e74c3c'  # One red (test)
bars = ax.barh(range(5), [100]*5, color=colors, edgecolor='black')
ax.set_yticks(range(5))
ax.set_yticklabels([SUBJECT_LABELS[s] for s in subjects_list])
ax.set_xlabel('Data %')
ax.set_title('A. Leave-One-Subject-Out (LOSO)\nMethods 1 & 3', fontweight='bold')
ax.legend([plt.Rectangle((0,0),1,1,fc='#2ecc71'), plt.Rectangle((0,0),1,1,fc='#e74c3c')],
          ['Train (4 subjects)', 'Test (1 subject)'], loc='lower right')

# B. Within-Subject Split
ax = axes[0, 1]
train_pct = 20
test_pct = 80
for i, subj in enumerate(subjects_list):
    ax.barh(i, train_pct, color='#2ecc71', edgecolor='black', label='Train' if i==0 else '')
    ax.barh(i, test_pct, left=train_pct, color='#e74c3c', edgecolor='black', label='Test' if i==0 else '')
ax.set_yticks(range(5))
ax.set_yticklabels([SUBJECT_LABELS[s] for s in subjects_list])
ax.set_xlabel('Data %')
ax.set_title('B. Within-Subject Split (20/80)\nMethod 4', fontweight='bold')
ax.legend(loc='lower right')

# C. Sample Distribution in LOSO
ax = axes[1, 0]
sample_counts = df_full.groupby('subject').size()
train_samples = []
test_samples = []
for subj in subjects_list:
    test_n = sample_counts[subj]
    train_n = sample_counts.sum() - test_n
    train_samples.append(train_n)
    test_samples.append(test_n)

x = np.arange(5)
width = 0.35
ax.bar(x - width/2, train_samples, width, label='Train', color='#2ecc71', edgecolor='black')
ax.bar(x + width/2, test_samples, width, label='Test', color='#e74c3c', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([f'{SUBJECT_LABELS[s]} left out' for s in subjects_list], rotation=45, ha='right')
ax.set_ylabel('Number of Samples')
ax.set_title('C. LOSO Sample Counts per Fold', fontweight='bold')
ax.legend()

# D. Category Balance Check
ax = axes[1, 1]
cat_labels = ['LOW\n(Borg 0-2)', 'MOD\n(Borg 3-4)', 'HIGH\n(Borg 5+)']
df_full['category'] = df_full['borg'].apply(to_cat_label)
cat_counts = df_full['category'].value_counts()[['LOW', 'MOD', 'HIGH']]

# Per subject
width = 0.15
for i, subj in enumerate(subjects_list):
    subj_cats = df_full[df_full['subject'] == subj]['category'].value_counts()
    counts = [subj_cats.get('LOW', 0), subj_cats.get('MOD', 0), subj_cats.get('HIGH', 0)]
    ax.bar(np.arange(3) + i*width, counts, width, label=SUBJECT_LABELS[subj], 
           color=SUBJECT_COLORS[subj], edgecolor='black')

ax.set_xticks(np.arange(3) + 2*width)
ax.set_xticklabels(cat_labels)
ax.set_ylabel('Sample Count')
ax.set_title('D. Category Distribution by Subject', fontweight='bold')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '15_train_test_split.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 15_train_test_split.png")

# =============================================================================
# PLOT 16: SHAP ANALYSIS (if available)
# =============================================================================
if SHAP_AVAILABLE:
    print("Generating SHAP plots...")
    
    # Train a model on all data for SHAP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # SHAP explainer
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    
    # 16a. SHAP Beeswarm Plot
    print("Generating 16_shap_beeswarm.png...")
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_scaled, feature_names=feat_cols, show=False, max_display=20)
    plt.title('SHAP Feature Importance (Beeswarm)\nImpact on Borg Prediction', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '16_shap_beeswarm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 16_shap_beeswarm.png")
    
    # 16b. SHAP Bar Plot (Mean |SHAP|)
    print("Generating 17_shap_bar.png...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=feat_cols, plot_type='bar', 
                      show=False, max_display=20)
    plt.title('Mean |SHAP| Value (Feature Importance)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '17_shap_bar.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 17_shap_bar.png")
    
    # 16c. SHAP Waterfall for specific predictions
    print("Generating 18_shap_waterfall.png...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find examples: LOW, MOD, HIGH predictions
    y_pred = model.predict(X_scaled)
    
    # LOW example (predicted ~1)
    low_idx = np.argmin(np.abs(y_pred - 1))
    # MOD example (predicted ~3.5)
    mod_idx = np.argmin(np.abs(y_pred - 3.5))
    # HIGH example (predicted ~6)
    high_idx = np.argmin(np.abs(y_pred - 6))
    
    for i, (idx, title) in enumerate([(low_idx, f'LOW Prediction\n(Actual: {y[low_idx]:.0f}, Pred: {y_pred[low_idx]:.1f})'),
                                       (mod_idx, f'MOD Prediction\n(Actual: {y[mod_idx]:.0f}, Pred: {y_pred[mod_idx]:.1f})'),
                                       (high_idx, f'HIGH Prediction\n(Actual: {y[high_idx]:.0f}, Pred: {y_pred[high_idx]:.1f})')]):
        plt.sca(axes[i])
        shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                              base_values=explainer.expected_value,
                                              data=X_scaled[idx],
                                              feature_names=feat_cols), 
                            max_display=10, show=False)
        axes[i].set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '18_shap_waterfall.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 18_shap_waterfall.png")
    
    # 16d. SHAP Dependence Plot for top feature
    print("Generating 19_shap_dependence.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Get top 3 features by mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_3_idx = np.argsort(mean_shap)[-3:][::-1]
    
    for i, feat_idx in enumerate(top_3_idx):
        plt.sca(axes[i])
        shap.dependence_plot(feat_idx, shap_values, X_scaled, feature_names=feat_cols,
                             interaction_index=None, show=False, ax=axes[i])
        axes[i].set_title(f'{feat_cols[feat_idx]}', fontweight='bold')
    
    plt.suptitle('SHAP Dependence Plots (Top 3 Features)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '19_shap_dependence.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 19_shap_dependence.png")

else:
    print("⚠️ Skipping SHAP plots (not installed)")

# =============================================================================
# PLOT 20: INTERPRETABILITY SUMMARY
# =============================================================================
print("Generating 20_interpretability_summary.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# A. Feature importance by correlation with Borg
ax = axes[0, 0]
correlations = []
for feat in feat_cols:
    r, _ = pearsonr(df_full[feat].fillna(0), df_full['borg'])
    correlations.append((feat, r))
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
top_15 = correlations[:15]

colors = []
for f, r in top_15:
    if f.startswith('eda_'): colors.append('#3498db')
    elif f.startswith('acc_'): colors.append('#e74c3c')
    else: colors.append('#2ecc71')

y_pos = np.arange(len(top_15))
ax.barh(y_pos, [abs(r) for f, r in top_15], color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f[0][:30] for f in top_15], fontsize=8)
ax.set_xlabel('|Correlation with Borg|')
ax.set_title('A. Top Features by Correlation', fontweight='bold')
ax.invert_yaxis()

# B. Feature type contribution
ax = axes[0, 1]
# Ridge coefficients
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0).fit(X_scaled, y)
coefs = np.abs(model.coef_)

eda_importance = sum(coefs[i] for i, f in enumerate(feat_cols) if f.startswith('eda_'))
imu_importance = sum(coefs[i] for i, f in enumerate(feat_cols) if f.startswith('acc_'))
ppg_importance = sum(coefs[i] for i, f in enumerate(feat_cols) if f.startswith('ppg_'))
total = eda_importance + imu_importance + ppg_importance

wedges, texts, autotexts = ax.pie([eda_importance, imu_importance, ppg_importance],
                                   labels=['EDA', 'IMU', 'PPG'],
                                   colors=['#3498db', '#e74c3c', '#2ecc71'],
                                   autopct='%1.1f%%', startangle=90,
                                   explode=[0.02, 0.02, 0.02])
ax.set_title('B. Model Weight by Modality', fontweight='bold')

# C. Ridge Coefficients (top features)
ax = axes[1, 0]
coef_sorted = sorted(zip(feat_cols, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:15]

colors = []
for f, c in coef_sorted:
    if f.startswith('eda_'): colors.append('#3498db')
    elif f.startswith('acc_'): colors.append('#e74c3c')
    else: colors.append('#2ecc71')

y_pos = np.arange(len(coef_sorted))
vals = [c for f, c in coef_sorted]
ax.barh(y_pos, vals, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels([f[0][:30] for f in coef_sorted], fontsize=8)
ax.set_xlabel('Ridge Coefficient')
ax.set_title('C. Ridge Regression Coefficients (Top 15)', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
ax.invert_yaxis()

# D. Physical interpretation summary
ax = axes[1, 1]
ax.axis('off')

summary_text = """
PHYSICAL INTERPRETATION OF KEY FEATURES

🟢 PPG Features (Cardiovascular):
• ppg_green_max: Peak blood volume → Higher effort = stronger pulse
• ppg_*_n_peaks: Heart rate indicator → Effort increases HR
• ppg_*_range: Pulse amplitude → Cardiovascular response

🔴 IMU Features (Movement):
• acc_*_sum_of_changes: Total activity level → More movement = more effort
• acc_*_entropy: Movement complexity → Fatigue increases irregularity
• acc_*_fractal_dimension: Movement pattern → Effort changes gait

🔵 EDA Features (Stress):
• eda_stress_skin_*: Sympathetic nervous system activation
• eda_scr_rate: Frequency of stress responses
• Higher EDA = higher physiological arousal/effort

CLINICAL RELEVANCE:
• LOW (0-2): Resting, minimal physiological response
• MOD (3-4): Light activity, moderate cardiovascular activation
• HIGH (5+): Significant effort, full sympathetic activation
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_title('D. Clinical Interpretation', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '20_interpretability_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 20_interpretability_summary.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL ADVANCED PLOTS GENERATED!")
print("="*70)
print(f"\nOutput: {OUTPUT_DIR}")
print("\nNew plots created:")
print("  13_feature_correlation_matrix.png - Correlation heatmap (verifies pruning worked)")
print("  14_feature_explanations.png       - Feature descriptions table")
print("  15_train_test_split.png           - LOSO and within-subject split visualization")
if SHAP_AVAILABLE:
    print("  16_shap_beeswarm.png              - SHAP feature importance")
    print("  17_shap_bar.png                   - Mean |SHAP| bar plot")
    print("  18_shap_waterfall.png             - Waterfall plots for LOW/MOD/HIGH")
    print("  19_shap_dependence.png            - Dependence plots for top features")
print("  20_interpretability_summary.png   - Clinical interpretation summary")
print("="*70)
