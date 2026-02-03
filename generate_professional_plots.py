#!/usr/bin/env python3
"""
Professional Thesis Plots: SHAP, Feature Correlation Matrix, Interpretability
Clean formatting, publication-ready quality
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
SELECTED_FEATURES_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined/ml_expert_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
df_full = pd.read_csv(DATA_PATH)
df_full = df_full.dropna(subset=['borg'])
selected_features = pd.read_csv(SELECTED_FEATURES_PATH, header=None)[0].tolist()

feat_cols = [c for c in selected_features if c in df_full.columns]
print(f"Dataset: {len(df_full)} samples, {len(feat_cols)} features")

X = df_full[feat_cols].values
y = df_full['borg'].values

# Train model for SHAP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Short feature names for display
def shorten_name(name, max_len=25):
    """Shorten feature names for display"""
    name = name.replace('_dyn__', '.')
    name = name.replace('ppg_green_', 'ppg.g.')
    name = name.replace('ppg_red_', 'ppg.r.')
    name = name.replace('ppg_infra_', 'ppg.i.')
    name = name.replace('eda_stress_skin_', 'eda.ss.')
    name = name.replace('acc_x_', 'acc.x.')
    name = name.replace('acc_y_', 'acc.y.')
    name = name.replace('acc_z_', 'acc.z.')
    if len(name) > max_len:
        name = name[:max_len-2] + '..'
    return name

short_names = [shorten_name(f) for f in feat_cols]

# =============================================================================
# PLOT 1: FEATURE CORRELATION MATRIX (TOP 20)
# =============================================================================
print("\nGenerating 21_feature_corr_top20.png...")

# Get top 20 features by correlation with Borg
correlations = [(f, abs(pearsonr(df_full[f].fillna(0), df_full['borg'])[0])) for f in feat_cols]
correlations.sort(key=lambda x: x[1], reverse=True)
top20_feats = [f for f, r in correlations[:20]]
top20_short = [shorten_name(f, 20) for f in top20_feats]

fig, ax = plt.subplots(figsize=(12, 10))

corr_matrix = df_full[top20_feats].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr_matrix, 
            mask=mask,
            cmap=cmap, 
            center=0,
            square=True, 
            linewidths=0.5,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8},
            xticklabels=top20_short, 
            yticklabels=top20_short,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
            vmin=-1, vmax=1)

ax.set_title('Feature Correlation Matrix (Top 20 by |r| with Borg)\nAfter Redundancy Pruning (threshold = 0.90)', 
             fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '21_feature_corr_top20.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 21_feature_corr_top20.png")

# =============================================================================
# PLOT 2: FULL CORRELATION MATRIX (CLEAN)
# =============================================================================
print("Generating 22_feature_corr_full.png...")

fig, ax = plt.subplots(figsize=(14, 12))

corr_full = df_full[feat_cols].corr()
mask_full = np.triu(np.ones_like(corr_full, dtype=bool), k=1)

sns.heatmap(corr_full, 
            mask=mask_full,
            cmap=cmap, 
            center=0,
            square=True, 
            linewidths=0.3,
            xticklabels=short_names, 
            yticklabels=short_names,
            cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
            vmin=-1, vmax=1)

ax.set_title('Full Feature Correlation Matrix (34 Selected Features)', fontweight='bold', pad=15)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '22_feature_corr_full.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 22_feature_corr_full.png")

# =============================================================================
# SHAP PLOTS (Professional)
# =============================================================================
if SHAP_AVAILABLE:
    print("\nGenerating SHAP plots...")
    
    # Compute SHAP values
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    
    # =========================================================================
    # PLOT 3: SHAP BAR (Feature Importance)
    # =========================================================================
    print("Generating 23_shap_importance.png...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1][:20]  # Top 20
    
    # Color by modality
    colors = []
    for idx in sorted_idx:
        feat = feat_cols[idx]
        if feat.startswith('eda_'):
            colors.append('#3498db')  # Blue for EDA
        elif feat.startswith('acc_'):
            colors.append('#e74c3c')  # Red for IMU
        else:
            colors.append('#2ecc71')  # Green for PPG
    
    y_pos = np.arange(len(sorted_idx))
    bars = ax.barh(y_pos, mean_shap[sorted_idx], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_name(feat_cols[i], 30) for i in sorted_idx])
    ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
    ax.set_title('Feature Importance (SHAP Analysis)\nTop 20 Features by Impact on Borg Prediction', 
                 fontweight='bold', pad=15)
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='EDA (Electrodermal)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='IMU (Accelerometer)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='PPG (Cardiovascular)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '23_shap_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 23_shap_importance.png")
    
    # =========================================================================
    # PLOT 4: SHAP BEESWARM (Clean)
    # =========================================================================
    print("Generating 24_shap_beeswarm.png...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create SHAP explanation object with short names
    shap.summary_plot(shap_values, X_scaled, 
                      feature_names=short_names,
                      show=False, 
                      max_display=20,
                      plot_size=(12, 10))
    
    plt.title('SHAP Summary Plot\nFeature Values vs Impact on Borg Prediction', 
              fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('SHAP Value (Impact on Prediction)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '24_shap_beeswarm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 24_shap_beeswarm.png")
    
    # =========================================================================
    # PLOT 5: SHAP WATERFALL (3 Examples)
    # =========================================================================
    print("Generating 25_shap_waterfall.png...")
    
    y_pred = model.predict(X_scaled)
    
    # Find representative examples
    low_idx = np.where((y < 2) & (y_pred < 2))[0]
    mod_idx = np.where((y >= 3) & (y <= 4) & (y_pred >= 2.5) & (y_pred <= 4.5))[0]
    high_idx = np.where((y >= 5) & (y_pred >= 4.5))[0]
    
    if len(low_idx) > 0:
        low_idx = low_idx[np.argmin(np.abs(y_pred[low_idx] - 1))]
    else:
        low_idx = np.argmin(y_pred)
    
    if len(mod_idx) > 0:
        mod_idx = mod_idx[np.argmin(np.abs(y_pred[mod_idx] - 3.5))]
    else:
        mod_idx = np.argmin(np.abs(y_pred - 3.5))
    
    if len(high_idx) > 0:
        high_idx = high_idx[np.argmax(y_pred[high_idx])]
    else:
        high_idx = np.argmax(y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    examples = [
        (low_idx, 'LOW Effort Example', '#2ecc71'),
        (mod_idx, 'MODERATE Effort Example', '#f39c12'),
        (high_idx, 'HIGH Effort Example', '#e74c3c')
    ]
    
    for i, (idx, title, color) in enumerate(examples):
        plt.sca(axes[i])
        
        # Create explanation
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_scaled[idx],
            feature_names=short_names
        )
        
        shap.waterfall_plot(exp, max_display=12, show=False)
        
        axes[i].set_title(f'{title}\nActual: {y[idx]:.0f}, Predicted: {y_pred[idx]:.1f}', 
                          fontweight='bold', fontsize=12, color=color)
    
    plt.suptitle('SHAP Waterfall Plots: How Features Contribute to Individual Predictions', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '25_shap_waterfall.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 25_shap_waterfall.png")
    
    # =========================================================================
    # PLOT 6: SHAP DEPENDENCE (Top 4 Features)
    # =========================================================================
    print("Generating 26_shap_dependence.png...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    top_4_idx = np.argsort(mean_shap)[-4:][::-1]
    
    for i, feat_idx in enumerate(top_4_idx):
        ax = axes[i // 2, i % 2]
        
        # Scatter plot
        feat_values = X_scaled[:, feat_idx]
        shap_feat = shap_values[:, feat_idx]
        
        scatter = ax.scatter(feat_values, shap_feat, c=y, cmap='RdYlGn_r', 
                            alpha=0.6, s=20, edgecolor='white', linewidth=0.3)
        
        # Add trend line
        z = np.polyfit(feat_values, shap_feat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(feat_values.min(), feat_values.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.8, label='Trend')
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel(f'{short_names[feat_idx]} (Standardized)', fontweight='bold')
        ax.set_ylabel('SHAP Value', fontweight='bold')
        ax.set_title(f'{feat_cols[feat_idx]}', fontweight='bold', fontsize=11)
        
        # Correlation annotation
        r, _ = pearsonr(feat_values, shap_feat)
        ax.annotate(f'r = {r:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, label='Actual Borg Rating')
    
    plt.suptitle('SHAP Dependence Plots: How Feature Values Affect Predictions', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '26_shap_dependence.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ 26_shap_dependence.png")

# =============================================================================
# PLOT 7: MODEL COEFFICIENTS (Ridge)
# =============================================================================
print("Generating 27_ridge_coefficients.png...")

fig, ax = plt.subplots(figsize=(12, 10))

# Sort by absolute value
coef_sorted_idx = np.argsort(np.abs(model.coef_))[::-1][:25]

colors = []
for idx in coef_sorted_idx:
    feat = feat_cols[idx]
    if feat.startswith('eda_'):
        colors.append('#3498db')
    elif feat.startswith('acc_'):
        colors.append('#e74c3c')
    else:
        colors.append('#2ecc71')

y_pos = np.arange(len(coef_sorted_idx))
coef_vals = model.coef_[coef_sorted_idx]

# Separate positive and negative
pos_mask = coef_vals >= 0
neg_mask = coef_vals < 0

ax.barh(y_pos[pos_mask], coef_vals[pos_mask], color=[colors[i] for i in np.where(pos_mask)[0]], 
        edgecolor='black', linewidth=0.5, alpha=0.8)
ax.barh(y_pos[neg_mask], coef_vals[neg_mask], color=[colors[i] for i in np.where(neg_mask)[0]], 
        edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels([shorten_name(feat_cols[i], 35) for i in coef_sorted_idx])
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Ridge Coefficient (Standardized Features)', fontweight='bold')
ax.set_title('Ridge Regression Coefficients\nPositive = Increases Borg, Negative = Decreases Borg', 
             fontweight='bold', pad=15)
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='EDA'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='IMU'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='PPG')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '27_ridge_coefficients.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 27_ridge_coefficients.png")

# =============================================================================
# PLOT 8: FEATURE INTERPRETATION INFOGRAPHIC
# =============================================================================
print("Generating 28_feature_interpretation.png...")

fig, axes = plt.subplots(1, 3, figsize=(16, 10))

# Get top features per modality
eda_feats = [(f, pearsonr(df_full[f].fillna(0), df_full['borg'])[0]) for f in feat_cols if f.startswith('eda_')]
imu_feats = [(f, pearsonr(df_full[f].fillna(0), df_full['borg'])[0]) for f in feat_cols if f.startswith('acc_')]
ppg_feats = [(f, pearsonr(df_full[f].fillna(0), df_full['borg'])[0]) for f in feat_cols if f.startswith('ppg_')]

modalities = [
    ('EDA (Electrodermal Activity)', eda_feats, '#3498db', 
     'Measures sympathetic nervous\nsystem activation (stress/arousal)\n\n'
     '↑ Effort → ↑ Sweat gland activity\n→ ↑ Skin conductance'),
    ('IMU (Accelerometer)', imu_feats, '#e74c3c',
     'Measures physical movement\npatterns and intensity\n\n'
     '↑ Effort → ↑ Movement variability\n→ Complex motion patterns'),
    ('PPG (Photoplethysmography)', ppg_feats, '#2ecc71',
     'Measures blood volume changes\n(cardiovascular response)\n\n'
     '↑ Effort → Blood to muscles\n→ ↓ Peripheral blood volume')
]

for i, (title, feats, color, description) in enumerate(modalities):
    ax = axes[i]
    
    # Sort by absolute correlation
    feats.sort(key=lambda x: abs(x[1]), reverse=True)
    top_feats = feats[:6] if len(feats) > 6 else feats
    
    y_pos = np.arange(len(top_feats))
    correlations = [r for f, r in top_feats]
    
    # Color bars by sign
    bar_colors = [color if r > 0 else '#95a5a6' for r in correlations]
    
    bars = ax.barh(y_pos, correlations, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_name(f, 25) for f, r in top_feats], fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Correlation with Borg', fontweight='bold')
    ax.set_xlim(-0.5, 0.5)
    ax.invert_yaxis()
    
    # Title with background
    ax.set_title(f'{title}\n({len(feats)} features)', fontweight='bold', fontsize=12, 
                 color='white', backgroundcolor=color, pad=10)
    
    # Description box
    ax.text(0.5, -0.15, description, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Feature Interpretation by Modality\nHow Each Sensor Type Relates to Perceived Effort', 
             fontweight='bold', fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '28_feature_interpretation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ 28_feature_interpretation.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL PROFESSIONAL PLOTS GENERATED!")
print("="*70)
print(f"\nOutput: {OUTPUT_DIR}")
print("\nNew plots:")
print("  21_feature_corr_top20.png     - Top 20 features correlation matrix")
print("  22_feature_corr_full.png      - Full 34 features correlation matrix")
if SHAP_AVAILABLE:
    print("  23_shap_importance.png        - SHAP feature importance (bar)")
    print("  24_shap_beeswarm.png          - SHAP beeswarm summary")
    print("  25_shap_waterfall.png         - Waterfall for LOW/MOD/HIGH")
    print("  26_shap_dependence.png        - Top 4 feature dependence plots")
print("  27_ridge_coefficients.png     - Model coefficients")
print("  28_feature_interpretation.png - Modality interpretation infographic")
print("="*70)
