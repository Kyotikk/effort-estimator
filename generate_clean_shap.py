#!/usr/bin/env python3
"""
Single Clean SHAP Plot - No Overlap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

import shap

# Load data
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
SELECTED_FEATURES_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv')
OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined/ml_expert_plots')

df = pd.read_csv(DATA_PATH).dropna(subset=['borg'])
selected_features = pd.read_csv(SELECTED_FEATURES_PATH, header=None)[0].tolist()
feat_cols = [c for c in selected_features if c in df.columns]

X = df[feat_cols].values
y = df['borg'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Short names
def short_name(n):
    n = n.replace('ppg_green_', 'PPG.g.').replace('ppg_red_', 'PPG.r.').replace('ppg_infra_', 'PPG.i.')
    n = n.replace('acc_x_', 'ACC.x.').replace('acc_y_', 'ACC.y.').replace('acc_z_', 'ACC.z.')
    n = n.replace('eda_stress_skin_', 'EDA.')
    n = n.replace('_dyn__', '.')
    return n[:30]

short_names = [short_name(f) for f in feat_cols]

# SHAP
explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# =============================================================================
# PLOT 1: CLEAN BAR CHART (TOP 15)
# =============================================================================
print("Generating clean SHAP bar plot...")

fig, ax = plt.subplots(figsize=(10, 8))

mean_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[-15:]  # Top 15

y_pos = np.arange(len(sorted_idx))
values = mean_shap[sorted_idx]
names = [short_names[i] for i in sorted_idx]

# Color by modality
colors = []
for idx in sorted_idx:
    f = feat_cols[idx]
    if 'eda' in f: colors.append('#3498db')
    elif 'acc' in f: colors.append('#e74c3c')
    else: colors.append('#27ae60')

ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (SHAP)\nTop 15 Features by Impact on Borg Prediction', 
             fontsize=14, fontweight='bold', pad=15)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='EDA'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='IMU'),
    Patch(facecolor='#27ae60', edgecolor='black', label='PPG')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '31_shap_bar_clean.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 31_shap_bar_clean.png")

# =============================================================================
# PLOT 2: SINGLE WATERFALL (HIGH EFFORT EXAMPLE)
# =============================================================================
print("Generating single waterfall plot...")

y_pred = model.predict(X_scaled)

# Find a good HIGH effort example
high_idx = np.where((y >= 5) & (y_pred >= 4))[0]
if len(high_idx) > 0:
    idx = high_idx[np.argmax(y_pred[high_idx])]
else:
    idx = np.argmax(y_pred)

fig, ax = plt.subplots(figsize=(10, 10))

exp = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_scaled[idx],
    feature_names=short_names
)

shap.waterfall_plot(exp, max_display=15, show=False)

plt.title(f'SHAP Waterfall: HIGH Effort Example\nActual Borg: {y[idx]:.0f}, Predicted: {y_pred[idx]:.1f}', 
          fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '32_shap_waterfall_single.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 32_shap_waterfall_single.png")

# =============================================================================
# PLOT 3: BEESWARM (CLEAN)
# =============================================================================
print("Generating clean beeswarm plot...")

fig, ax = plt.subplots(figsize=(12, 10))

shap.summary_plot(shap_values, X_scaled, 
                  feature_names=short_names,
                  max_display=15,
                  show=False,
                  plot_size=None)

plt.title('SHAP Summary: Feature Value Impact\nRed=High feature value, Blue=Low feature value', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '33_shap_beeswarm_clean.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 33_shap_beeswarm_clean.png")

print("\n✅ Done! Clean SHAP plots saved:")
print("   31_shap_bar_clean.png      - Feature importance bar chart")
print("   32_shap_waterfall_single.png - Single waterfall example")
print("   33_shap_beeswarm_clean.png - Beeswarm summary")
