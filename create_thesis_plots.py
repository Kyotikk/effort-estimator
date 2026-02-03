#!/usr/bin/env python3
"""
Professional visualizations for thesis presentation:
- Feature importance
- SHAP values
- Confusion matrix (binned Borg)
- Comparison explanation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed, skipping SHAP plots")

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

# Load data
print("Loading data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        dfs.append(df)
        print(f"  P{i}: {len(df)} windows")

df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} windows")

# Define features
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]

# Get top 10 IMU features
def get_top_features(feature_cols, n_top=10):
    clean_df = df.dropna(subset=['borg'])
    valid_cols = [c for c in feature_cols if c in clean_df.columns]
    X = clean_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = clean_df['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.Series(rf.feature_importances_, index=valid_cols)
    return importance.nlargest(n_top).index.tolist(), importance.nlargest(n_top)

top_imu, imu_importance = get_top_features(imu_cols, 10)
top_ppg, ppg_importance = get_top_features(ppg_cols, 10)

# Train final model for visualizations
print("\nTraining model for visualizations...")
clean_df = df.dropna(subset=['borg'])
X_all = clean_df[top_imu].fillna(0).replace([np.inf, -np.inf], 0)
y_all = clean_df['borg'].values

rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_model.fit(X_all, y_all)

# LOSO predictions for confusion matrix
print("Running LOSO for predictions...")
all_y_true = []
all_y_pred = []
all_subjects = []

for test_subj in sorted(df['subject'].unique()):
    train_df = df[df['subject'] != test_subj].dropna(subset=['borg'])
    test_df = df[df['subject'] == test_subj].dropna(subset=['borg'])
    
    X_train = train_df[top_imu].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['borg'].values
    X_test = test_df[top_imu].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_subjects.extend([test_subj] * len(y_test))

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# ============================================================
# FIGURE 1: Professional Feature Importance
# ============================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

# IMU importance
ax = axes1[0]
imu_imp_sorted = imu_importance.sort_values(ascending=True)
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(imu_imp_sorted)))
bars = ax.barh(range(len(imu_imp_sorted)), imu_imp_sorted.values, color=colors, edgecolor='darkgreen', linewidth=0.5)
ax.set_yticks(range(len(imu_imp_sorted)))
labels = [f.replace('acc_', '').replace('_dyn__', ': ').replace('_', ' ') for f in imu_imp_sorted.index]
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Random Forest Importance', fontsize=12)
ax.set_title('Top 10 IMU Features\n(LOSO r = 0.54)', fontsize=14, fontweight='bold', color='#228B22')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# PPG importance
ax = axes1[1]
ppg_imp_sorted = ppg_importance.sort_values(ascending=True)
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(ppg_imp_sorted)))
bars = ax.barh(range(len(ppg_imp_sorted)), ppg_imp_sorted.values, color=colors, edgecolor='darkred', linewidth=0.5)
ax.set_yticks(range(len(ppg_imp_sorted)))
labels = [f.replace('ppg_green_', '').replace('ppg_', '').replace('_', ' ') for f in ppg_imp_sorted.index]
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Random Forest Importance', fontsize=12)
ax.set_title('Top 10 PPG Features\n(LOSO r = 0.33)', fontsize=14, fontweight='bold', color='#B22222')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/output/feature_importance_professional.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("✅ Saved: output/feature_importance_professional.png")

# ============================================================
# FIGURE 2: SHAP Values (if available)
# ============================================================
if HAS_SHAP:
    print("\nCalculating SHAP values...")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    # Use a sample for speed
    sample_idx = np.random.choice(len(X_all), min(500, len(X_all)), replace=False)
    X_sample = X_all.iloc[sample_idx]
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, 
                      feature_names=[f.replace('acc_', '').replace('_dyn__', ':\n') for f in top_imu])
    plt.title('SHAP Feature Importance (Top 10 IMU)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/output/shap_importance.png', 
                dpi=200, bbox_inches='tight', facecolor='white')
    print("✅ Saved: output/shap_importance.png")
    plt.close()
    
    # SHAP beeswarm
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False,
                      feature_names=[f.replace('acc_', '').replace('_dyn__', ':\n') for f in top_imu])
    plt.title('SHAP Values Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/output/shap_beeswarm.png', 
                dpi=200, bbox_inches='tight', facecolor='white')
    print("✅ Saved: output/shap_beeswarm.png")
    plt.close()

# ============================================================
# FIGURE 3: Confusion Matrix (Binned Borg)
# ============================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

# Bin Borg into categories
bins = [-0.1, 1.5, 3.5, 10]
labels_bin = ['Low (0-1)', 'Medium (2-3)', 'High (4+)']
y_true_binned = pd.cut(all_y_true, bins=bins, labels=labels_bin).astype(str)
y_pred_binned = pd.cut(all_y_pred, bins=bins, labels=labels_bin).astype(str)

cm = confusion_matrix(y_true_binned, y_pred_binned, labels=labels_bin)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
ax = axes4[0]
im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=1)
ax.set_xticks(range(len(labels_bin)))
ax.set_yticks(range(len(labels_bin)))
ax.set_xticklabels(labels_bin, fontsize=11)
ax.set_yticklabels(labels_bin, fontsize=11)
ax.set_xlabel('Predicted Effort', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Effort', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix (LOSO)\nTop 10 IMU Features', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(labels_bin)):
    for j in range(len(labels_bin)):
        color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.0%})', 
                ha='center', va='center', color=color, fontsize=11, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Proportion', fontsize=11)

# Predicted vs Actual scatter
ax = axes4[1]
for subj in sorted(set(all_subjects)):
    mask = np.array(all_subjects) == subj
    ax.scatter(all_y_true[mask], all_y_pred[mask], alpha=0.5, s=30, label=f'P{subj}', edgecolor='white', linewidth=0.3)

ax.plot([0, 6], [0, 6], 'k--', linewidth=2, alpha=0.7, label='Perfect')
ax.fill_between([0, 6], [0-1, 6-1], [0+1, 6+1], alpha=0.15, color='green', label='±1 Borg')
ax.set_xlabel('Actual Borg CR10', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Borg CR10', fontsize=12, fontweight='bold')
ax.set_title(f'Predicted vs Actual (LOSO)\nr = {pearsonr(all_y_true, all_y_pred)[0]:.2f}, MAE = {mean_absolute_error(all_y_true, all_y_pred):.2f}', 
             fontsize=14, fontweight='bold')
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 6.5)
ax.legend(loc='lower right', fontsize=9)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/output/confusion_matrix_professional.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("✅ Saved: output/confusion_matrix_professional.png")

# ============================================================
# FIGURE 4: Modality Comparison Bar Chart
# ============================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

approaches = ['Top 10 IMU', 'All IMU (30)', 'Top 10 PPG', 'All PPG (183)', 'IMU + PPG', 'All Combined']
loso_r = [0.54, 0.52, 0.33, 0.26, 0.48, 0.25]
colors = ['#228B22', '#32CD32', '#CD5C5C', '#B22222', '#DAA520', '#808080']

bars = ax5.bar(approaches, loso_r, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, val in zip(bars, loso_r):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
             ha='center', fontsize=12, fontweight='bold')

ax5.axhline(y=0, color='black', linewidth=0.5)
ax5.set_ylabel('LOSO Correlation (r)', fontsize=12, fontweight='bold')
ax5.set_title('Cross-Subject Generalization by Modality\n(Leave-One-Subject-Out Validation)', 
              fontsize=14, fontweight='bold')
ax5.set_ylim(0, 0.7)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/output/modality_comparison_professional.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("✅ Saved: output/modality_comparison_professional.png")

# ============================================================
# Print explanation
# ============================================================
print("\n" + "="*70)
print("EXPLANATION: YOUR SCREENSHOT vs CURRENT ML APPROACH")
print("="*70)
print("""
YOUR SCREENSHOT (r = 0.82):
─────────────────────────────────────────────────────────────
• Unit: ACTIVITY-LEVEL (one point per activity)
• Features: HAND-CRAFTED physiological formulas
  - HR Load = HR_delta × √duration (Stevens Power Law)
  - TRIMP Banister = training impulse formula
  - Combined = 0.8×HR + 0.2×MAD
• Evaluation: POOLED (all 5 subjects together)
• N samples: ~100 activities total

CURRENT ML APPROACH (r = 0.54):
─────────────────────────────────────────────────────────────
• Unit: WINDOW-LEVEL (5-second segments)  
• Features: RAW signal features (entropy, variance, etc.)
• Evaluation: LOSO (Leave-One-Subject-Out)
• N samples: ~1400 windows

WHY THE DIFFERENCE?
─────────────────────────────────────────────────────────────
1. POOLED vs LOSO:
   - Pooled inflates r because subject-specific patterns leak
   - LOSO tests true generalization to unseen subjects

2. ACTIVITY vs WINDOW:
   - Activity-level has fewer, less noisy points
   - Window-level is noisier but more granular

3. HAND-CRAFTED vs RAW:
   - HR formulas (TRIMP) encode physiological knowledge
   - Raw window features don't have duration/context

YOUR PPG/IMU FEATURES (from 5s windows):
─────────────────────────────────────────────────────────────
• Within-subject correlation: WEAK (r ≈ 0.15)
• These are NOT the same as the screenshot features!
• Screenshot used: HR_delta × √duration (activity-level)
• Your features: ppg_green_max, entropy, etc. (window-level)

BOTTOM LINE:
─────────────────────────────────────────────────────────────
Screenshot (r=0.82): Pooled, activity-level, hand-crafted features
Your LOSO (r=0.54): Honest, window-level, automated features

Both are valid but measure DIFFERENT THINGS.
For deployment to new subjects, LOSO r=0.54 is the realistic expectation.
""")

plt.show()
print("\nDone!")
