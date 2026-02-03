#!/usr/bin/env python3
"""
Visualization of modality comparison results + explanation of feature selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

# Load data
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Define modalities
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]
eda_cols = [c for c in df.columns if 'eda_' in c]

def loso_evaluate(feature_cols):
    """Run LOSO and return per-subject results."""
    results = {}
    for test_subj in sorted(df['subject'].unique()):
        train_df = df[df['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df[df['subject'] == test_subj].dropna(subset=['borg'])
        
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if len(y_test) > 2 and np.std(y_test) > 0:
            r, _ = pearsonr(y_test, y_pred)
            results[f'P{test_subj}'] = r
    
    return results

def get_top_features(feature_cols, n_top=10):
    """Get top features by RF importance."""
    clean_df = df.dropna(subset=['borg'])
    valid_cols = [c for c in feature_cols if c in clean_df.columns]
    X = clean_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = clean_df['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.Series(rf.feature_importances_, index=valid_cols)
    return importance.nlargest(n_top).index.tolist(), importance.nlargest(n_top)

# Get top features
top_imu, imu_importance = get_top_features(imu_cols, 10)
top_ppg, ppg_importance = get_top_features(ppg_cols, 10)
top_eda, eda_importance = get_top_features(eda_cols, 5)

# Run experiments
print("Running LOSO evaluations...")
experiments = {
    'Top 10 IMU': top_imu,
    'All IMU (30)': imu_cols,
    'Top 10 PPG': top_ppg,
    'All PPG (183)': ppg_cols,
    'Top 5 EDA': top_eda,
    'IMU + PPG': top_imu + top_ppg,
    'All Combined': imu_cols + ppg_cols + eda_cols,
}

all_results = {}
for name, cols in experiments.items():
    all_results[name] = loso_evaluate(cols)
    mean_r = np.mean(list(all_results[name].values()))
    print(f"  {name}: r = {mean_r:.3f}")

# ============================================================
# FIGURE 1: Bar chart comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Mean LOSO r by approach
ax = axes[0]
approaches = ['Top 10 IMU', 'All IMU (30)', 'IMU + PPG', 'Top 10 PPG', 'All PPG (183)', 'All Combined']
means = [np.mean(list(all_results[a].values())) for a in approaches]
colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b', '#95a5a6']

bars = ax.barh(approaches, means, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('LOSO Correlation (r)', fontsize=12)
ax.set_title('Cross-Subject Generalization by Approach', fontsize=14, fontweight='bold')
ax.set_xlim(-0.1, 0.7)

# Add value labels
for bar, val in zip(bars, means):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
            va='center', fontsize=11, fontweight='bold')

# Right: Per-subject breakdown for top approaches
ax = axes[1]
subjects = ['P1', 'P2', 'P3', 'P4', 'P5']
x = np.arange(len(subjects))
width = 0.25

top_approaches = ['Top 10 IMU', 'Top 10 PPG', 'All Combined']
colors_sub = ['#2ecc71', '#e74c3c', '#95a5a6']

for i, approach in enumerate(top_approaches):
    vals = [all_results[approach].get(s, 0) for s in subjects]
    ax.bar(x + i*width, vals, width, label=approach, color=colors_sub[i], edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('LOSO Correlation (r)', fontsize=12)
ax.set_title('Per-Subject Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(subjects)
ax.legend(loc='upper right')
ax.set_ylim(-0.2, 0.8)

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/output/modality_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ Saved: output/modality_comparison.png")

# ============================================================
# FIGURE 2: Feature importance visualization
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Left: Top IMU features
ax = axes2[0]
imu_importance_sorted = imu_importance.sort_values(ascending=True)
colors_imu = ['#2ecc71'] * len(imu_importance_sorted)
ax.barh(range(len(imu_importance_sorted)), imu_importance_sorted.values, color=colors_imu, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(imu_importance_sorted)))
ax.set_yticklabels([f.replace('acc_', '').replace('_dyn__', '\n') for f in imu_importance_sorted.index], fontsize=9)
ax.set_xlabel('RF Feature Importance', fontsize=11)
ax.set_title('Top 10 IMU Features\n(LOSO r = 0.54)', fontsize=13, fontweight='bold', color='#27ae60')

# Right: Top PPG features
ax = axes2[1]
ppg_importance_sorted = ppg_importance.sort_values(ascending=True)
colors_ppg = ['#e74c3c'] * len(ppg_importance_sorted)
ax.barh(range(len(ppg_importance_sorted)), ppg_importance_sorted.values, color=colors_ppg, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(ppg_importance_sorted)))
ax.set_yticklabels([f.replace('ppg_green_', 'green_').replace('ppg_', '') for f in ppg_importance_sorted.index], fontsize=9)
ax.set_xlabel('RF Feature Importance', fontsize=11)
ax.set_title('Top 10 PPG Features\n(LOSO r = 0.33)', fontsize=13, fontweight='bold', color='#c0392b')

fig2.suptitle('Feature Selection: RF Importance on Training Data', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/output/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: output/feature_importance.png")

# ============================================================
# FIGURE 3: Explanation diagram
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.axis('off')

explanation = """
HOW FEATURE SELECTION WORKS IN THIS PIPELINE
═══════════════════════════════════════════════════════════════════════════════════

STEP 1: TRAIN RF ON POOLED DATA
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Train RandomForest on ALL subjects together                                     │
│  → Get feature importance scores                                                 │
│  → Select Top N features (e.g., Top 10 IMU, Top 10 PPG)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
STEP 2: EVALUATE WITH LOSO (Leave-One-Subject-Out)
┌─────────────────────────────────────────────────────────────────────────────────┐
│  For each subject:                                                               │
│    • Train on 4 subjects                                                         │
│    • Test on 1 held-out subject                                                  │
│    • Measure correlation (r) on held-out subject                                │
│  → Mean r across all folds = GENERALIZATION PERFORMANCE                         │
└─────────────────────────────────────────────────────────────────────────────────┘

WHY IMU GENERALIZES BUT PPG DOESN'T
═══════════════════════════════════════════════════════════════════════════════════

IMU (Accelerometer):                        PPG (Photoplethysmography):
✓ Motion patterns similar across people     ✗ Absolute values differ per person
✓ Walking = similar acceleration for all    ✗ Person A: PPG max = 150,000
✓ Features capture MOVEMENT INTENSITY       ✗ Person B: PPG max = 200,000
                                            ✗ Model learns WHO, not EFFORT

WILL THE PIPELINE ADAPT WITH MORE DATA?
═══════════════════════════════════════════════════════════════════════════════════

YES! The pipeline is DATA-DRIVEN:

Current (5 subjects):                       With more subjects (e.g., 20):
• Top features: mostly IMU                  • Feature ranking may change
• PPG doesn't generalize                    • PPG might become useful IF:
• IMU captures cross-subject patterns         - More diverse training data
                                              - Per-subject normalization added
                                              - Relative HR features (not absolute)

The pipeline will AUTOMATICALLY re-rank features based on new data.
Current conclusion: IMU is best FOR THIS DATASET (5 elderly subjects).
"""

ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('/Users/pascalschlegel/effort-estimator/output/pipeline_explanation.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: output/pipeline_explanation.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY FOR PRESENTATION")
print("="*70)
print("""
1. FEATURE SELECTION METHOD:
   - RF importance on pooled training data → selects Top N features
   - Evaluated with LOSO → measures GENERALIZATION to new subjects

2. KEY FINDING:
   - Top 10 IMU: r = 0.54 (BEST)
   - Adding PPG: r = 0.48 (WORSE)
   - All combined: r = 0.25 (WORST)
   
3. WHY IMU WINS:
   - Motion patterns are similar across people
   - PPG absolute values vary too much between individuals
   
4. PIPELINE ADAPTABILITY:
   - YES, it will re-rank features with new data
   - Current conclusion is specific to this 5-subject dataset
   - With more/different subjects, PPG might become useful
""")

plt.show()
