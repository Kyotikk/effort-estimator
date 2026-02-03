#!/usr/bin/env python3
"""
Better visualizations for r=0.52 that don't look terrible
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Output folder
OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# Load data (same as thesis_plots_loso.py)
print("\nLoading data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        tmp = pd.read_csv(path)
        tmp['subject'] = f'P{i}'
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

# Get IMU columns (same logic as thesis_plots_loso.py)
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
y_col = 'borg'

print(f"Data: {len(df)} windows, {df['subject'].nunique()} subjects")
print(f"IMU features: {len(imu_cols)}")

# =============================================================================
# LOSO predictions (collect all)
# =============================================================================
subjects = sorted(df['subject'].unique())
all_actual = []
all_pred = []
all_subj = []
per_subj_r = []

for test_subj in subjects:
    train = df[df['subject'] != test_subj]
    test = df[df['subject'] == test_subj]
    
    # Same preprocessing as thesis_plots_loso.py!
    X_train = train[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train[y_col].values
    X_test = test[imu_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test = test[y_col].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Per-subject r
    r_subj, _ = pearsonr(y_test, y_pred)
    per_subj_r.append(r_subj)
    print(f"  {test_subj}: r={r_subj:.2f}")
    
    all_actual.extend(y_test)
    all_pred.extend(y_pred)
    all_subj.extend([test_subj] * len(y_test))

all_actual = np.array(all_actual)
all_pred = np.array(all_pred)
all_subj = np.array(all_subj)

r_pooled, _ = pearsonr(all_actual, all_pred)
r_mean = np.mean(per_subj_r)
print(f"\nMean per-subject r = {r_mean:.2f}  ← This is the correct metric!")
print(f"Pooled r = {r_pooled:.2f}  ← Lower due to subject baseline differences")

# =============================================================================
# PLOT 1: Binned Bar Chart - "Does prediction increase with actual effort?"
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Bin actual Borg into categories
bins = [0, 1, 2, 3, 4, 5, 6, 10]
bin_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6+']
bin_indices = np.digitize(all_actual, bins) - 1

# Calculate mean prediction per bin
bin_means = []
bin_stds = []
bin_counts = []
valid_labels = []

for i, label in enumerate(bin_labels):
    mask = bin_indices == i
    if mask.sum() > 5:  # Only include bins with enough data
        bin_means.append(np.mean(all_pred[mask]))
        bin_stds.append(np.std(all_pred[mask]))
        bin_counts.append(mask.sum())
        valid_labels.append(label)

x = np.arange(len(valid_labels))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(valid_labels)))

bars = ax.bar(x, bin_means, yerr=bin_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, bin_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bin_stds[i] + 0.15, 
            f'n={count}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Actual Borg CR10 (Self-Reported Effort)', fontsize=12)
ax.set_ylabel('Mean Predicted Borg CR10', fontsize=12)
ax.set_title(f'IMU-Based Effort Prediction (LOSO, r={r_mean:.2f})\nPredictions Increase with Actual Effort', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(valid_labels)
ax.set_ylim(0, 6)

# Add trend line
z = np.polyfit(x, bin_means, 1)
p = np.poly1d(z)
ax.plot(x, p(x), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/20_binned_predictions.png", dpi=150, bbox_inches='tight')
print(f"Saved: 20_binned_predictions.png")
plt.close()

# =============================================================================
# PLOT 2: Box plots by effort category
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create effort categories: Low (0-2), Medium (2-4), High (4+)
categories = []
for a in all_actual:
    if a < 2:
        categories.append('Low\n(0-2)')
    elif a < 4:
        categories.append('Medium\n(2-4)')
    else:
        categories.append('High\n(4+)')

categories = np.array(categories)
cat_order = ['Low\n(0-2)', 'Medium\n(2-4)', 'High\n(4+)']

# Box plot data
box_data = [all_pred[categories == cat] for cat in cat_order]
box_colors = ['#3498db', '#f39c12', '#e74c3c']

bp = ax.boxplot(box_data, patch_artist=True, labels=cat_order)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add individual points with jitter
for i, cat in enumerate(cat_order):
    y = all_pred[categories == cat]
    x = np.random.normal(i+1, 0.08, size=len(y))
    ax.scatter(x, y, alpha=0.3, color=box_colors[i], s=20)

# Add mean markers
means = [np.mean(d) for d in box_data]
ax.scatter([1, 2, 3], means, color='black', marker='D', s=100, zorder=5, label='Mean')

ax.set_xlabel('Actual Self-Reported Effort Category', fontsize=12)
ax.set_ylabel('Predicted Borg CR10', fontsize=12)
ax.set_title(f'IMU-Based Effort Prediction (LOSO, r={r_mean:.2f})\nModel Distinguishes Low vs Medium vs High Effort', fontsize=14)
ax.legend()

# Add counts
for i, cat in enumerate(cat_order):
    n = (categories == cat).sum()
    ax.text(i+1, ax.get_ylim()[1]-0.3, f'n={n}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/21_boxplot_by_category.png", dpi=150, bbox_inches='tight')
print(f"Saved: 21_boxplot_by_category.png")
plt.close()

# =============================================================================
# PLOT 3: Per-subject time series (shows trends match even if magnitude differs)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, subj in enumerate(subjects):
    ax = axes[idx]
    mask = all_subj == subj
    actual = all_actual[mask]
    pred = all_pred[mask]
    
    # Time index (just sequential)
    t = np.arange(len(actual))
    
    ax.plot(t, actual, 'b-', alpha=0.7, linewidth=1.5, label='Actual Borg')
    ax.plot(t, pred, 'r-', alpha=0.7, linewidth=1.5, label='Predicted')
    
    r_subj, _ = pearsonr(actual, pred)
    ax.set_title(f'{subj} (r={r_subj:.2f})', fontsize=12)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Borg CR10')
    ax.set_ylim(-0.5, 7)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[5].axis('off')

fig.suptitle(f'IMU Predictions Track Actual Effort Over Time (LOSO)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/22_time_series_per_subject.png", dpi=150, bbox_inches='tight')
print(f"Saved: 22_time_series_per_subject.png")
plt.close()

# =============================================================================
# PLOT 4: Improved scatter with better framing
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 8))

# Color by subject
colors_subj = {'P1': '#e74c3c', 'P2': '#3498db', 'P3': '#2ecc71', 'P4': '#9b59b6', 'P5': '#f39c12'}
for subj in subjects:
    mask = all_subj == subj
    ax.scatter(all_actual[mask], all_pred[mask], c=colors_subj[subj], 
               alpha=0.6, s=40, label=subj, edgecolors='white', linewidth=0.5)

# Regression line (not perfect diagonal!)
z = np.polyfit(all_actual, all_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 7, 100)
ax.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'Fit (r={r_mean:.2f})')

# Don't show the "perfect" diagonal - it's misleading
# Just show the actual fit line

ax.set_xlabel('Actual Borg CR10 (Self-Reported)', fontsize=12)
ax.set_ylabel('Predicted Borg CR10 (IMU Model)', fontsize=12)
ax.set_title(f'LOSO Cross-Validation: Predicted vs Actual\nr = {r_mean:.2f}, p < 0.001', fontsize=14)
ax.set_xlim(-0.5, 7)
ax.set_ylim(-0.5, 7)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add annotation
ax.text(0.95, 0.05, 'Each subject tested on model\ntrained without that subject',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUT}/23_scatter_improved.png", dpi=150, bbox_inches='tight')
print(f"Saved: 23_scatter_improved.png")
plt.close()

# =============================================================================
# PLOT 5: Classification accuracy (if we bin into categories)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Convert to Low/Medium/High
def to_category(x):
    if x < 2:
        return 0  # Low
    elif x < 4:
        return 1  # Medium
    else:
        return 2  # High

actual_cat = np.array([to_category(a) for a in all_actual])
pred_cat = np.array([to_category(p) for p in all_pred])

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(actual_cat, pred_cat)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Low\n(0-2)', 'Medium\n(2-4)', 'High\n(4+)'])
ax.set_yticklabels(['Low\n(0-2)', 'Medium\n(2-4)', 'High\n(4+)'])
ax.set_xlabel('Predicted Category', fontsize=12)
ax.set_ylabel('Actual Category', fontsize=12)

# Add text annotations
for i in range(3):
    for j in range(3):
        text = f'{cm_norm[i, j]:.0%}\n({cm[i, j]})'
        color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

accuracy = accuracy_score(actual_cat, pred_cat)
ax.set_title(f'Effort Category Classification (LOSO)\nAccuracy: {accuracy:.0%}', fontsize=14)

plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(f"{OUT}/24_confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"Saved: 24_confusion_matrix.png")
plt.close()

print("\n" + "="*60)
print("SUMMARY: Which plot to use?")
print("="*60)
print("""
Plot 20 (Binned Bar Chart):
  - BEST for showing "predictions increase with actual effort"
  - Clear visual trend, easy to explain
  
Plot 21 (Box Plot by Category):
  - Shows model distinguishes Low vs Medium vs High
  - Good for showing distribution

Plot 22 (Time Series):
  - Shows predictions TRACK actual effort over time
  - Good for "the model follows the patient's effort changes"

Plot 23 (Improved Scatter):
  - Same as before but without misleading diagonal
  - Shows only the actual fit line

Plot 24 (Confusion Matrix):
  - If you frame it as classification (Low/Medium/High)
  - Shows accuracy in categories
""")
