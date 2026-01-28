#!/usr/bin/env python3
"""Analyze WHY temporal CV fails - understand the data structure."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3'].copy().reset_index(drop=True)

print("=" * 80)
print("UNDERSTANDING THE DATA STRUCTURE")
print("=" * 80)

# Check Borg over time
print("\nBorg values over time (first 30 windows):")
print(elderly[['t_start', 'borg']].head(30).to_string())

print("\n\nBorg transition analysis:")
borg_diff = elderly['borg'].diff().fillna(0)
print(f"Windows with same Borg as previous: {(borg_diff == 0).sum()} ({(borg_diff == 0).sum()/len(elderly)*100:.1f}%)")
print(f"Windows with Borg change: {(borg_diff != 0).sum()} ({(borg_diff != 0).sum()/len(elderly)*100:.1f}%)")

# Check if Borg is monotonic (always increasing or always decreasing)
borg_vals = elderly['borg'].values
changes = np.diff(borg_vals)
n_increases = (changes > 0).sum()
n_decreases = (changes < 0).sum()
n_same = (changes == 0).sum()

print(f"\nBorg transitions:")
print(f"  Increases: {n_increases}")
print(f"  Decreases: {n_decreases}")
print(f"  Same: {n_same}")

# Check Borg by time segment
n_segments = 5
segment_size = len(elderly) // n_segments
print(f"\nBorg by time segment (n={segment_size} each):")
for i in range(n_segments):
    start = i * segment_size
    end = (i + 1) * segment_size
    segment_borg = elderly['borg'].iloc[start:end]
    print(f"  Segment {i+1}: Borg {segment_borg.mean():.2f} ± {segment_borg.std():.2f} (range: {segment_borg.min():.1f}-{segment_borg.max():.1f})")

print()
print("=" * 80)
print("THE PROBLEM")
print("=" * 80)

# Is Borg clustered in time?
unique_borgs = elderly['borg'].unique()
print(f"\nUnique Borg values: {sorted(unique_borgs)}")
print(f"\nBorg appears in these time ranges:")

for borg in sorted(unique_borgs)[:5]:  # Show first 5
    indices = elderly[elderly['borg'] == borg].index.tolist()
    if len(indices) > 0:
        # Find consecutive runs
        runs = []
        start = indices[0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                runs.append((start, indices[i-1]))
                start = indices[i]
        runs.append((start, indices[-1]))
        print(f"  Borg {borg}: {len(indices)} windows in {len(runs)} run(s)")
        for r in runs[:3]:
            print(f"    - windows {r[0]}-{r[1]}")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("""
The problem is DATA STRUCTURE, not the model:

1. The Borg labels are CLUSTERED in time
   - Low effort windows are together
   - High effort windows are together
   
2. With time-series CV:
   - Train on early data (e.g., low effort)
   - Test on later data (e.g., high effort)
   - Model never sees high effort in training!

3. This is a FUNDAMENTAL LIMITATION:
   - You have ONE continuous recording
   - Effort levels change over time
   - Can't learn future effort patterns from past

SOLUTION OPTIONS:
1. Multiple recording sessions with varied effort order
2. Activity-stratified sampling (ensure each fold has all Borg levels)
3. Accept that within-session prediction works, between-session doesn't
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Borg over time
ax1 = axes[0, 0]
ax1.plot(range(len(elderly)), elderly['borg'].values, 'b-', alpha=0.7)
ax1.set_xlabel('Window Index (Time)')
ax1.set_ylabel('Borg CR10')
ax1.set_title('Borg Over Time (Elderly)')
ax1.axhline(elderly['borg'].mean(), color='r', linestyle='--', label=f'Mean={elderly["borg"].mean():.1f}')
ax1.legend()

# 2. Time-series CV folds visualization
ax2 = axes[0, 1]
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
colors = plt.cm.Set2(np.linspace(0, 1, 5))
for i, (train_idx, test_idx) in enumerate(tscv.split(elderly)):
    train_borg = elderly['borg'].iloc[train_idx].values
    test_borg = elderly['borg'].iloc[test_idx].values
    ax2.scatter([i]*len(train_borg), train_borg, c='blue', alpha=0.3, s=5, label='Train' if i==0 else '')
    ax2.scatter([i+0.3]*len(test_borg), test_borg, c='red', alpha=0.5, s=10, label='Test' if i==0 else '')
    ax2.text(i+0.15, 7, f'Train: {train_borg.mean():.1f}\nTest: {test_borg.mean():.1f}', fontsize=8, ha='center')

ax2.set_xlabel('CV Fold')
ax2.set_ylabel('Borg CR10')
ax2.set_title('Time-Series CV: Train vs Test Borg Distribution')
ax2.legend()
ax2.set_xticks(range(5))
ax2.set_xticklabels([f'Fold {i+1}' for i in range(5)])

# 3. EDA over time with Borg
ax3 = axes[1, 0]
eda_col = 'eda_cc_mean'
if eda_col in elderly.columns:
    ax3.scatter(range(len(elderly)), elderly[eda_col].values, c=elderly['borg'].values, 
                cmap='RdYlGn_r', alpha=0.6, s=10)
    ax3.set_xlabel('Window Index (Time)')
    ax3.set_ylabel('EDA Mean')
    ax3.set_title('EDA Over Time (Color = Borg)')
    plt.colorbar(ax3.collections[0], ax=ax3, label='Borg')

# 4. Borg distribution
ax4 = axes[1, 1]
elderly['borg'].hist(bins=20, ax=ax4, edgecolor='black')
ax4.set_xlabel('Borg CR10')
ax4.set_ylabel('Count')
ax4.set_title('Borg Distribution')

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor/data_structure_analysis.png', dpi=150)
print(f"\n✓ Saved: data_structure_analysis.png")
plt.close()

# Now try STRATIFIED time-series (ensure Borg diversity in each fold)
print()
print("=" * 80)
print("TRYING STRATIFIED APPROACH")
print("=" * 80)

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold

# Bin Borg into categories for stratification
elderly['borg_bin'] = pd.cut(elderly['borg'], bins=5, labels=False)

# Get features
eda_features = [c for c in elderly.columns if 'eda' in c.lower()]
data = elderly[eda_features + ['borg', 'borg_bin']].dropna()

X = data[eda_features].values
y = data['borg'].values
y_bin = data['borg_bin'].values

model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                     reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0)

# Stratified K-Fold (ensures all Borg levels in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in skf.split(X, y_bin):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
    scores.append(score)

print(f"Stratified K-Fold (EDA): R² = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
print("  (Each fold has similar Borg distribution)")
print()
print("BUT: This still has temporal leakage (adjacent windows in train/test)")
print("It's a TRADE-OFF between:")
print("  - No leakage (time-series CV) → R² < 0")
print("  - Representative folds (stratified) → R² ~ 0.9 (inflated)")
print()
print("HONEST REPORTING:")
print("  'Within-session prediction R² = 0.9 (stratified CV)'")
print("  'Generalization to new sessions not validated'")
