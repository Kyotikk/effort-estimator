#!/usr/bin/env python3
"""Simple visualization of WHY the model can't predict new sessions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3'].copy().reset_index(drop=True)

# Get EDA and Borg (drop NaN)
data = elderly[['eda_cc_mean', 'borg']].dropna().reset_index(drop=True)

print("=" * 70)
print("WHY MODEL CAN'T PREDICT NEW SESSIONS")
print("=" * 70)

# Split into "sessions" (first half = train, second half = test)
mid = len(data) // 2
train = data.iloc[:mid]
test = data.iloc[mid:]

print(f"\nTRAIN (first half of recording):")
print(f"  Borg range: {train['borg'].min():.1f} - {train['borg'].max():.1f}")
print(f"  Borg mean: {train['borg'].mean():.2f}")
print(f"  EDA range: {train['eda_cc_mean'].min():.3f} - {train['eda_cc_mean'].max():.3f}")

print(f"\nTEST (second half of recording):")
print(f"  Borg range: {test['borg'].min():.1f} - {test['borg'].max():.1f}")
print(f"  Borg mean: {test['borg'].mean():.2f}")
print(f"  EDA range: {test['eda_cc_mean'].min():.3f} - {test['eda_cc_mean'].max():.3f}")

# The key insight
print("\n" + "=" * 70)
print("THE PROBLEM:")
print("=" * 70)
print("""
The model learns: "EDA value X → Borg value Y"

But EDA is affected by:
  1. Effort level (what we want)
  2. Time of day
  3. Hydration
  4. Skin temperature  
  5. Electrode contact
  6. Individual baseline (varies per person, per session!)

In training data:
  - EDA = 0.5 → Borg = 2.0 (for THIS session)
  
In new session:
  - EDA = 0.5 → Borg = 5.0 (different baseline!)
  
The model memorized the EDA-Borg mapping for ONE session.
It doesn't generalize because the baseline shifts.
""")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: EDA vs Borg for train vs test
ax1 = axes[0]
ax1.scatter(train['eda_cc_mean'], train['borg'], c='blue', alpha=0.5, label=f'Train (n={len(train)})', s=20)
ax1.scatter(test['eda_cc_mean'], test['borg'], c='red', alpha=0.5, label=f'Test (n={len(test)})', s=20)
ax1.set_xlabel('EDA (Skin Conductance)')
ax1.set_ylabel('Borg CR10')
ax1.set_title('Train vs Test: Different EDA-Borg Relationships!')
ax1.legend()

# Add regression lines
from scipy import stats
slope_train, intercept_train, r_train, _, _ = stats.linregress(train['eda_cc_mean'], train['borg'])
slope_test, intercept_test, r_test, _, _ = stats.linregress(test['eda_cc_mean'], test['borg'])

x_range = np.linspace(data['eda_cc_mean'].min(), data['eda_cc_mean'].max(), 100)
ax1.plot(x_range, slope_train * x_range + intercept_train, 'b--', label=f'Train fit (r={r_train:.2f})')
ax1.plot(x_range, slope_test * x_range + intercept_test, 'r--', label=f'Test fit (r={r_test:.2f})')
ax1.legend()

# Plot 2: Borg over time
ax2 = axes[1]
ax2.plot(range(len(train)), train['borg'].values, 'b-', alpha=0.7, label='Train')
ax2.plot(range(len(train), len(data)), test['borg'].values, 'r-', alpha=0.7, label='Test')
ax2.axvline(len(train), color='black', linestyle='--', label='Train/Test Split')
ax2.set_xlabel('Time (Window Index)')
ax2.set_ylabel('Borg CR10')
ax2.set_title('Borg Over Time')
ax2.legend()

# Plot 3: EDA over time
ax3 = axes[2]
ax3.plot(range(len(train)), train['eda_cc_mean'].values, 'b-', alpha=0.7, label='Train')
ax3.plot(range(len(train), len(data)), test['eda_cc_mean'].values, 'r-', alpha=0.7, label='Test')
ax3.axvline(len(train), color='black', linestyle='--', label='Train/Test Split')
ax3.set_xlabel('Time (Window Index)')
ax3.set_ylabel('EDA Mean')
ax3.set_title('EDA Over Time')
ax3.legend()

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor/why_no_generalization.png', dpi=150)
print(f"\n✓ Saved: why_no_generalization.png")
plt.close()

# Show the actual prediction failure
print("\n" + "=" * 70)
print("DEMONSTRATION: Train on first half, test on second half")
print("=" * 70)

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

eda_features = [c for c in elderly.columns if 'eda' in c.lower()]
data_full = elderly[eda_features + ['borg']].dropna().reset_index(drop=True)

mid = len(data_full) // 2
X_train = data_full[eda_features].iloc[:mid].values
y_train = data_full['borg'].iloc[:mid].values
X_test = data_full[eda_features].iloc[mid:].values
y_test = data_full['borg'].iloc[mid:].values

model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
model.fit(X_train, y_train)

# Train performance
train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
train_mae = mean_absolute_error(y_train, train_pred)

# Test performance  
test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\nTrain performance:")
print(f"  R² = {train_r2:.3f}")
print(f"  MAE = {train_mae:.2f} Borg points")
print(f"  Actual Borg range: {y_train.min():.1f} - {y_train.max():.1f}")

print(f"\nTest performance (simulated 'new session'):")
print(f"  R² = {test_r2:.3f}")
print(f"  MAE = {test_mae:.2f} Borg points")
print(f"  Actual Borg range: {y_test.min():.1f} - {y_test.max():.1f}")
print(f"  Predicted Borg range: {test_pred.min():.1f} - {test_pred.max():.1f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
Train R² = {train_r2:.2f} (good)
Test R²  = {test_r2:.2f} (poor)

The model works within the training session but FAILS on "future" data.

WHY?
- EDA baseline drifts over time
- Different activities have different EDA-Borg relationships
- Model learns spurious patterns specific to training period

FOR NEW SESSION TO WORK, YOU NEED:
1. Multiple sessions with RANDOMIZED effort order
2. Or: Calibration phase at start of each session
3. Or: Person-specific model with their data
""")
