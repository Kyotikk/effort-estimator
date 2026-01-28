#!/usr/bin/env python3
"""Check for overfitting and data leakage in elderly EDA model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, LeaveOneGroupOut, train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3'].copy()

# Get EDA features
eda_features = [c for c in elderly.columns if 'eda' in c.lower()]
data = elderly[eda_features + ['borg']].dropna()

X = data[eda_features].values
y = data['borg'].values

# Create pseudo-activities based on Borg ranges
activities = pd.cut(y, bins=[0, 2, 4, 7], labels=['low', 'medium', 'high']).astype(str)

print(f"Samples: {len(data)}")
print(f"Features: {len(eda_features)}")
print(f"Borg distribution: {data['borg'].value_counts().sort_index().to_dict()}")
print()

# ============================================================================
# TEST 1: Train/Test Split (simple overfitting check)
# ============================================================================
print("=" * 80)
print("TEST 1: TRAIN/TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, 
                     reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0)
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print(f"Train R²: {train_r2:.3f}")
print(f"Test R²:  {test_r2:.3f}")
print(f"Gap:      {train_r2 - test_r2:.3f}")

if train_r2 - test_r2 > 0.15:
    print("⚠️  OVERFITTING: Train-test gap > 0.15")
else:
    print("✓ No severe overfitting (gap < 0.15)")

# ============================================================================
# TEST 2: Leave-One-Activity-Out (check if model generalizes across activities)
# ============================================================================
print()
print("=" * 80)
print("TEST 2: LEAVE-ONE-ACTIVITY-OUT CV")
print("=" * 80)

logo = LeaveOneGroupOut()
scores = cross_val_score(model, X, y, cv=logo, groups=activities, scoring='r2')
print(f"Leave-One-Activity-Out R²: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Per-activity scores: {dict(zip(np.unique(activities), scores))}")

if scores.mean() < 0.5:
    print("⚠️  WARNING: Model doesn't generalize well across activities")
else:
    print("✓ Model generalizes across activities")

# ============================================================================
# TEST 3: Temporal Leakage Check (consecutive windows might share info)
# ============================================================================
print()
print("=" * 80)
print("TEST 3: TEMPORAL LEAKAGE CHECK")
print("=" * 80)

# Split by time: first 70% train, last 30% test
n_train = int(len(X) * 0.7)
X_train_temp, X_test_temp = X[:n_train], X[n_train:]
y_train_temp, y_test_temp = y[:n_train], y[n_train:]

model.fit(X_train_temp, y_train_temp)
train_r2_temp = model.score(X_train_temp, y_train_temp)
test_r2_temp = model.score(X_test_temp, y_test_temp)

print(f"Temporal Train R² (first 70%): {train_r2_temp:.3f}")
print(f"Temporal Test R² (last 30%):   {test_r2_temp:.3f}")

if test_r2_temp < 0.3:
    print("⚠️  POSSIBLE TEMPORAL LEAKAGE: Poor performance on future data")
else:
    print("✓ Model works on temporally separated data")

# ============================================================================
# TEST 4: Feature Autocorrelation (EDA changes slowly -> adjacent windows similar)
# ============================================================================
print()
print("=" * 80)
print("TEST 4: WINDOW AUTOCORRELATION CHECK")
print("=" * 80)

# Check if adjacent windows have similar Borg values
borg_autocorr = np.corrcoef(y[:-1], y[1:])[0,1]
print(f"Borg autocorrelation (adjacent windows): {borg_autocorr:.3f}")

# Check EDA autocorrelation
eda_mean = data['eda_cc_mean'].values
eda_autocorr = np.corrcoef(eda_mean[:-1], eda_mean[1:])[0,1]
print(f"EDA autocorrelation (adjacent windows):  {eda_autocorr:.3f}")

if eda_autocorr > 0.9:
    print("⚠️  HIGH AUTOCORRELATION: Adjacent windows are very similar")
    print("   This inflates CV scores because train/test windows are not independent")
else:
    print("✓ Moderate autocorrelation")

# ============================================================================
# TEST 5: Shuffle Labels (Sanity Check)
# ============================================================================
print()
print("=" * 80)
print("TEST 5: SHUFFLE LABELS SANITY CHECK")
print("=" * 80)

np.random.seed(42)
y_shuffled = np.random.permutation(y)
scores_shuffled = cross_val_score(model, X, y_shuffled, cv=5, scoring='r2')
print(f"R² with shuffled labels: {scores_shuffled.mean():.3f} ± {scores_shuffled.std():.3f}")

if scores_shuffled.mean() > 0.1:
    print("⚠️  WARNING: Model fits random labels (possible issue)")
else:
    print("✓ Model cannot fit random labels (good sign)")

# ============================================================================
# TEST 6: Learning Curve
# ============================================================================
print()
print("=" * 80)
print("TEST 6: LEARNING CURVE")
print("=" * 80)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring='r2', random_state=42
)

print(f"{'Train Size':<15} {'Train R²':<15} {'Test R²':<15}")
for size, train_s, test_s in zip(train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)):
    print(f"{size:<15} {train_s:<15.3f} {test_s:<15.3f}")

# ============================================================================
# CONCLUSION
# ============================================================================
print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"""
OVERFITTING CHECK:
- Train/Test gap: {train_r2 - test_r2:.3f} (< 0.15 is OK)
- Leave-Activity-Out: {scores.mean():.3f} (> 0.5 is OK)
- Temporal split: {test_r2_temp:.3f}

LEAKAGE CHECK:
- Borg autocorrelation: {borg_autocorr:.3f}
- EDA autocorrelation: {eda_autocorr:.3f}
- Shuffled labels R²: {scores_shuffled.mean():.3f} (~0 expected)

HIGH AUTOCORRELATION MEANS:
- Adjacent 10-second windows share similar physiology
- This is NOT cheating, it's how biosignals work
- But it means test set isn't truly "independent"
- Real-world validation: test on NEW recording sessions
""")
