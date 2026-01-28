#!/usr/bin/env python3
"""Proper XGBoost model for elderly using EDA + IMU (most complete data)."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load elderly data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3'].copy()

print(f"Total elderly samples: {len(elderly)}")
print(f"Borg range: {elderly['borg'].min():.1f} - {elderly['borg'].max():.1f}")
print()

# Get features by modality
exclude_cols = ['borg', 'subject_id', 'timestamp', 'window_start', 'window_end', 'activity', 'adl_name']

def get_features_by_modality(df, modality):
    if modality == 'EDA':
        return [c for c in df.columns if 'eda' in c.lower() or 'scl' in c.lower() or 'scr' in c.lower()]
    elif modality == 'IMU':
        return [c for c in df.columns if any(x in c.lower() for x in ['acc_', 'gyro', 'mag_'])]
    elif modality == 'HRV':
        return [c for c in df.columns if any(x in c.lower() for x in ['ibi', 'rmssd', 'sdnn', 'pnn', 'hr_', 'lf', 'hf'])]
    elif modality == 'PPG':
        return [c for c in df.columns if 'ppg' in c.lower()]
    return []

eda_features = get_features_by_modality(elderly, 'EDA')
imu_features = get_features_by_modality(elderly, 'IMU')
hrv_features = get_features_by_modality(elderly, 'HRV')
ppg_features = get_features_by_modality(elderly, 'PPG')

print(f"EDA features: {len(eda_features)}")
print(f"IMU features: {len(imu_features)}")
print(f"HRV features: {len(hrv_features)}")
print(f"PPG features (all): {len(ppg_features)}")

# Test different feature combinations
combos = [
    ('EDA only', eda_features),
    ('IMU only', imu_features),
    ('HRV only', hrv_features),
    ('EDA + IMU', eda_features + imu_features),
    ('EDA + HRV', eda_features + hrv_features),
    ('IMU + HRV', imu_features + hrv_features),
    ('EDA + IMU + HRV', eda_features + imu_features + hrv_features),
    ('ALL PPG + EDA + IMU', ppg_features + eda_features + imu_features),
]

print()
print("=" * 80)
print("XGBOOST 5-FOLD CV RESULTS (Elderly Only)")
print("=" * 80)
print(f"{'Combination':<25} {'n_samples':>10} {'n_features':>12} {'CV R²':>15}")
print("-" * 80)

best_r2 = -999
best_combo = None

for name, features in combos:
    # Keep only features that exist and aren't all NaN
    valid_features = [f for f in features if f in elderly.columns]
    valid_features = [f for f in valid_features if elderly[f].notna().sum() > 100]
    
    if len(valid_features) == 0:
        continue
    
    # Prepare data
    data = elderly[valid_features + ['borg']].copy()
    
    # Drop rows with any NaN (strict)
    data_clean = data.dropna()
    
    if len(data_clean) < 100:
        print(f"{name:<25} {'Too few samples':<40}")
        continue
    
    X = data_clean[valid_features].values
    y = data_clean['borg'].values
    
    # XGBoost with regularization
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=1.0,  # L1
        reg_lambda=1.0,  # L2
        random_state=42,
        verbosity=0
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    
    print(f"{name:<25} {len(data_clean):>10} {len(valid_features):>12} {mean_r2:>10.3f} ± {std_r2:.3f}")
    
    if mean_r2 > best_r2:
        best_r2 = mean_r2
        best_combo = (name, valid_features, data_clean)

print()
print("=" * 80)
print(f"BEST COMBINATION: {best_combo[0]} (R² = {best_r2:.3f})")
print("=" * 80)

# Train best model and get feature importance
name, features, data = best_combo
X = data[features].values
y = data['borg'].values

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
model.fit(X, y)

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
for i, (_, row) in enumerate(importances.head(15).iterrows()):
    modality = 'EDA' if 'eda' in row['feature'].lower() else 'IMU' if any(x in row['feature'].lower() for x in ['acc', 'gyro']) else 'HRV' if any(x in row['feature'].lower() for x in ['ibi', 'rmssd', 'hr_']) else 'PPG'
    print(f"  {i+1:>2}. {row['feature']:<45} {row['importance']:.3f} ({modality})")

print()
print("=" * 80)
print("CORRELATION (r) vs R² EXPLANATION")
print("=" * 80)
print("""
r = 0.45 (correlation) means R² = 0.20 for single feature!
  - r² = 0.45² = 0.20

To get R² = 0.64 (which is r ≈ 0.8):
  - Need MULTIPLE features working together
  - Current best: R² ≈ 0.50-0.60 with EDA + IMU

To get higher R²:
  1. More training data (you only have ~500-900 samples)
  2. Better feature engineering
  3. Activity-specific models (sit vs walk have different physiology)
  4. Time-series models (LSTM, temporal patterns)
""")

# Quick check: correlation between prediction and actual
from scipy import stats
cv = KFold(n_splits=5, shuffle=True, random_state=42)
all_preds = []
all_true = []
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    all_preds.extend(preds)
    all_true.extend(y[test_idx])

r, p = stats.pearsonr(all_preds, all_true)
print(f"\nPredicted vs Actual Borg correlation: r = {r:.3f} (p = {p:.2e})")
print(f"This is equivalent to √R² ≈ {np.sqrt(max(0, best_r2)):.3f}")
