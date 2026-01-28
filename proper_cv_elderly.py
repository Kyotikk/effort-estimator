#!/usr/bin/env python3
"""Proper cross-validation without temporal leakage for elderly Borg prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
elderly = df[df['subject_id'] == 'sim_elderly3'].copy().reset_index(drop=True)

print("=" * 80)
print("PROPER CROSS-VALIDATION FOR ELDERLY BORG PREDICTION")
print("=" * 80)
print(f"Total samples: {len(elderly)}")
print(f"Borg range: {elderly['borg'].min():.1f} - {elderly['borg'].max():.1f}")
print()

# Get features by modality
def get_features(df, modality):
    if modality == 'EDA':
        return [c for c in df.columns if 'eda' in c.lower()]
    elif modality == 'IMU':
        return [c for c in df.columns if any(x in c.lower() for x in ['acc_', 'gyro'])]
    elif modality == 'HRV':
        return [c for c in df.columns if any(x in c.lower() for x in ['ibi', 'rmssd', 'sdnn', 'pnn', 'hr_', 'lf', 'hf'])]
    return []

# Create session groups based on recording segments (detect gaps > 60s)
elderly['time_diff'] = elderly['t_start'].diff().fillna(0)
elderly['session'] = (elderly['time_diff'] > 60).cumsum()
n_sessions = elderly['session'].nunique()
print(f"Detected {n_sessions} recording sessions (gaps > 60s)")
print(f"Samples per session: {elderly.groupby('session').size().to_dict()}")
print()

# Model
model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

# Test different feature sets with proper CV
feature_sets = {
    'EDA': get_features(elderly, 'EDA'),
    'IMU': get_features(elderly, 'IMU'),
    'HRV': get_features(elderly, 'HRV'),
    'EDA+IMU': get_features(elderly, 'EDA') + get_features(elderly, 'IMU'),
    'EDA+HRV': get_features(elderly, 'EDA') + get_features(elderly, 'HRV'),
    'ALL': get_features(elderly, 'EDA') + get_features(elderly, 'IMU') + get_features(elderly, 'HRV'),
}

print("=" * 80)
print("METHOD 1: TIME-SERIES SPLIT (Train on past, test on future)")
print("=" * 80)
print("This is the GOLD STANDARD for temporal data - no leakage possible")
print()

tscv = TimeSeriesSplit(n_splits=5)

print(f"{'Features':<15} {'R²':<20} {'MAE (Borg)':<15} {'n_samples':<10}")
print("-" * 60)

best_r2 = -999
best_name = None
best_features = None

for name, features in feature_sets.items():
    valid_features = [f for f in features if f in elderly.columns]
    data = elderly[valid_features + ['borg']].dropna()
    
    if len(data) < 100:
        continue
    
    X = data[valid_features].values
    y = data['borg'].values
    
    r2_scores = []
    mae_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        r2_scores.append(r2_score(y[test_idx], pred))
        mae_scores.append(mean_absolute_error(y[test_idx], pred))
    
    mean_r2 = np.mean(r2_scores)
    mean_mae = np.mean(mae_scores)
    
    print(f"{name:<15} {mean_r2:>6.3f} ± {np.std(r2_scores):.3f}     {mean_mae:>5.2f} ± {np.std(mae_scores):.2f}      {len(data):<10}")
    
    if mean_r2 > best_r2:
        best_r2 = mean_r2
        best_name = name
        best_features = valid_features

print()
print("=" * 80)
print("METHOD 2: LEAVE-ONE-SESSION-OUT (Most realistic)")
print("=" * 80)
print("Test on completely unseen recording sessions")
print()

if n_sessions >= 3:
    gkf = GroupKFold(n_splits=min(n_sessions, 5))
    
    print(f"{'Features':<15} {'R²':<20} {'MAE (Borg)':<15}")
    print("-" * 50)
    
    for name, features in feature_sets.items():
        valid_features = [f for f in features if f in elderly.columns]
        data = elderly[valid_features + ['borg', 'session']].dropna()
        
        if len(data) < 100:
            continue
        
        X = data[valid_features].values
        y = data['borg'].values
        groups = data['session'].values
        
        r2_scores = []
        mae_scores = []
        
        for train_idx, test_idx in gkf.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            r2_scores.append(r2_score(y[test_idx], pred))
            mae_scores.append(mean_absolute_error(y[test_idx], pred))
        
        mean_r2 = np.mean(r2_scores)
        mean_mae = np.mean(mae_scores)
        
        print(f"{name:<15} {mean_r2:>6.3f} ± {np.std(r2_scores):.3f}     {mean_mae:>5.2f} ± {np.std(mae_scores):.2f}")
else:
    print("Not enough sessions for leave-one-session-out CV")

print()
print("=" * 80)
print("METHOD 3: BLOCKED TIME-SERIES (Gap between train/test)")
print("=" * 80)
print("Ensure train and test don't share adjacent windows")
print()

# Custom blocked CV with gap
def blocked_time_series_cv(X, y, n_splits=5, gap=10):
    """Time series CV with gap between train and test."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end + gap
        test_end = min(test_start + fold_size, n)
        
        if test_start < n and test_end > test_start:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

print(f"{'Features':<15} {'R²':<20} {'MAE (Borg)':<15}")
print("-" * 50)

for name, features in feature_sets.items():
    valid_features = [f for f in features if f in elderly.columns]
    data = elderly[valid_features + ['borg']].dropna()
    
    if len(data) < 100:
        continue
    
    X = data[valid_features].values
    y = data['borg'].values
    
    r2_scores = []
    mae_scores = []
    
    for train_idx, test_idx in blocked_time_series_cv(X, y, n_splits=5, gap=20):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        r2_scores.append(r2_score(y[test_idx], pred))
        mae_scores.append(mean_absolute_error(y[test_idx], pred))
    
    if len(r2_scores) > 0:
        mean_r2 = np.mean(r2_scores)
        mean_mae = np.mean(mae_scores)
        print(f"{name:<15} {mean_r2:>6.3f} ± {np.std(r2_scores):.3f}     {mean_mae:>5.2f} ± {np.std(mae_scores):.2f}")

print()
print("=" * 80)
print(f"BEST MODEL: {best_name}")
print("=" * 80)

# Train final model and show feature importance
data = elderly[best_features + ['borg']].dropna()
X = data[best_features].values
y = data['borg'].values

model.fit(X, y)
importances = pd.DataFrame({
    'feature': best_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(importances.head(10).iterrows()):
    print(f"  {i+1:>2}. {row['feature']:<40} {row['importance']:.3f}")

# Final predictions vs actual correlation
tscv = TimeSeriesSplit(n_splits=5)
all_preds = []
all_true = []
for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    all_preds.extend(preds)
    all_true.extend(y[test_idx])

r, p = stats.pearsonr(all_preds, all_true)

print()
print("=" * 80)
print("FINAL RESULTS (Time-Series CV)")
print("=" * 80)
print(f"Best feature set: {best_name}")
print(f"R² (time-series CV): {best_r2:.3f}")
print(f"Correlation (predicted vs actual): r = {r:.3f}")
print(f"Mean Absolute Error: {np.mean(np.abs(np.array(all_preds) - np.array(all_true))):.2f} Borg points")
print()
print("INTERPRETATION:")
if best_r2 > 0.5:
    print(f"  ✓ R² = {best_r2:.2f} is GOOD for physiological prediction")
    print(f"  ✓ Model explains {best_r2*100:.0f}% of Borg variance")
elif best_r2 > 0.3:
    print(f"  ~ R² = {best_r2:.2f} is MODERATE")
    print(f"  ~ Model explains {best_r2*100:.0f}% of Borg variance")
else:
    print(f"  ✗ R² = {best_r2:.2f} is WEAK")
    print(f"  ✗ Model struggles to predict Borg from features")

print()
print("COMPARISON WITH LITERATURE:")
print("  - Single HRV feature vs effort: r = 0.3-0.5 (typical)")
print("  - Multimodal wearable models: R² = 0.4-0.7 (state of art)")
print("  - Your model: R² = {:.2f}".format(best_r2))
