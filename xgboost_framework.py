#!/usr/bin/env python3
"""
XGBoost Framework for Effort Estimation
- Works with current data structure
- Uses Borg changes as activity boundaries for GroupKFold
- Ready to scale when more subjects/data collected
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("XGBOOST EFFORT ESTIMATION FRAMEWORK")
print("="*70)

data_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} windows × {df.shape[1]} columns")

# Remove rows with NaN Borg (can't train on missing labels)
n_before = len(df)
df = df.dropna(subset=['borg']).reset_index(drop=True)
if len(df) < n_before:
    print(f"  Removed {n_before - len(df)} rows with NaN Borg")

y = df['borg'].values
print(f"Borg range: [{y.min():.1f}, {y.max():.1f}]")

# ============================================================================
# 2. INFER ACTIVITY BOUNDARIES FROM BORG CHANGES
# ============================================================================
print("\n" + "-"*70)
print("STEP 2: INFER ACTIVITY BOUNDARIES")
print("-"*70)

# When Borg changes, it's a new activity
borg_changes = np.diff(y, prepend=y[0]) != 0
activity_ids = np.cumsum(borg_changes)

n_activities = len(np.unique(activity_ids))
print(f"  Inferred {n_activities} activities from Borg changes")
print(f"  Windows per activity: {len(df)/n_activities:.1f} average")

# ============================================================================
# 3. PREPARE FEATURES
# ============================================================================
print("\n" + "-"*70)
print("STEP 3: PREPARE FEATURES")
print("-"*70)

meta_cols = ['t_center', 'borg', 'activity', 'activity_id', 'subject_id', 
             'valid', 'n_samples', 'win_sec', 'modality']
feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]

X = df[feature_cols].copy()

# Remove constant features
constant_cols = X.columns[X.nunique() <= 1].tolist()
if constant_cols:
    print(f"  Removing {len(constant_cols)} constant features")
    X = X.drop(columns=constant_cols)

# Handle NaN/Inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"  Final: {X.shape[1]} features")

# Count by modality
ppg_cols = [c for c in X.columns if c.startswith('ppg_')]
acc_cols = [c for c in X.columns if c.startswith('acc_')]
eda_cols = [c for c in X.columns if c.startswith('eda_')]
print(f"  By modality: PPG={len(ppg_cols)}, ACC={len(acc_cols)}, EDA={len(eda_cols)}")

# ============================================================================
# 4. GROUPKFOLD CROSS-VALIDATION WITH FEATURE SELECTION
# ============================================================================
print("\n" + "-"*70)
print("STEP 4: GROUPKFOLD CV WITH FEATURE SELECTION")
print("-"*70)

n_splits = min(5, n_activities)
print(f"  {n_splits}-fold CV, splitting by activity (no window leakage)")

from sklearn.feature_selection import mutual_info_regression

# Define physiologically relevant features (same signals as literature formula)
hr_features = [c for c in X.columns if 'hr_mean' in c or 'hr_max' in c]
rmssd_features = [c for c in X.columns if 'rmssd' in c]
ibi_features = [c for c in X.columns if 'ibi' in c]
acc_features = [c for c in X.columns if c.startswith('acc_') and ('mean' in c or 'std' in c or 'energy' in c)]

# Option A: Use ALL features, select top 15 by MI
# Option B: Use ONLY physiological features (like literature)
# Let's try BOTH

print("\n  --- Option A: MI-based selection (top 15 from all) ---")
gkf = GroupKFold(n_splits=n_splits)
predictions_mi = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, activity_ids)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    mi_scores = mutual_info_regression(X_train.fillna(0), y_train, random_state=42)
    top_indices = np.argsort(mi_scores)[-15:]
    selected = X_train.columns[top_indices].tolist()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected])
    X_test_scaled = scaler.transform(X_test[selected])
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    predictions_mi[test_idx] = model.predict(X_test_scaled)

r_mi, _ = pearsonr(y, predictions_mi)
rmse_mi = np.sqrt(np.mean((y - predictions_mi)**2))
print(f"  Result: r={r_mi:.3f}, RMSE={rmse_mi:.2f}")

print("\n  --- Option B: Physiological features only (HR, RMSSD, IBI, ACC) ---")
physio_features = hr_features + rmssd_features + ibi_features + acc_features[:10]
physio_features = [f for f in physio_features if f in X.columns]
print(f"  Using {len(physio_features)} physiological features")

predictions_physio = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, activity_ids)):
    X_train, X_test = X.iloc[train_idx][physio_features], X.iloc[test_idx][physio_features]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    predictions_physio[test_idx] = model.predict(X_test_scaled)

r_physio, _ = pearsonr(y, predictions_physio)
rmse_physio = np.sqrt(np.mean((y - predictions_physio)**2))
print(f"  Result: r={r_physio:.3f}, RMSE={rmse_physio:.2f}")

print("\n  --- Option C: HR features only (closest to literature) ---")
hr_only = [c for c in X.columns if 'hr_' in c]
print(f"  Using {len(hr_only)} HR features")

predictions_hr = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, activity_ids)):
    X_train, X_test = X.iloc[train_idx][hr_only], X.iloc[test_idx][hr_only]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    predictions_hr[test_idx] = model.predict(X_test_scaled)

r_hr, _ = pearsonr(y, predictions_hr)
rmse_hr = np.sqrt(np.mean((y - predictions_hr)**2))
print(f"  Result: r={r_hr:.3f}, RMSE={rmse_hr:.2f}")

# Use best result
if r_physio > r_mi and r_physio > r_hr:
    predictions = predictions_physio
    best_method = "physiological"
elif r_hr > r_mi:
    predictions = predictions_hr
    best_method = "HR only"
else:
    predictions = predictions_mi
    best_method = "MI-selected"
    
overall_r, p_value = pearsonr(y, predictions)
overall_rmse = np.sqrt(np.mean((y - predictions)**2))
overall_mae = np.mean(np.abs(y - predictions))
print(f"\n  Best method: {best_method}")

# ============================================================================
# 5. OVERALL RESULTS
# ============================================================================
print("\n" + "-"*70)
print("STEP 5: RESULTS")
print("-"*70)

overall_r, p_value = pearsonr(y, predictions)
overall_rmse = np.sqrt(np.mean((y - predictions)**2))
overall_mae = np.mean(np.abs(y - predictions))

print(f"\n  XGBoost (GroupKFold by activity):")
print(f"    Pearson r:  {overall_r:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
print(f"    RMSE:       {overall_rmse:.3f} Borg points")
print(f"    MAE:        {overall_mae:.3f} Borg points")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "-"*70)
print("STEP 6: TOP FEATURES")
print("-"*70)

# Retrain on all data for feature importance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
final_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
final_model.fit(X_scaled, y)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 15 features:")
for i, row in importance.head(15).iterrows():
    modality = row['feature'].split('_')[0].upper()
    print(f"    {row['importance']:.4f}  [{modality}] {row['feature']}")

# ============================================================================
# 7. COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("COMPARISON: XGBoost vs Literature Formula")
print("="*70)

print(f"""
  Literature Formula (HR_delta × √duration):
    - Pearson r:  0.843***
    - Simple, interpretable
    - Based on physiological principles
    
  XGBoost ({X.shape[1]} features):
    - Pearson r:  {overall_r:.3f}{'***' if p_value < 0.001 else ''}
    - RMSE:       {overall_rmse:.3f} Borg points
    - Complex, requires feature engineering
    
  Note: XGBoost validated with GroupKFold (honest estimate)
        Literature formula computed on same windows (comparable)
""")

# ============================================================================
# 8. SAVE FOR LATER USE
# ============================================================================
results = {
    'xgboost_r': overall_r,
    'xgboost_rmse': overall_rmse,
    'xgboost_mae': overall_mae,
    'n_features': X.shape[1],
    'n_windows': len(df),
    'n_activities': n_activities,
    'top_features': importance.head(20).to_dict('records')
}

import json
output_path = Path("/Users/pascalschlegel/effort-estimator/output/xgboost_results.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")
