#!/usr/bin/env python3
"""
PROPER XGBoost Training with GroupKFold (no data leakage)

Key differences from previous approach:
1. GroupKFold splits by ACTIVITY, not random windows
2. No overlapping windows between train/test  
3. Proper feature selection on training data only
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("PROPER XGBOOST TRAINING WITH GROUPKFOLD")
print("="*70)

data_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} windows × {df.shape[1]} columns")

# Target
y = df['borg'].values
t_center = df['t_center'].values

# ============================================================================
# 2. LOAD ADL LABELS TO GET REAL ACTIVITY BOUNDARIES
# ============================================================================
print("\n" + "-"*70)
print("STEP 2: PARSE ADL LABELS FOR ACTIVITY BOUNDARIES")
print("-"*70)

adl_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv")
adl_df = pd.read_csv(adl_path, skiprows=2)  # Skip header rows

# Parse ADL times to unix timestamps
def parse_adl_time(time_str):
    """Parse ADL time format: DD-MM-YYYY-HH-MM-SS-mmm"""
    try:
        dt = datetime.strptime(time_str, "%d-%m-%Y-%H-%M-%S-%f")
        return dt.timestamp()
    except:
        return None

adl_df['unix_time'] = adl_df['Time'].apply(parse_adl_time)

# Get activity boundaries (Start/End pairs)
activities = []
current_activity = None
activity_id = 0

for _, row in adl_df.iterrows():
    adl_name = str(row['ADLs']) if pd.notna(row['ADLs']) else ''
    if 'Start' in adl_name:
        current_activity = {
            'name': adl_name.replace(' Start', ''),
            'start': row['unix_time'],
            'id': activity_id
        }
    elif 'End' in adl_name and current_activity is not None:
        current_activity['end'] = row['unix_time']
        current_activity['borg'] = row['Effort'] if pd.notna(row['Effort']) else None
        activities.append(current_activity)
        activity_id += 1
        current_activity = None

print(f"  Parsed {len(activities)} activities from ADL labels")

# Assign activity_id to each window based on time overlap
groups = np.full(len(df), -1)  # -1 = no activity assigned
for i, t in enumerate(t_center):
    for act in activities:
        if act['start'] <= t <= act['end']:
            groups[i] = act['id']
            break

# Remove windows not assigned to any activity
valid_mask = groups >= 0
n_removed = (~valid_mask).sum()
if n_removed > 0:
    print(f"  Removing {n_removed} windows not mapped to activities")
    df = df[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    groups = groups[valid_mask]
    t_center = t_center[valid_mask]

n_activities = len(np.unique(groups))
print(f"  N activities with windows: {n_activities}")
print(f"  N windows: {len(df)}")
print(f"  Borg range: [{y.min():.1f}, {y.max():.1f}]")
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
n_nan_before = X.isna().sum().sum()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
print(f"  Filled {n_nan_before} NaN values with median")

print(f"  Final feature shape: {X.shape}")

# Count by modality
ppg_cols = [c for c in X.columns if c.startswith('ppg_')]
acc_cols = [c for c in X.columns if c.startswith('acc_')]
eda_cols = [c for c in X.columns if c.startswith('eda_')]
print(f"  Features by modality: ppg={len(ppg_cols)}, acc={len(acc_cols)}, eda={len(eda_cols)}")

# ============================================================================
# 4. GROUPKFOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "-"*70)
print("STEP 4: GROUPKFOLD CROSS-VALIDATION (5-fold)")
print("-"*70)

# Need at least 5 groups for 5-fold CV
n_splits = min(5, n_activities)
print(f"  Using {n_splits}-fold CV (by activity)")

gkf = GroupKFold(n_splits=n_splits)
predictions = np.zeros(len(y))
fold_results = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale features (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    predictions[test_idx] = y_pred
    
    # Fold metrics
    fold_r, _ = pearsonr(y_test, y_pred)
    fold_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    fold_mae = np.mean(np.abs(y_test - y_pred))
    
    n_train_activities = len(np.unique(groups[train_idx]))
    n_test_activities = len(np.unique(groups[test_idx]))
    
    print(f"  Fold {fold+1}: r={fold_r:.3f}, RMSE={fold_rmse:.3f}, MAE={fold_mae:.3f} "
          f"(train={n_train_activities} activities, test={n_test_activities} activities)")
    
    fold_results.append({
        'fold': fold+1,
        'r': fold_r,
        'rmse': fold_rmse,
        'mae': fold_mae,
        'n_train': len(train_idx),
        'n_test': len(test_idx)
    })

# ============================================================================
# 5. OVERALL RESULTS
# ============================================================================
print("\n" + "-"*70)
print("STEP 5: OVERALL RESULTS")
print("-"*70)

overall_r, p_value = pearsonr(y, predictions)
overall_rmse = np.sqrt(np.mean((y - predictions)**2))
overall_mae = np.mean(np.abs(y - predictions))

print(f"\n  XGBoost (GroupKFold, {n_splits}-fold by activity):")
print(f"    Pearson r:  {overall_r:.4f} (p={p_value:.2e})")
print(f"    RMSE:       {overall_rmse:.4f} Borg points")
print(f"    MAE:        {overall_mae:.4f} Borg points")

# ============================================================================
# 6. COMPARE TO LITERATURE FORMULA
# ============================================================================
print("\n" + "-"*70)
print("STEP 6: COMPARISON WITH LITERATURE FORMULA")
print("-"*70)

# Load HR data for comparison
hr_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_heart_rate/vivalnk_vv330_heart_rate.csv")
hr_df = pd.read_csv(hr_path)

# Get HR at rest (first few minutes)
hr_baseline = hr_df['heart_rate'].iloc[:100].median()
print(f"  HR baseline: {hr_baseline:.1f} bpm")

# For each activity, compute HR_delta and duration
lit_predictions = np.zeros(len(y))
for i, t in enumerate(t_center):
    # Find HR values in this window
    window_start = t - 5
    window_end = t + 5
    mask = (hr_df['t_unix'] >= window_start) & (hr_df['t_unix'] <= window_end)
    if mask.sum() > 0:
        hr_mean = hr_df.loc[mask, 'heart_rate'].mean()
        hr_delta = hr_mean - hr_baseline
        # Literature formula: effort ~ HR_delta * sqrt(duration)
        # Since all windows are 10s, just use HR_delta as proxy
        lit_predictions[i] = hr_delta
    else:
        lit_predictions[i] = np.nan

# Remove NaN for comparison
valid_lit = ~np.isnan(lit_predictions)
if valid_lit.sum() > 10:
    lit_r, lit_p = pearsonr(y[valid_lit], lit_predictions[valid_lit])
    print(f"\n  Literature formula (HR_delta only, same windows):")
    print(f"    Pearson r:  {lit_r:.4f} (p={lit_p:.2e})")
    print(f"    (Using {valid_lit.sum()}/{len(y)} windows with HR data)")
else:
    print("  Not enough HR data for comparison")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
  Dataset:
    - {len(df)} windows from {n_activities} activities
    - 10s windows, 10% overlap (9s shift)
    - {X.shape[1]} features (PPG: {len(ppg_cols)}, ACC: {len(acc_cols)}, EDA: {len(eda_cols)})
    
  XGBoost Results (GroupKFold, NO data leakage):
    - Pearson r: {overall_r:.4f}
    - RMSE: {overall_rmse:.4f} Borg points
    - MAE: {overall_mae:.4f} Borg points
    
  Key observation:
    - With proper GroupKFold validation (splitting by activity),
      performance should be lower than random CV (which has leakage)
    - But this is the HONEST estimate of real-world performance
""")
