#!/usr/bin/env python3
"""
XGBoost with per-activity aggregation and Leave-One-Activity-Out CV.
This avoids temporal leakage by treating each activity as one sample.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import xgboost as xgb

DATA_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
ADL_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv")

print("="*70)
print("XGBoost: Per-Activity Aggregation + Leave-One-Out CV")
print("="*70)

# ============================================================================
# LOAD ADL DATA TO GET ACTIVITY BOUNDARIES
# ============================================================================
def parse_time(t):
    try:
        dt = datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None

adl_raw = pd.read_csv(ADL_PATH, skiprows=2)
adl_raw.columns = ['Time', 'ADLs', 'Effort']
adl_raw['timestamp'] = adl_raw['Time'].apply(parse_time)

# Get activity intervals (Start to End)
activities = []
current_activity = None
start_time = None

for _, row in adl_raw.iterrows():
    if pd.isna(row['timestamp']):
        continue
    if 'Start' in str(row['ADLs']):
        current_activity = row['ADLs'].replace(' Start', '')
        start_time = row['timestamp']
    elif 'End' in str(row['ADLs']) and current_activity is not None:
        activities.append({
            'activity': current_activity,
            't_start': start_time,
            't_end': row['timestamp'],
            'borg': row['Effort']
        })
        current_activity = None

activities_df = pd.DataFrame(activities)
activities_df['borg'] = pd.to_numeric(activities_df['borg'], errors='coerce')
activities_df = activities_df.dropna(subset=['borg'])
print(f"Found {len(activities_df)} activities with Borg ratings")

# ============================================================================
# LOAD WINDOW DATA AND ALIGN TIMESTAMPS
# ============================================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} windows")

# Compute offset between fused and ADL timestamps
fused_start = df['t_center'].min()
adl_start = activities_df['t_start'].min()
offset = adl_start - fused_start
print(f"Timestamp offset: {offset:.1f}s ({offset/3600:.1f}h)")

# Shift activities to match fused timestamps
activities_df['t_start_adj'] = activities_df['t_start'] - offset
activities_df['t_end_adj'] = activities_df['t_end'] - offset

# Assign each window to an activity based on t_center
def assign_activity(t_center, idx):
    for i, act in activities_df.iterrows():
        if act['t_start_adj'] <= t_center <= act['t_end_adj']:
            return act['activity'], i
    return None, None

df['activity'] = None
df['activity_idx'] = None
for idx in df.index:
    act, act_idx = assign_activity(df.loc[idx, 't_center'], idx)
    df.loc[idx, 'activity'] = act
    df.loc[idx, 'activity_idx'] = act_idx

df = df[df['activity'].notna()].copy()
print(f"Windows with activity: {len(df)}")
print(f"Unique activities: {df['activity'].nunique()}")

# Get feature columns
meta_cols = ['t_center', 'borg', 'borg_cr10', 'activity', 'activity_id', 'activity_idx', 'time_diff',
             'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 'win_sec_r', 
             't_start', 't_end', 'window_id', 'modality', 'start_idx', 'end_idx']
feature_cols = [c for c in df.columns if c not in meta_cols 
                and not c.startswith('Unnamed') 
                and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"Feature columns: {len(feature_cols)}")

# ============================================================================
# AGGREGATE TO PER-ACTIVITY
# ============================================================================
print("\n" + "="*70)
print("AGGREGATING TO PER-ACTIVITY")
print("="*70)

# Group by activity_idx (unique activity index from activities_df)
# This ensures we use the Borg from activities_df, not from the fused file
agg_dict = {col: 'mean' for col in feature_cols}
agg_dict['activity'] = 'first'

activity_features = df.groupby('activity_idx').agg(agg_dict).reset_index()

# Merge with activities_df to get correct Borg
activity_features['activity_idx'] = activity_features['activity_idx'].astype(int)
activity_df = activity_features.merge(
    activities_df[['borg']].reset_index().rename(columns={'index': 'activity_idx'}),
    on='activity_idx'
)

print(f"Per-activity samples: {len(activity_df)}")

# Prepare X, y
X = activity_df[feature_cols].fillna(0)
y = activity_df['borg'].astype(float)

# Check for NaN in target
print(f"Target NaN count: {y.isna().sum()}")
if y.isna().any():
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    activity_df = activity_df[valid_mask]
    print(f"After removing NaN targets: {len(y)} activities")

# Remove constant columns
X = X.loc[:, X.std() > 1e-6]
print(f"Non-constant features: {len(X.columns)}")
print(f"Target range: {y.min():.1f} - {y.max():.1f}")

# ============================================================================
# LEAVE-ONE-ACTIVITY-OUT CROSS-VALIDATION
# ============================================================================
print("\n" + "="*70)
print("LEAVE-ONE-ACTIVITY-OUT CROSS-VALIDATION")
print("="*70)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# XGBoost model (simpler to avoid overfitting on small data)
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.5,
    random_state=42,
    n_jobs=-1
)

# Leave-one-out CV
loo = LeaveOneOut()
y_pred_cv = cross_val_predict(model, X_scaled, y, cv=loo)

# CV metrics
cv_r2 = r2_score(y, y_pred_cv)
cv_rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
cv_mae = mean_absolute_error(y, y_pred_cv)

print(f"\nLOO-CV Results (n={len(y)} activities):")
print(f"  R²:   {cv_r2:.4f}")
print(f"  RMSE: {cv_rmse:.4f}")
print(f"  MAE:  {cv_mae:.4f}")

# Also fit on all data to get feature importance
model.fit(X_scaled, y)
y_train_pred = model.predict(X_scaled)
train_r2 = r2_score(y, y_train_pred)

print(f"\nTrain (all data): R² = {train_r2:.4f}")
print(f"CV (held-out):    R² = {cv_r2:.4f}")
print(f"Gap:              {train_r2 - cv_r2:.4f}")

# ============================================================================
# TOP FEATURES
# ============================================================================
print("\n" + "="*70)
print("TOP 15 FEATURES")
print("="*70)

importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")

# ============================================================================
# COMPARISON WITH LINEAR FORMULA
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print("""
Method                      | Samples | CV R²  | Notes
----------------------------|---------|--------|---------------------------
XGBoost (window, 70% OL)    | 429     | ~0.66  | Temporal leakage
XGBoost (window, 10% OL)    | 143     | ~0.68  | Still leaks within activity
XGBoost (per-activity, LOO) | {:<3}     | {:.2f}   | No leakage ✓
Linear formula (HR×√dur)    | 32      | 0.82   | No leakage ✓
""".format(len(y), cv_r2))

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================
results = activity_df[['activity', 'borg']].copy()
results['predicted'] = y_pred_cv
results['error'] = results['predicted'] - results['borg']
results.to_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/xgboost_per_activity_predictions.csv', index=False)
print(f"\n✓ Predictions saved")

# Show worst predictions
print("\nWorst predictions (|error| > 1):")
worst = results[abs(results['error']) > 1].sort_values('error', key=abs, ascending=False)
for _, row in worst.head(5).iterrows():
    print(f"  {row['activity'][:30]:30s} True={row['borg']:.1f} Pred={row['predicted']:.1f} Err={row['error']:+.1f}")
