#!/usr/bin/env python3
"""
ML PIPELINE WITH CORRECT SPLIT (No Leakage)

Instead of random window split, use:
- GroupKFold: All windows from same activity stay together
- This prevents adjacent overlapping windows from leaking between train/test
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

print("="*80)
print("ML PIPELINE WITH CORRECT SPLIT (GROUP BY ACTIVITY)")
print("="*80)

#==============================================================================
# STEP 1: Load raw data
#==============================================================================
print("\n▶ Loading raw data...")

hr = pd.read_csv(BASE_PATH / "vivalnk_vv330_heart_rate" / "data_1.csv.gz")
hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]

acc_files = list((BASE_PATH / "corsano_wrist_acc").glob("*.csv.gz"))
acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
acc = acc.rename(columns={'time': 'timestamp'})
acc['magnitude'] = np.sqrt(acc['accX']**2 + acc['accY']**2 + acc['accZ']**2)

adl = pd.read_csv(BASE_PATH / "scai_app" / "ADLs_1.csv", skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

# Time alignment
adl_start = adl['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()

print(f"  HR: {len(hr)} samples")
print(f"  ACC: {len(acc)} samples")

#==============================================================================
# STEP 2: Create windows WITH activity labels
#==============================================================================
print("\n▶ Creating windows with activity grouping...")

WINDOW_SEC = 10.0
OVERLAP = 0.7  # 70% overlap
STEP_SEC = WINDOW_SEC * (1 - OVERLAP)  # 3 seconds

# Parse activity intervals
activities = []
current = None
start_time = None
activity_id = 0

for _, row in adl.iterrows():
    if pd.isna(row['timestamp']):
        continue
    if 'Start' in str(row['ADLs']):
        current = row['ADLs'].replace(' Start', '')
        start_time = row['timestamp']
    elif 'End' in str(row['ADLs']) and current:
        end_time = row['timestamp']
        borg = float(row['Effort']) if pd.notna(row['Effort']) else np.nan
        
        if not np.isnan(borg):
            activities.append({
                'activity_id': activity_id,
                'activity': current,
                't_start': start_time,
                't_end': end_time,
                'borg': borg
            })
            activity_id += 1
        current = None

print(f"  Found {len(activities)} labeled activities")

# Create windows within each activity
windows = []
for act in activities:
    t = act['t_start']
    while t + WINDOW_SEC <= act['t_end']:
        # Get HR for this window
        t_start_hr = t - hr_offset
        t_end_hr = t + WINDOW_SEC - hr_offset
        mask = (hr['timestamp'] >= t_start_hr) & (hr['timestamp'] <= t_end_hr)
        hr_vals = hr.loc[mask, 'heart_rate'].values
        
        # Get ACC for this window
        t_start_acc = t - acc_offset
        t_end_acc = t + WINDOW_SEC - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask]
        
        if len(hr_vals) >= 5 and len(acc_vals) >= 50:
            acc_mag = acc_vals['magnitude'].values
            
            # Compute features for this window
            windows.append({
                'activity_id': act['activity_id'],  # GROUP KEY for proper splitting!
                'activity': act['activity'],
                'borg': act['borg'],
                't_center': t + WINDOW_SEC/2,
                # HR features
                'hr_mean': hr_vals.mean(),
                'hr_std': hr_vals.std(),
                'hr_min': hr_vals.min(),
                'hr_max': hr_vals.max(),
                'hr_range': hr_vals.max() - hr_vals.min(),
                # ACC features
                'acc_mean': acc_mag.mean(),
                'acc_std': acc_mag.std(),
                'acc_min': acc_mag.min(),
                'acc_max': acc_mag.max(),
                'acc_range': acc_mag.max() - acc_mag.min(),
                # ACC axis features
                'accX_mean': acc_vals['accX'].mean(),
                'accX_std': acc_vals['accX'].std(),
                'accY_mean': acc_vals['accY'].mean(),
                'accY_std': acc_vals['accY'].std(),
                'accZ_mean': acc_vals['accZ'].mean(),
                'accZ_std': acc_vals['accZ'].std(),
            })
        
        t += STEP_SEC

df_windows = pd.DataFrame(windows)
print(f"  Created {len(df_windows)} windows from {df_windows['activity_id'].nunique()} activities")
print(f"  Windows per activity: {len(df_windows) / df_windows['activity_id'].nunique():.1f} average")

# Show distribution
print(f"\n  Activity breakdown:")
for act_id in df_windows['activity_id'].unique()[:5]:
    subset = df_windows[df_windows['activity_id'] == act_id]
    print(f"    Activity {act_id} ({subset['activity'].iloc[0][:15]}): {len(subset)} windows, Borg={subset['borg'].iloc[0]}")
print("    ...")

#==============================================================================
# STEP 3: Feature columns
#==============================================================================
feature_cols = [
    'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
    'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'acc_range',
    'accX_mean', 'accX_std', 'accY_mean', 'accY_std', 'accZ_mean', 'accZ_std'
]

X = df_windows[feature_cols].values
y = df_windows['borg'].values
groups = df_windows['activity_id'].values  # KEY: Group by activity!

print(f"\n▶ Features: {len(feature_cols)}")
print(f"▶ Samples (windows): {len(X)}")
print(f"▶ Groups (activities): {len(np.unique(groups))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#==============================================================================
# STEP 4: CORRECT SPLIT - Leave-One-Activity-Out (GroupKFold)
#==============================================================================
print("\n" + "="*80)
print("COMPARISON: WRONG vs CORRECT SPLIT")
print("="*80)

# WRONG WAY: Random split (causes leakage)
print("\n❌ WRONG: Random window split (like original pipeline)")
print("-"*60)

from sklearn.model_selection import KFold
kf_random = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost with random split
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
y_pred_random = cross_val_predict(xgb_model, X_scaled, y, cv=kf_random)
r2_random = r2_score(y, y_pred_random)
mae_random = mean_absolute_error(y, y_pred_random)
print(f"  XGBoost (random 5-fold):  CV R² = {r2_random:.3f}, MAE = {mae_random:.2f}")

# Ridge with random split
y_pred_ridge_random = cross_val_predict(Ridge(alpha=1.0), X_scaled, y, cv=kf_random)
r2_ridge_random = r2_score(y, y_pred_ridge_random)
mae_ridge_random = mean_absolute_error(y, y_pred_ridge_random)
print(f"  Ridge (random 5-fold):    CV R² = {r2_ridge_random:.3f}, MAE = {mae_ridge_random:.2f}")

print("\n  ⚠️  These R² values are INFLATED due to leakage!")
print("     Adjacent windows share 70% of raw data.")

# CORRECT WAY: Group by activity
print("\n✓ CORRECT: GroupKFold (all windows from same activity together)")
print("-"*60)

# GroupKFold
gkf = GroupKFold(n_splits=5)

# XGBoost with group split
y_pred_group = cross_val_predict(xgb_model, X_scaled, y, cv=gkf, groups=groups)
r2_group = r2_score(y, y_pred_group)
mae_group = mean_absolute_error(y, y_pred_group)
print(f"  XGBoost (GroupKFold):     CV R² = {r2_group:.3f}, MAE = {mae_group:.2f}")

# Ridge with group split
y_pred_ridge_group = cross_val_predict(Ridge(alpha=1.0), X_scaled, y, cv=gkf, groups=groups)
r2_ridge_group = r2_score(y, y_pred_ridge_group)
mae_ridge_group = mean_absolute_error(y, y_pred_ridge_group)
print(f"  Ridge (GroupKFold):       CV R² = {r2_ridge_group:.3f}, MAE = {mae_ridge_group:.2f}")

# Leave-One-Activity-Out (most stringent)
print("\n✓ MOST CORRECT: Leave-One-Activity-Out (LOGO)")
print("-"*60)

logo = LeaveOneGroupOut()

# XGBoost with LOGO
y_pred_logo = cross_val_predict(xgb_model, X_scaled, y, cv=logo, groups=groups)
r2_logo = r2_score(y, y_pred_logo)
mae_logo = mean_absolute_error(y, y_pred_logo)
print(f"  XGBoost (LOGO):           CV R² = {r2_logo:.3f}, MAE = {mae_logo:.2f}")

# Ridge with LOGO
y_pred_ridge_logo = cross_val_predict(Ridge(alpha=1.0), X_scaled, y, cv=logo, groups=groups)
r2_ridge_logo = r2_score(y, y_pred_ridge_logo)
mae_ridge_logo = mean_absolute_error(y, y_pred_ridge_logo)
print(f"  Ridge (LOGO):             CV R² = {r2_ridge_logo:.3f}, MAE = {mae_ridge_logo:.2f}")

#==============================================================================
# STEP 5: Summary
#==============================================================================
print("\n" + "="*80)
print("SUMMARY: Impact of Correct Splitting")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│ SPLIT METHOD             │ XGBoost R²   │ Ridge R²    │ Status            │
├──────────────────────────┼──────────────┼─────────────┼───────────────────┤
│ Random KFold (WRONG)     │    {r2_random:>6.3f}     │   {r2_ridge_random:>6.3f}     │ ⚠️  DATA LEAKAGE   │
│ GroupKFold (CORRECT)     │    {r2_group:>6.3f}     │   {r2_ridge_group:>6.3f}     │ ✓  Valid           │
│ Leave-One-Group-Out      │    {r2_logo:>6.3f}     │   {r2_ridge_logo:>6.3f}     │ ✓  Most stringent  │
└──────────────────────────┴──────────────┴─────────────┴───────────────────┘

KEY INSIGHT:
  Random split R² = {r2_random:.3f}  →  GroupKFold R² = {r2_group:.3f}
  
  The drop of {r2_random - r2_group:.3f} R² is the "leakage inflation"!
  
  With proper splitting, XGBoost R² = {r2_logo:.3f} vs Ridge R² = {r2_ridge_logo:.3f}
  → Similar performance, but Ridge is simpler and more interpretable.
""")

#==============================================================================
# STEP 6: Visual explanation of the split
#==============================================================================
print("\n" + "="*80)
print("HOW THE CORRECT SPLIT WORKS")
print("="*80)

print("""
WRONG (Random Split):
  Activity 1: [W1, W2, W3, W4, W5]  ← Windows 70% overlapping
              Train    Test  Train  Test  Train
              ↑               ↑
              W1 and W3 share raw data! LEAKAGE!

CORRECT (GroupKFold by Activity):
  Fold 1: Train=[Act2, Act3, Act4, Act5], Test=[Act1]
  Fold 2: Train=[Act1, Act3, Act4, Act5], Test=[Act2]
  ...
  
  ALL windows from Activity 1 go to test together
  → No leakage because different activities don't share raw data!
""")

# Show actual fold composition
print("\nActual GroupKFold composition:")
print("-"*60)
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_activities = np.unique(groups[train_idx])
    test_activities = np.unique(groups[test_idx])
    print(f"  Fold {fold_idx+1}: Train {len(train_activities)} activities, Test {len(test_activities)} activities")
    print(f"          Test activities: {test_activities[:5].tolist()}{'...' if len(test_activities) > 5 else ''}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
For the ML pipeline to be valid, you MUST use GroupKFold or LeaveOneGroupOut
where groups = activity_id.

This ensures:
  1. All overlapping windows from same activity stay together
  2. Test set contains ONLY windows from unseen activities
  3. No data leakage between train and test

The correct R² for this ML pipeline with {len(feature_cols)} features is:
  XGBoost: CV R² = {r2_logo:.3f}
  Ridge:   CV R² = {r2_ridge_logo:.3f}

NOT the inflated R² = 0.95+ from random splitting!
""")
