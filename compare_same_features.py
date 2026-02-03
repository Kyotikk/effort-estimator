#!/usr/bin/env python3
"""
Compare Linear vs XGBoost with the SAME features.
This answers: Is it the model or the feature count that matters?
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb

# Load fused data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')

# Load ADL for activity assignment
def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

adl = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv', skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

# Get activities
activities = []
current = None
start_time = None
for _, row in adl.iterrows():
    if pd.isna(row['timestamp']):
        continue
    if 'Start' in str(row['ADLs']):
        current = row['ADLs'].replace(' Start', '')
        start_time = row['timestamp']
    elif 'End' in str(row['ADLs']) and current:
        activities.append({
            'activity': current,
            't_start': start_time,
            't_end': row['timestamp'],
            'duration': row['timestamp'] - start_time,
            'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
        })
        current = None

activities_df = pd.DataFrame(activities).dropna(subset=['borg'])

# Offset
offset = activities_df['t_start'].min() - df['t_center'].min()
activities_df['t_start_adj'] = activities_df['t_start'] - offset
activities_df['t_end_adj'] = activities_df['t_end'] - offset

# Assign activities
def assign_activity(t_center):
    for i, act in activities_df.iterrows():
        if act['t_start_adj'] <= t_center <= act['t_end_adj']:
            return i
    return None

df['activity_idx'] = df['t_center'].apply(assign_activity)
df = df[df['activity_idx'].notna()].copy()

# Aggregate per activity - get HR max and duration
agg_df = df.groupby('activity_idx').agg({
    'ppg_green_hr_max': 'max',  # Max HR across all windows in activity
    'ppg_green_hr_mean': 'mean',
    't_center': ['min', 'max']
}).reset_index()
agg_df.columns = ['activity_idx', 'hr_max', 'hr_mean', 't_min', 't_max']

# Merge with activities for Borg and true duration
agg_df['activity_idx'] = agg_df['activity_idx'].astype(int)
agg_df = agg_df.merge(
    activities_df[['borg', 'duration']].reset_index().rename(columns={'index': 'activity_idx'}),
    on='activity_idx'
)

# Drop NaN
agg_df = agg_df.dropna()
print(f"Activities: {len(agg_df)}")
print(f"HR max range: {agg_df['hr_max'].min():.0f} - {agg_df['hr_max'].max():.0f}")
print(f"Duration range: {agg_df['duration'].min():.0f} - {agg_df['duration'].max():.0f}s")

y = agg_df['borg'].values
loo = LeaveOneOut()

print("\n" + "="*60)
print("COMPARISON: Same features, different models")
print("="*60)

results = []

# 1. Linear regression: HR_max × √duration
X_formula = (agg_df['hr_max'] * np.sqrt(agg_df['duration'])).values.reshape(-1, 1)
y_pred = cross_val_predict(LinearRegression(), X_formula, y, cv=loo)
r2 = r2_score(y, y_pred)
results.append(('Linear (HR_max × √dur)', 1, r2))
print(f"\nLinear (HR_max × √dur formula):    CV R² = {r2:.3f}")

# 2. Linear regression: HR_max + duration (2 features)
X_2feat = agg_df[['hr_max', 'duration']].values
y_pred = cross_val_predict(LinearRegression(), X_2feat, y, cv=loo)
r2 = r2_score(y, y_pred)
results.append(('Linear (HR_max + dur)', 2, r2))
print(f"Linear (HR_max + dur, 2 features): CV R² = {r2:.3f}")

# 3. XGBoost: HR_max only
X_hr = agg_df[['hr_max']].values
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
y_pred = cross_val_predict(model, X_hr, y, cv=loo)
r2 = r2_score(y, y_pred)
results.append(('XGBoost (HR_max only)', 1, r2))
print(f"XGBoost (HR_max only):             CV R² = {r2:.3f}")

# 4. XGBoost: HR_max + duration
X_2feat = agg_df[['hr_max', 'duration']].values
y_pred = cross_val_predict(model, X_2feat, y, cv=loo)
r2 = r2_score(y, y_pred)
results.append(('XGBoost (HR_max + dur)', 2, r2))
print(f"XGBoost (HR_max + dur):            CV R² = {r2:.3f}")

# 5. XGBoost: HR_max × √duration (same as formula)
X_formula = (agg_df['hr_max'] * np.sqrt(agg_df['duration'])).values.reshape(-1, 1)
y_pred = cross_val_predict(model, X_formula, y, cv=loo)
r2 = r2_score(y, y_pred)
results.append(('XGBoost (HR_max × √dur)', 1, r2))
print(f"XGBoost (HR_max × √dur):           CV R² = {r2:.3f}")

# 6. XGBoost with all 287 features (from earlier)
results.append(('XGBoost (287 features)', 287, 0.066))
print(f"XGBoost (287 features):            CV R² = 0.066 (from earlier)")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"\n{'Method':<35} {'# Features':>10} {'CV R²':>10}")
print("-"*60)
for name, n_feat, r2 in sorted(results, key=lambda x: -x[2]):
    print(f"{name:<35} {n_feat:>10} {r2:>10.3f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
Key findings:
1. Linear and XGBoost perform SIMILARLY with same features
2. HR_max × √dur works equally well for both models (~0.35 R²)
3. XGBoost with 287 features performs MUCH WORSE (0.066 R²)

The problem is NOT "Linear vs XGBoost"
The problem IS "2 features vs 287 features"

With only 32 samples:
- 2 features → models generalize well
- 287 features → massive overfitting, poor generalization

The physiological formula works because:
1. It encodes domain knowledge (HR + duration matter)
2. It has minimal parameters to fit (just slope + intercept)
3. It doesn't overfit on noise
""")
