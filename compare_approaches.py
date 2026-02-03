#!/usr/bin/env python3
"""Compare Window-based vs Activity-based approaches."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

print("="*70)
print("TWO APPROACHES COMPARISON")
print("="*70)

# Load all subjects
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'P{i}'
        dfs.append(df)
        print(f"P{i}: {len(df)} windows")

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total windows: {len(df_all)}")

# Create activity-level data
print("\nAggregating to ACTIVITY level...")
activity_data = []
for subj in df_all['subject'].unique():
    subj_df = df_all[df_all['subject'] == subj].copy().reset_index(drop=True)
    subj_df['borg_change'] = subj_df['borg'].diff().fillna(1) != 0
    subj_df['activity_id'] = subj_df['borg_change'].cumsum()
    
    imu_cols = [c for c in subj_df.columns if 'acc_' in c and '_r' not in c][:30]
    
    for act_id in subj_df['activity_id'].unique():
        act_df = subj_df[subj_df['activity_id'] == act_id]
        row = {'subject': subj, 'borg': act_df['borg'].iloc[0], 'n_windows': len(act_df)}
        for col in imu_cols:
            row[col] = act_df[col].mean()
        activity_data.append(row)

df_activity = pd.DataFrame(activity_data)
print(f"Total activities: {len(df_activity)}")

# LOSO for activity-based
print("\n" + "="*70)
print("ACTIVITY-BASED LOSO")
print("="*70)

# Drop rows with NaN borg
df_activity = df_activity.dropna(subset=['borg'])
print(f"Activities with valid Borg: {len(df_activity)}")

imu_cols = [c for c in df_activity.columns if 'acc_' in c]
activity_results = []
for test_subj in df_activity['subject'].unique():
    train = df_activity[df_activity['subject'] != test_subj].dropna(subset=['borg'])
    test = df_activity[df_activity['subject'] == test_subj].dropna(subset=['borg'])
    
    X_train = train[imu_cols].fillna(0)
    y_train = train['borg'].values
    X_test = test[imu_cols].fillna(0)
    y_test = test['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    if y_test.std() > 0:
        r, _ = pearsonr(y_test, y_pred)
        activity_results.append(r)
        print(f"  {test_subj}: r = {r:.3f} (n={len(test)} activities)")

mean_r = np.mean(activity_results)
print(f"\nACTIVITY-BASED LOSO mean r = {mean_r:.3f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"""
| Approach        | Unit       | N samples | LOSO r |
|-----------------|------------|-----------|--------|
| Window-based    | 5s window  | ~1400     | 0.55   |
| Activity-based  | Activity   | {len(df_activity)}        | {mean_r:.2f}   |
| Screenshot      | Activity   | ~100      | 0.84*  |

* Screenshot was POOLED (no cross-validation) - inflated!
""")
