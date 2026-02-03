#!/usr/bin/env python3
"""
Investigate: Why is duration so predictive?
Resting for a long time shouldn't be hard!
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

# Load elderly3 data
subject = 'sim_elderly3'
subj_path = BASE_PATH / subject

# Load ADL
adl = pd.read_csv(subj_path / "scai_app" / "ADLs_1.csv", skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

# Load wrist ACC
acc_files = list((subj_path / "corsano_wrist_acc").glob("*.csv.gz"))
acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
acc = acc.rename(columns={'time': 'timestamp'})

# Load HR
hr = pd.read_csv(subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz")
hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]

# Compute offsets
adl_start = adl['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()

# Parse activities with features
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
        duration = row['timestamp'] - start_time
        
        # Get ACC data
        t_start_acc = start_time - acc_offset
        t_end_acc = row['timestamp'] - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask]
        
        acc_mag = np.nan
        if len(acc_vals) > 0 and 'accX' in acc_vals.columns:
            mag = np.sqrt(acc_vals['accX']**2 + acc_vals['accY']**2 + acc_vals['accZ']**2)
            acc_mag = mag.mean()
        
        # Get HR data
        t_start_hr = start_time - hr_offset
        t_end_hr = row['timestamp'] - hr_offset
        mask = (hr['timestamp'] >= t_start_hr) & (hr['timestamp'] <= t_end_hr)
        hr_vals = hr.loc[mask, 'heart_rate']
        
        hr_mean = hr_vals.mean() if len(hr_vals) > 0 else np.nan
        hr_max = hr_vals.max() if len(hr_vals) > 0 else np.nan
        
        activities.append({
            'activity': current,
            'duration': duration,
            'acc_mag': acc_mag,
            'hr_mean': hr_mean,
            'hr_max': hr_max,
            'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
        })
        current = None

df = pd.DataFrame(activities).dropna()

print("="*70)
print("INVESTIGATING: Why is duration so predictive?")
print("="*70)

# Sort by duration
df_sorted = df.sort_values('duration', ascending=False)

print("\n1. ACTIVITIES SORTED BY DURATION:")
print("-"*70)
print(f"{'Activity':<25} {'Duration':>10} {'Borg':>6} {'ACC_mag':>10} {'HR_max':>8}")
print("-"*70)
for _, row in df_sorted.iterrows():
    print(f"{row['activity'][:24]:<25} {row['duration']:>10.0f}s {row['borg']:>6.1f} {row['acc_mag']:>10.0f} {row['hr_max']:>8.0f}")

print("\n2. CORRELATIONS:")
print("-"*70)
r_dur, p = pearsonr(df['duration'], df['borg'])
print(f"Duration vs Borg:    r = {r_dur:.3f} (p = {p:.4f})")
r_acc, p = pearsonr(df['acc_mag'], df['borg'])
print(f"ACC_mag vs Borg:     r = {r_acc:.3f} (p = {p:.4f})")
r_hr, p = pearsonr(df['hr_max'], df['borg'])
print(f"HR_max vs Borg:      r = {r_hr:.3f} (p = {p:.4f})")

print("\n3. THE PROBLEM: Long activities with HIGH Borg:")
print("-"*70)
long_high = df[(df['duration'] > 60) & (df['borg'] > 3)]
print(f"Found {len(long_high)} activities > 60s with Borg > 3:")
for _, row in long_high.iterrows():
    print(f"  {row['activity']}: {row['duration']:.0f}s, Borg={row['borg']}")

print("\n4. RESTING activities:")
print("-"*70)
resting = df[df['activity'].str.contains('Rest|rest', na=False)]
if len(resting) > 0:
    for _, row in resting.iterrows():
        print(f"  {row['activity']}: {row['duration']:.0f}s, Borg={row['borg']}")
else:
    print("  No 'Resting' activities found")

# Check activities with "rest" or low movement
print("\n5. LOW MOVEMENT activities (ACC_mag < median):")
print("-"*70)
median_acc = df['acc_mag'].median()
low_movement = df[df['acc_mag'] < median_acc].sort_values('borg', ascending=False)
for _, row in low_movement.head(10).iterrows():
    print(f"  {row['activity'][:24]:<25} ACC={row['acc_mag']:.0f} Borg={row['borg']:.1f} Dur={row['duration']:.0f}s")

print("\n" + "="*70)
print("6. BETTER APPROACH: Use intensity, not duration")
print("="*70)

# Compute intensity-based features
df['intensity'] = df['acc_mag'] * df['hr_max']  # Movement × HR
df['effort_rate'] = df['acc_mag'] / df['duration']  # Movement per second

print("\nNew feature correlations:")
r, p = pearsonr(df['intensity'], df['borg'])
print(f"  Intensity (ACC × HR):     r = {r:.3f}")
r, p = pearsonr(df['effort_rate'], df['borg'])
print(f"  Effort rate (ACC/dur):    r = {r:.3f}")
r, p = pearsonr(df['acc_mag'], df['borg'])
print(f"  ACC_mag alone:            r = {r:.3f}")
r, p = pearsonr(df['hr_max'], df['borg'])
print(f"  HR_max alone:             r = {r:.3f}")

# LOO-CV comparison
y = df['borg'].values
loo = LeaveOneOut()

results = []

# Duration-based (current best)
X = df[['duration']].values
y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
results.append(('Duration only', r2_score(y, y_pred)))

# Intensity-based (no duration)
X = df[['acc_mag', 'hr_max']].values
y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
results.append(('ACC + HR (no duration)', r2_score(y, y_pred)))

X = df[['intensity']].values
y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
results.append(('Intensity (ACC × HR)', r2_score(y, y_pred)))

X = df[['acc_mag']].values
y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
results.append(('ACC_mag only', r2_score(y, y_pred)))

X = df[['hr_max']].values
y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
results.append(('HR_max only', r2_score(y, y_pred)))

# Combined
X = df[['acc_mag', 'hr_max', 'duration']].values
y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
results.append(('ACC + HR + Duration', r2_score(y, y_pred)))

print("\n7. CV R² COMPARISON (removing duration bias):")
print("-"*70)
for name, r2 in sorted(results, key=lambda x: -x[1]):
    print(f"  {name:<30} CV R² = {r2:.3f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The issue: Duration is confounded with activity type.
- Walking/Standing activities tend to be longer AND harder
- Short activities (buttons, sitting) tend to be easier

Without duration, we need intensity-based features:
- ACC magnitude (movement intensity)
- HR increase (cardiovascular response)
- ACC × HR (combined physical effort)

This is more physiologically meaningful than just duration!
""")
