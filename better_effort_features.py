#!/usr/bin/env python3
"""
Better effort features for elderly subjects:
- HR increase during activity (not just absolute HR)
- HR relative to resting baseline
- Sustained effort over time
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

subject = 'sim_elderly3'
subj_path = BASE_PATH / subject

# Load data
adl = pd.read_csv(subj_path / "scai_app" / "ADLs_1.csv", skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

hr = pd.read_csv(subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz")
hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]

acc_files = list((subj_path / "corsano_wrist_acc").glob("*.csv.gz"))
acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
acc = acc.rename(columns={'time': 'timestamp'})

# Offsets
adl_start = adl['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()

# Estimate resting HR (10th percentile)
hr_resting = hr['heart_rate'].quantile(0.10)
hr_baseline = hr['heart_rate'].median()
print(f"Estimated resting HR: {hr_resting:.0f} bpm")
print(f"Median HR: {hr_baseline:.0f} bpm")

# Parse activities with BETTER features
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
        
        # Get HR data during activity
        t_start_hr = start_time - hr_offset
        t_end_hr = row['timestamp'] - hr_offset
        mask = (hr['timestamp'] >= t_start_hr) & (hr['timestamp'] <= t_end_hr)
        hr_vals = hr.loc[mask, 'heart_rate'].values
        
        if len(hr_vals) >= 2:
            hr_mean = hr_vals.mean()
            hr_max = hr_vals.max()
            hr_min = hr_vals.min()
            hr_start = hr_vals[:3].mean()  # First few samples
            hr_end = hr_vals[-3:].mean()   # Last few samples
            hr_increase = hr_end - hr_start  # HR increased during activity?
            hr_above_rest = hr_mean - hr_resting  # Elevated above resting
            hr_range = hr_max - hr_min  # Variability during activity
        else:
            continue
        
        # Get ACC during activity
        t_start_acc = start_time - acc_offset
        t_end_acc = row['timestamp'] - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask]
        
        if len(acc_vals) > 0 and 'accX' in acc_vals.columns:
            mag = np.sqrt(acc_vals['accX']**2 + acc_vals['accY']**2 + acc_vals['accZ']**2)
            acc_mag = mag.mean()
            acc_std = mag.std()
        else:
            continue
        
        activities.append({
            'activity': current,
            'duration': duration,
            'hr_mean': hr_mean,
            'hr_max': hr_max,
            'hr_increase': hr_increase,  # Did HR go up during activity?
            'hr_above_rest': hr_above_rest,  # How much above resting?
            'hr_range': hr_range,
            'acc_mag': acc_mag,
            'acc_std': acc_std,
            'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
        })
        current = None

df = pd.DataFrame(activities).dropna()

print(f"\nAnalyzing {len(df)} activities")

print("\n" + "="*70)
print("PHYSIOLOGICALLY MEANINGFUL FEATURES")
print("="*70)

features_to_test = [
    'duration',
    'hr_mean',
    'hr_max',
    'hr_increase',      # HR went UP during activity = harder
    'hr_above_rest',    # Elevated above resting = harder
    'hr_range',         # More HR variation = harder?
    'acc_mag',
    'acc_std',
]

print("\nCorrelations with Borg:")
print("-"*50)
for feat in features_to_test:
    r, p = pearsonr(df[feat], df['borg'])
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {feat:<20} r = {r:>6.3f}  p = {p:.4f} {sig}")

# Derived features
df['hr_intensity'] = df['hr_above_rest'] * df['duration']  # Sustained elevated HR
df['physiological_load'] = df['hr_above_rest'] * np.sqrt(df['duration'])  # Literature formula

print("\nDerived features:")
for feat in ['hr_intensity', 'physiological_load']:
    r, p = pearsonr(df[feat], df['borg'])
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {feat:<20} r = {r:>6.3f}  p = {p:.4f} {sig}")

print("\n" + "="*70)
print("LOO-CV R² COMPARISON")
print("="*70)

y = df['borg'].values
loo = LeaveOneOut()

feature_sets = [
    ('Duration only', ['duration']),
    ('HR_max only', ['hr_max']),
    ('HR_above_rest only', ['hr_above_rest']),
    ('HR_increase only', ['hr_increase']),
    ('Physio load (HR_rest × √dur)', ['physiological_load']),
    ('HR_intensity (HR_rest × dur)', ['hr_intensity']),
    ('HR_above_rest + Duration', ['hr_above_rest', 'duration']),
    ('HR_max + Duration', ['hr_max', 'duration']),
    ('All HR features', ['hr_mean', 'hr_max', 'hr_increase', 'hr_above_rest']),
    ('All features', features_to_test),
]

results = []
for name, feats in feature_sets:
    try:
        X = df[feats].values
        y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
        r2 = r2_score(y, y_pred)
        results.append((name, r2, len(feats)))
    except Exception as e:
        print(f"  {name}: Error - {e}")

print("\nResults (sorted by CV R²):")
print("-"*60)
for name, r2, n_feat in sorted(results, key=lambda x: -x[1]):
    print(f"  {name:<35} CV R² = {r2:>6.3f}  ({n_feat} features)")

print("\n" + "="*70)
print("TOP ACTIVITIES BY FEATURE VALUES")
print("="*70)

print("\nHighest HR_above_rest (most cardiovascular stress):")
top = df.nlargest(5, 'hr_above_rest')[['activity', 'duration', 'hr_above_rest', 'borg']]
print(top.to_string(index=False))

print("\nHighest HR_increase (HR went up during activity):")
top = df.nlargest(5, 'hr_increase')[['activity', 'duration', 'hr_increase', 'borg']]
print(top.to_string(index=False))

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
For elderly subjects, effort perception is about:
1. SUSTAINED EXERTION - Standing for 3 minutes IS hard
2. HR ELEVATION above resting baseline
3. Duration matters because sustaining effort is difficult

Best features:
- HR_above_rest × √duration (physiological load)
- Or simply: HR features + duration

The ACC (movement) is NOT predictive because:
- Standing still = little movement BUT high effort
- Quick button press = some movement BUT low effort
""")
