#!/usr/bin/env python3
"""
CLEAN SCIENTIFIC APPROACH TO EFFORT ESTIMATION
Without HR_max (unknown) - using only measurable quantities

Based on literature:
1. %HRR simplified: (HR - HR_rest) / HR_rest × 100 = relative HR elevation
2. HRV decrease: Lower HRV = higher effort (RMSSD, SDNN)
3. Movement intensity: Accelerometer above resting
4. HRV Recovery Rate: Speed of autonomic recovery (from your earlier pipeline!)

Key insight from your plots:
- HRV Recovery Rate prediction: Test R²=0.89 (XGBoost)
- Borg prediction: Test R²=0.9574 BUT with window overlap leakage
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

print("="*70)
print("SCIENTIFIC EFFORT FEATURES (no HR_max needed)")
print("="*70)

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
acc['magnitude'] = np.sqrt(acc['accX']**2 + acc['accY']**2 + acc['accZ']**2)

# Compute offsets
adl_start = adl['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()

# Estimate baselines (at rest)
HR_rest = hr['heart_rate'].quantile(0.05)  # 5th percentile = resting HR
ACC_rest = acc['magnitude'].quantile(0.10)  # Gravity only
print(f"\nBaselines: HR_rest = {HR_rest:.0f} bpm, ACC_rest = {ACC_rest:.0f}")

# Parse activities with SCIENTIFIC features (no HR_max!)
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
        
        # Get HR during activity
        t_start = start_time - hr_offset
        t_end = row['timestamp'] - hr_offset
        mask = (hr['timestamp'] >= t_start) & (hr['timestamp'] <= t_end)
        hr_vals = hr.loc[mask, 'heart_rate'].values
        
        # Get ACC during activity
        t_start_acc = start_time - acc_offset
        t_end_acc = row['timestamp'] - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask, 'magnitude'].values
        
        if len(hr_vals) < 2 or len(acc_vals) < 10:
            current = None
            continue
        
        # === FEATURES WITHOUT HR_MAX ===
        
        # 1. Relative HR elevation (no HR_max needed!)
        #    % above resting: (HR - HR_rest) / HR_rest × 100
        hr_mean = hr_vals.mean()
        hr_elevation = (hr_mean - HR_rest) / HR_rest * 100  # % above rest
        hr_peak_elevation = (hr_vals.max() - HR_rest) / HR_rest * 100
        
        # 2. HR variability during activity (proxy for HRV from HR)
        #    Lower std = higher effort (less autonomic flexibility)
        hr_std = hr_vals.std()
        hr_range = hr_vals.max() - hr_vals.min()
        
        # 3. Movement intensity (above resting)
        acc_mean = acc_vals.mean()
        movement = max(0, acc_mean - ACC_rest)  # Above gravity baseline
        movement_peak = max(0, acc_vals.max() - ACC_rest)
        movement_std = acc_vals.std()  # Variability of movement
        
        # 4. Combined physiological load (HR × movement)
        physio_load = hr_elevation * movement / 100  # Scaled
        
        activities.append({
            'activity': current,
            'duration': duration,
            # HR-based (no HR_max!)
            'hr_elevation': hr_elevation,        # % above resting
            'hr_peak_elevation': hr_peak_elevation,
            'hr_std': hr_std,                    # HR variability
            'hr_range': hr_range,
            # Movement-based
            'movement': movement,                # Above-rest movement
            'movement_peak': movement_peak,
            'movement_std': movement_std,
            # Combined
            'physio_load': physio_load,
            # Target
            'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
        })
        current = None

df = pd.DataFrame(activities).dropna()
print(f"Activities: {len(df)}")
print(f"Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")

# Correlation analysis
print("\n" + "="*70)
print("CORRELATIONS WITH BORG (no HR_max features)")
print("="*70)

features = ['hr_elevation', 'hr_peak_elevation', 'hr_std', 'hr_range',
            'movement', 'movement_peak', 'movement_std', 'physio_load', 'duration']

for feat in features:
    r, p = pearsonr(df[feat], df['borg'])
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {feat:<22} r = {r:>6.3f}  p = {p:.4f} {sig}")

# LOO-CV comparison
print("\n" + "="*70)
print("LOO-CV R² (per-activity, no leakage)")
print("="*70)

y = df['borg'].values
loo = LeaveOneOut()

feature_sets = [
    # Single features
    ('HR elevation only', ['hr_elevation']),
    ('HR peak elevation', ['hr_peak_elevation']),
    ('Movement only', ['movement']),
    ('Physio load (HR×movement)', ['physio_load']),
    ('Duration only', ['duration']),
    # Combinations (no duration)
    ('HR + Movement', ['hr_elevation', 'movement']),
    ('HR peak + Movement peak', ['hr_peak_elevation', 'movement_peak']),
    ('All intensity features', ['hr_elevation', 'hr_peak_elevation', 'movement', 'movement_peak']),
    # With duration
    ('HR elevation + Duration', ['hr_elevation', 'duration']),
    ('All features', features),
]

results = []
for name, feats in feature_sets:
    X = df[feats].values
    y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    results.append((name, r2, 'duration' in feats))

print("\nWithout duration (pure intensity):")
print("-"*50)
for name, r2, has_dur in sorted(results, key=lambda x: -x[1]):
    if not has_dur:
        print(f"  {name:<30} CV R² = {r2:.3f}")

print("\nWith duration:")
print("-"*50)
for name, r2, has_dur in sorted(results, key=lambda x: -x[1]):
    if has_dur:
        print(f"  {name:<30} CV R² = {r2:.3f}")

# Show actual predictions for best model
print("\n" + "="*70)
print("BEST MODEL PREDICTIONS")
print("="*70)

# Use HR peak elevation (best single intensity feature based on correlation)
X = df[['hr_peak_elevation', 'movement']].values
y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)

df['predicted'] = y_pred
df['error'] = df['borg'] - df['predicted']

print("\nActivity predictions (HR_peak_elevation + Movement):")
print("-"*70)
print(f"{'Activity':<25} {'Borg':>6} {'Pred':>6} {'Error':>7} {'HR_elev':>8}")
for _, row in df.sort_values('error', key=abs, ascending=False).iterrows():
    print(f"{row['activity'][:24]:<25} {row['borg']:>6.1f} {row['predicted']:>6.1f} {row['error']:>+7.2f} {row['hr_peak_elevation']:>8.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Scientific approach WITHOUT HR_max:

1. HR Elevation = (HR - HR_rest) / HR_rest × 100%
   → Measures cardiovascular demand relative to baseline
   → No need for age-predicted max!

2. Movement Intensity = ACC - ACC_rest
   → Above-baseline movement (gravity-corrected)

3. Physio Load = HR_elev × Movement
   → Combined cardio + movement effort

Results:
- Best intensity-only: CV R² ≈ {max(r2 for _, r2, d in results if not d):.3f}
- With duration:       CV R² ≈ {max(r2 for _, r2, d in results if d):.3f}

Key insight: For THIS elderly population, duration captures real 
physiological fatigue (sustaining effort IS hard for them).

For healthy population, intensity features should work better!
""")
