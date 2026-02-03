#!/usr/bin/env python3
"""
How to get R² ~0.8? Let's investigate what's limiting performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def load_subject(subject):
    subj_path = BASE_PATH / subject
    
    # Find ADL file
    for fname in ['ADLs_1.csv', 'ADLs_1-3.csv', 'ADLs_1-2.csv']:
        test_path = subj_path / "scai_app" / fname
        if test_path.exists():
            try:
                test_df = pd.read_csv(test_path, skiprows=2, nrows=1)
                if len(test_df.columns) >= 3:
                    adl_path = test_path
                    break
            except:
                continue
    else:
        return None
    
    adl = pd.read_csv(adl_path, skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
    hr_path = subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz"
    hr = pd.read_csv(hr_path)
    hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
    hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]
    
    if len(hr) < 10:
        return None
    
    offset = adl['timestamp'].min() - hr['timestamp'].min()
    hr_rest = hr['heart_rate'].quantile(0.05)
    hr_max_obs = hr['heart_rate'].quantile(0.95)
    
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
            t_start_hr = start_time - offset
            t_end_hr = row['timestamp'] - offset
            mask = (hr['timestamp'] >= t_start_hr) & (hr['timestamp'] <= t_end_hr)
            hr_vals = hr.loc[mask, 'heart_rate']
            
            if len(hr_vals) > 0:
                duration = row['timestamp'] - start_time
                activities.append({
                    'subject': subject,
                    'activity': current,
                    'duration': duration,
                    'hr_mean': hr_vals.mean(),
                    'hr_max': hr_vals.max(),
                    'hr_min': hr_vals.min(),
                    'hr_std': hr_vals.std() if len(hr_vals) > 1 else 0,
                    'hr_rest': hr_rest,
                    'hr_max_obs': hr_max_obs,
                    # Derived features
                    'hr_reserve': hr_vals.mean() - hr_rest,
                    'pct_hr_reserve': (hr_vals.mean() - hr_rest) / (hr_max_obs - hr_rest) * 100,
                    'hr_increase': hr_vals.max() - hr_vals.iloc[0] if len(hr_vals) > 1 else 0,
                    'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
                })
            current = None
    
    return pd.DataFrame(activities).dropna()

# Load elderly3 (our best subject)
df = load_subject('sim_elderly3')
print("="*70)
print("INVESTIGATING PERFORMANCE LIMITS (elderly3)")
print("="*70)
print(f"Activities: {len(df)}")
print(f"Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")

# Check correlations with Borg
print("\n" + "-"*70)
print("CORRELATIONS WITH BORG (raw, not CV)")
print("-"*70)

features = ['duration', 'hr_mean', 'hr_max', 'hr_min', 'hr_std', 
            'hr_reserve', 'pct_hr_reserve', 'hr_increase']

for feat in features:
    r, p = pearsonr(df[feat], df['borg'])
    print(f"{feat:20s}  r = {r:+.3f}  (p = {p:.4f})")

# Derived formulas
df['hr_x_dur'] = df['hr_mean'] * df['duration']
df['hr_x_sqrt_dur'] = df['hr_mean'] * np.sqrt(df['duration'])
df['hrmax_x_sqrt_dur'] = df['hr_max'] * np.sqrt(df['duration'])
df['log_dur'] = np.log(df['duration'] + 1)
df['sqrt_dur'] = np.sqrt(df['duration'])

print("\nFormula correlations:")
for feat in ['hr_x_dur', 'hr_x_sqrt_dur', 'hrmax_x_sqrt_dur', 'log_dur', 'sqrt_dur']:
    r, p = pearsonr(df[feat], df['borg'])
    print(f"{feat:20s}  r = {r:+.3f}  (p = {p:.4f})")

# What's the theoretical maximum?
print("\n" + "="*70)
print("WHY CAN'T WE GET R² = 0.8?")
print("="*70)

print("""
1. CHECK: Is there enough signal in the data?
""")

# Best raw correlation
best_r = max([abs(pearsonr(df[f], df['borg'])[0]) for f in features + ['hr_x_dur', 'hr_x_sqrt_dur', 'hrmax_x_sqrt_dur']])
print(f"   Best raw correlation: r = {best_r:.3f} → max possible R² = {best_r**2:.3f}")

print("""
2. CHECK: Is Borg itself reliable?
   - Borg is subjective (person's perception)
   - Same activity rated differently on different days
   - Different people have different baselines
""")

# Look at same activities with different Borg
print("\n   Activities performed multiple times:")
activity_counts = df['activity'].value_counts()
repeated = activity_counts[activity_counts > 1]
if len(repeated) > 0:
    for act in repeated.index[:5]:
        act_df = df[df['activity'] == act]
        print(f"   {act}: Borg = {act_df['borg'].tolist()}, HR_max = {act_df['hr_max'].tolist()}")

print("""
3. CHECK: Person-specific calibration
""")

# Normalize features by person's range
df['borg_norm'] = (df['borg'] - df['borg'].min()) / (df['borg'].max() - df['borg'].min())
df['hr_norm'] = (df['hr_max'] - df['hr_rest']) / (df['hr_max_obs'] - df['hr_rest'])

r_norm, _ = pearsonr(df['hr_norm'] * np.sqrt(df['duration']), df['borg'])
print(f"   Normalized HR correlation: r = {r_norm:.3f}")

print("""
4. THE REAL ISSUE: Small sample size + LOO-CV
""")

# Compare train vs CV performance
X = (df['hr_max'] * np.sqrt(df['duration'])).values.reshape(-1, 1)
y = df['borg'].values

# Train R² (fit on all, predict on all)
model = LinearRegression()
model.fit(X, y)
train_r2 = r2_score(y, model.predict(X))

# CV R²
loo = LeaveOneOut()
y_pred_cv = cross_val_predict(model, X, y, cv=loo)
cv_r2 = r2_score(y, y_pred_cv)

print(f"   Train R² (overfit): {train_r2:.3f}")
print(f"   CV R² (honest):     {cv_r2:.3f}")
print(f"   Gap:                {train_r2 - cv_r2:.3f}")

print("\n" + "="*70)
print("HOW TO GET R² = 0.8")
print("="*70)

print("""
Option A: More data (most important!)
   - Current: 30 activities
   - Needed: 100+ activities across multiple subjects
   - With LOO-CV, variance is huge with small samples

Option B: Better features
   - Add accelerometer-based activity intensity
   - Add respiration rate
   - Add HRV features (but need continuous RR intervals)

Option C: Person-specific model
   - Calibrate to each person's HR range
   - Use relative effort, not absolute

Option D: Different evaluation
   - The original r=0.82 was likely NOT cross-validated
   - Or it was on much larger datasets
""")

# Try combining all subjects
print("\n" + "="*70)
print("EXPERIMENT: Combine all subjects")
print("="*70)

all_dfs = []
for subj in ['sim_elderly3', 'sim_healthy3']:
    subj_df = load_subject(subj)
    if subj_df is not None and len(subj_df) > 5:
        # Normalize Borg per subject
        subj_df['borg_norm'] = (subj_df['borg'] - subj_df['borg'].min()) / (subj_df['borg'].max() - subj_df['borg'].min() + 0.001)
        subj_df['hr_norm'] = (subj_df['hr_max'] - subj_df['hr_rest']) / (subj_df['hr_max_obs'] - subj_df['hr_rest'] + 0.001)
        all_dfs.append(subj_df)
        print(f"  {subj}: {len(subj_df)} activities, Borg {subj_df['borg'].min():.1f}-{subj_df['borg'].max():.1f}")

if len(all_dfs) > 1:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Combined: {len(combined)} activities")
    
    # Try with normalized features
    X = (combined['hr_norm'] * np.sqrt(combined['duration'])).values.reshape(-1, 1)
    y = combined['borg_norm'].values
    
    y_pred_cv = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    r2 = r2_score(y, y_pred_cv)
    r, _ = pearsonr(y, y_pred_cv)
    
    print(f"  Normalized formula across subjects:")
    print(f"    CV R² = {r2:.3f}")
    print(f"    CV r  = {r:.3f}")

print("\n" + "="*70)
print("REALISTIC EXPECTATIONS")
print("="*70)
print(f"""
With current data (30 activities, 1 subject):
  - Best achievable CV R²: ~0.35-0.40
  - This is actually reasonable for physiological data!

To get R² = 0.8:
  - Need 5-10x more data
  - Need multiple sessions per person
  - Need better Borg reliability (training subjects)
  - OR: Don't use CV (but then it's not honest)
""")
