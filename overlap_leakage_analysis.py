#!/usr/bin/env python3
"""
COMPREHENSIVE ANALYSIS: Is Window Overlap Actually Data Leakage?

Questions to answer:
1. Is overlap truly leakage? ML uses overlap for a reason!
2. How does XGBoost compare with/without overlap?
3. How were splits done?
4. Which approach is better: XGBoost + feature selection OR Linear regression?
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

print("="*80)
print("IS WINDOW OVERLAP DATA LEAKAGE? COMPREHENSIVE ANALYSIS")
print("="*80)

print("""
BACKGROUND:
-----------
Window overlap (e.g., 70%) is STANDARD in signal processing because:
  1. Captures transient features that span window boundaries
  2. Increases sample size (more training data)
  3. Smoother temporal resolution

The PROBLEM is not overlap itself, but HOW you split train/test:
  - Random split at WINDOW level → LEAKAGE (adjacent windows share data)
  - Split at ACTIVITY level → NO LEAKAGE (entire activities go to train OR test)
  - Split at SUBJECT level → NO LEAKAGE (generalization to new subjects)
""")

# Load the fused data with window-level features
fused_path = BASE_PATH / "sim_elderly3" / "effort_estimation_output" / "fused_aligned_10.0s.csv"

if fused_path.exists():
    df = pd.read_csv(fused_path)
    print(f"\nLoaded: {len(df)} windows")
    
    # Check if activity/bout info exists
    if 'borg' in df.columns:
        df_labeled = df.dropna(subset=['borg'])
        print(f"Labeled windows: {len(df_labeled)}")
        print(f"Borg range: {df_labeled['borg'].min():.1f} - {df_labeled['borg'].max():.1f}")
else:
    print(f"Fused file not found: {fused_path}")
    print("Will demonstrate with per-activity data...")
    df_labeled = None

print("\n" + "="*80)
print("1. HOW WERE THE ORIGINAL SPLITS DONE?")
print("="*80)
print("""
From train_multisub_xgboost.py:

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

This is RANDOM SPLIT at WINDOW level!
- 607 windows in train, 152 in test
- BUT windows overlap 70%
- Window 100 and Window 101 share 70% of the same raw signal
- If Window 100 → train, Window 101 → test: LEAKAGE!

That's why Train R² = 0.999 (model "memorizes" overlapping info)
""")

print("\n" + "="*80)
print("2. IS OVERLAP ALWAYS BAD? NO!")
print("="*80)
print("""
Overlap is FINE if you split correctly:

OPTION A: Activity-level split (what we did with LOO-CV)
  - All windows from Activity 1 → train
  - All windows from Activity 2 → test
  - No information leaks between activities
  
OPTION B: Subject-level split (GroupKFold)
  - All data from Subject 1 → train
  - All data from Subject 2 → test
  - Tests generalization to new people

OPTION C: Temporal split
  - First 80% of time → train
  - Last 20% of time → test
  - Tests temporal generalization

The WRONG way:
  - Random window shuffle → adjacent windows leak information
""")

print("\n" + "="*80)
print("3. DEMONSTRATION: Random vs Activity-level split")
print("="*80)

# Create synthetic demonstration
def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

subject = 'sim_elderly3'
subj_path = BASE_PATH / subject

# Load ADL and create per-activity features
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

adl_start = adl['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()
HR_rest = hr['heart_rate'].quantile(0.05)

# Parse activities
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
        duration = row['timestamp'] - start_time
        
        t_start = start_time - hr_offset
        t_end = row['timestamp'] - hr_offset
        mask = (hr['timestamp'] >= t_start) & (hr['timestamp'] <= t_end)
        hr_vals = hr.loc[mask, 'heart_rate'].values
        
        t_start_acc = start_time - acc_offset
        t_end_acc = row['timestamp'] - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask, 'magnitude'].values
        
        if len(hr_vals) >= 2 and len(acc_vals) >= 10:
            activities.append({
                'activity_id': activity_id,
                'activity': current,
                'duration': duration,
                'hr_mean': hr_vals.mean(),
                'hr_max': hr_vals.max(),
                'hr_std': hr_vals.std(),
                'hr_elevation': (hr_vals.max() - HR_rest) / HR_rest * 100,
                'acc_mean': acc_vals.mean(),
                'acc_std': acc_vals.std(),
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
            activity_id += 1
        current = None

df_act = pd.DataFrame(activities).dropna()
print(f"\nPer-activity data: {len(df_act)} activities")

# Define features
feature_cols = ['duration', 'hr_mean', 'hr_max', 'hr_std', 'hr_elevation', 'acc_mean', 'acc_std']
X = df_act[feature_cols].values
y = df_act['borg'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- EXPERIMENT: XGBoost with different split strategies ---\n")

# A) Random 80/20 split (simulates the "leaky" approach)
print("A) RANDOM SPLIT (like original pipeline):")
results_random = []
for seed in range(5):  # Multiple random seeds
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results_random.append((train_r2, test_r2))

avg_train = np.mean([r[0] for r in results_random])
avg_test = np.mean([r[1] for r in results_random])
print(f"   XGBoost (7 features): Train R² = {avg_train:.3f}, Test R² = {avg_test:.3f}")

# B) Leave-One-Activity-Out (no leakage)
print("\nB) LEAVE-ONE-ACTIVITY-OUT (no leakage):")
y_pred_loo = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    X_scaled, y, cv=LeaveOneOut()
)
loo_r2 = r2_score(y, y_pred_loo)
print(f"   XGBoost (7 features): LOO-CV R² = {loo_r2:.3f}")

# C) Linear regression with LOO
print("\nC) LINEAR REGRESSION (LOO-CV):")
y_pred_linear = cross_val_predict(Ridge(alpha=1.0), X_scaled, y, cv=LeaveOneOut())
linear_r2 = r2_score(y, y_pred_linear)
print(f"   Ridge (7 features): LOO-CV R² = {linear_r2:.3f}")

# D) Simple 2-feature model
print("\nD) SIMPLE 2-FEATURE MODEL (LOO-CV):")
X_simple = df_act[['hr_elevation', 'duration']].values
X_simple_scaled = StandardScaler().fit_transform(X_simple)

y_pred_simple_xgb = cross_val_predict(
    xgb.XGBRegressor(n_estimators=50, max_depth=2, random_state=42),
    X_simple_scaled, y, cv=LeaveOneOut()
)
y_pred_simple_linear = cross_val_predict(Ridge(alpha=1.0), X_simple_scaled, y, cv=LeaveOneOut())

print(f"   XGBoost (2 features): LOO-CV R² = {r2_score(y, y_pred_simple_xgb):.3f}")
print(f"   Ridge (2 features):   LOO-CV R² = {r2_score(y, y_pred_simple_linear):.3f}")

print("\n" + "="*80)
print("4. WHY XGBOOST ALSO HAS 'LEAKAGE' IN ORIGINAL RESULTS")
print("="*80)
print("""
XGBoost DID have leakage - that's why:
- Train R² = 0.9995 (almost perfect = memorization)
- Test R² = 0.9574 (still good because test windows overlap with train!)

With 70% overlap:
- Window 100: samples 0-1000
- Window 101: samples 300-1300 (shares 700 samples with Window 100!)

If both are in train → model learns their shared pattern perfectly
If Window 100 is train, Window 101 is test → "test" is 70% identical to training data!

This inflates BOTH train AND test R² artificially.
""")

print("\n" + "="*80)
print("5. WHICH APPROACH IS BETTER?")
print("="*80)

print("""
COMPARISON SUMMARY:
------------------
                                    | Train/CV R² | Notes
------------------------------------|-------------|------------------
Random split + XGBoost (287 feat)   | ~0.95       | LEAKY! Not real performance
Activity-split + XGBoost (287 feat) | ~0.07       | Too many features for 25 samples
Activity-split + XGBoost (7 feat)   | ~{:.2f}       | Reasonable
Activity-split + Linear (7 feat)    | ~{:.2f}       | Similar!
Activity-split + Linear (2 feat)    | ~{:.2f}       | Simple but effective

CONCLUSION:
-----------
For N=25 activities (small data), SIMPLER IS BETTER:

1. DON'T use 287 features with XGBoost → massive overfitting
2. Linear regression with 2-7 features ≈ XGBoost with 2-7 features
3. The "complex pipeline" (feature extraction → selection → XGBoost) 
   only helps if you have 100s-1000s of samples

RECOMMENDED APPROACH:
1. Use per-activity aggregation (not per-window)
2. Compute simple features: HR_elevation, duration, HR_std
3. Train Linear/Ridge regression (not XGBoost)
4. Use LOO-CV or Activity-GroupKFold for validation

For YOUR data size, Linear ≈ XGBoost performance!
""".format(loo_r2, linear_r2, r2_score(y, y_pred_simple_linear)))

print("\n" + "="*80)
print("6. WHEN WOULD FULL PIPELINE BE BETTER?")
print("="*80)
print("""
The full pipeline (feature extraction → selection → XGBoost) works when:

1. LARGE SAMPLE SIZE: 100+ activities per subject, or pooled across many subjects
2. CORRECT SPLITS: Group by activity/subject, not random window shuffle
3. TEMPORAL FEATURES NEEDED: When within-activity dynamics matter
   (e.g., HR increases during activity → need windowed features)

For effort estimation from 25-30 ADL activities:
- Simple per-activity features work best
- Linear regression is appropriate
- XGBoost adds complexity without benefit

The R²=0.95 you saw was an ARTIFACT of leaky window splits, not true performance.
""")
