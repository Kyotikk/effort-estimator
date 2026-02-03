#!/usr/bin/env python3
"""
LEAVE-ONE-SUBJECT-OUT Cross-Validation
Train on 2 subjects, test on 1 - rotate through all 3
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def load_subject_data(subject):
    """Load and compute per-activity features for a subject."""
    subj_path = BASE_PATH / subject
    
    # Load ADL - try different filenames
    adl_path = None
    for pattern in ["ADLs_1.csv", "ADLs_1-2.csv", "ADLs_1-3.csv", "ADL_1.csv"]:
        candidate = subj_path / "scai_app" / pattern
        if candidate.exists():
            adl_path = candidate
            break
    
    if adl_path is None:
        print(f"  ⚠️ No ADL file for {subject}")
        return None
    
    adl = pd.read_csv(adl_path, skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
    # Load HR
    hr_path = subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz"
    if not hr_path.exists():
        print(f"  ⚠️ No HR file for {subject}")
        return None
    
    hr = pd.read_csv(hr_path)
    hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
    hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]
    
    # Load wrist ACC
    acc_path = subj_path / "corsano_wrist_acc"
    if not acc_path.exists():
        print(f"  ⚠️ No ACC folder for {subject}")
        return None
    
    acc_files = list(acc_path.glob("*.csv.gz"))
    if not acc_files:
        print(f"  ⚠️ No ACC files for {subject}")
        return None
    
    acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
    acc = acc.rename(columns={'time': 'timestamp'})
    acc['magnitude'] = np.sqrt(acc['accX']**2 + acc['accY']**2 + acc['accZ']**2)
    
    # Compute offsets
    adl_start = adl['timestamp'].min()
    hr_offset = adl_start - hr['timestamp'].min()
    acc_offset = adl_start - acc['timestamp'].min()
    
    # Baselines
    HR_rest = hr['heart_rate'].quantile(0.05)
    ACC_rest = acc['magnitude'].quantile(0.10)
    
    # Parse activities
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
                    'subject': subject,
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
            current = None
    
    df = pd.DataFrame(activities).dropna()
    return df

print("="*70)
print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
print("="*70)

# Load all subjects
subjects = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
all_data = []

for subj in subjects:
    print(f"\nLoading {subj}...")
    df = load_subject_data(subj)
    if df is not None and len(df) > 0:
        print(f"  ✓ {len(df)} activities, Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
        all_data.append(df)
    else:
        print(f"  ✗ Failed to load")

# Combine
df_all = pd.concat(all_data, ignore_index=True)
print(f"\n{'='*70}")
print(f"COMBINED: {len(df_all)} activities from {df_all['subject'].nunique()} subjects")
print(f"{'='*70}")

# Feature columns
feature_cols = ['duration', 'hr_elevation', 'hr_max', 'hr_std', 'acc_mean', 'acc_std']

# Leave-One-Subject-Out CV
print("\n" + "="*70)
print("LEAVE-ONE-SUBJECT-OUT RESULTS")
print("="*70)

results = []

for test_subj in df_all['subject'].unique():
    train_mask = df_all['subject'] != test_subj
    test_mask = df_all['subject'] == test_subj
    
    X_train = df_all.loc[train_mask, feature_cols].values
    y_train = df_all.loc[train_mask, 'borg'].values
    X_test = df_all.loc[test_mask, feature_cols].values
    y_test = df_all.loc[test_mask, 'borg'].values
    
    # Skip if test set has no variance
    if y_test.std() < 0.5:
        print(f"\n⚠️ {test_subj}: Borg std={y_test.std():.2f} too low, skipping")
        continue
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n{'='*50}")
    print(f"TEST: {test_subj} ({len(X_test)} activities)")
    print(f"TRAIN: {df_all.loc[train_mask, 'subject'].unique().tolist()} ({len(X_train)} activities)")
    print(f"{'='*50}")
    
    # 1. Ridge (simple)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    
    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    
    # 3. Simple 2-feature model
    X_train_simple = df_all.loc[train_mask, ['hr_elevation', 'duration']].values
    X_test_simple = df_all.loc[test_mask, ['hr_elevation', 'duration']].values
    scaler_simple = StandardScaler()
    X_train_simple_s = scaler_simple.fit_transform(X_train_simple)
    X_test_simple_s = scaler_simple.transform(X_test_simple)
    
    ridge_simple = Ridge(alpha=1.0)
    ridge_simple.fit(X_train_simple_s, y_train)
    y_pred_simple = ridge_simple.predict(X_test_simple_s)
    r2_simple = r2_score(y_test, y_pred_simple)
    mae_simple = mean_absolute_error(y_test, y_pred_simple)
    
    print(f"\n  Ridge (6 feat):    R² = {r2_ridge:>6.3f}  MAE = {mae_ridge:.2f}")
    print(f"  XGBoost (6 feat):  R² = {r2_xgb:>6.3f}  MAE = {mae_xgb:.2f}")
    print(f"  Ridge (2 feat):    R² = {r2_simple:>6.3f}  MAE = {mae_simple:.2f}")
    
    # Sample predictions
    print(f"\n  Sample predictions (Ridge 2-feat):")
    for i in range(min(5, len(y_test))):
        act = df_all.loc[test_mask, 'activity'].iloc[i]
        print(f"    {act[:20]:<20} Actual={y_test[i]:.1f}  Pred={y_pred_simple[i]:.1f}")
    
    results.append({
        'test_subject': test_subj,
        'n_test': len(y_test),
        'r2_ridge': r2_ridge,
        'r2_xgb': r2_xgb,
        'r2_simple': r2_simple,
        'mae_simple': mae_simple
    })

# Summary
print("\n" + "="*70)
print("SUMMARY: Leave-One-Subject-Out CV")
print("="*70)

if results:
    df_results = pd.DataFrame(results)
    print(f"\n{'Test Subject':<15} {'N':>4} {'Ridge(6)':>10} {'XGB(6)':>10} {'Ridge(2)':>10}")
    print("-"*55)
    for _, row in df_results.iterrows():
        print(f"{row['test_subject']:<15} {row['n_test']:>4} {row['r2_ridge']:>10.3f} {row['r2_xgb']:>10.3f} {row['r2_simple']:>10.3f}")
    
    print("-"*55)
    print(f"{'MEAN':<15} {'':>4} {df_results['r2_ridge'].mean():>10.3f} {df_results['r2_xgb'].mean():>10.3f} {df_results['r2_simple'].mean():>10.3f}")
    
    print(f"""
INTERPRETATION:
- Positive R² = model generalizes to new subjects
- Negative R² = model fails to generalize
- R² ~ 0.3-0.5 = decent for cross-subject
- R² > 0.5 = good generalization
""")
