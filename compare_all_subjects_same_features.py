#!/usr/bin/env python3
"""
Compare Linear vs XGBoost with same features - ALL SUBJECTS
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")
SUBJECTS = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def load_and_process_subject(subject):
    """Load HR and ADL data, compute per-activity features"""
    subj_path = BASE_PATH / subject
    
    # Find ADL file with Borg ratings
    adl_path = None
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
    
    if adl_path is None:
        return None, "No ADL file with Borg"
    
    # Load ADL
    adl = pd.read_csv(adl_path, skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
    # Load HR
    hr_path = subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz"
    if not hr_path.exists():
        return None, "No HR file"
    
    hr = pd.read_csv(hr_path)
    hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
    hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]  # Filter valid HR
    
    if len(hr) < 10:
        return None, f"Too few HR samples ({len(hr)})"
    
    # Compute offset
    hr_start = hr['timestamp'].min()
    adl_start = adl['timestamp'].min()
    offset = adl_start - hr_start
    
    # Parse activities with HR
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
                    'activity': current,
                    'duration': duration,
                    'hr_mean': hr_vals.mean(),
                    'hr_max': hr_vals.max(),
                    'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
                })
            current = None
    
    df = pd.DataFrame(activities).dropna()
    
    if len(df) < 5:
        return None, f"Too few activities ({len(df)})"
    
    return df, None

def evaluate_models(df, subject):
    """Evaluate different models with LOO-CV"""
    y = df['borg'].values
    loo = LeaveOneOut()
    
    results = []
    
    # 1. Linear: HR_max × √duration (best formula)
    X = (df['hr_max'] * np.sqrt(df['duration'])).values.reshape(-1, 1)
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
    results.append(('Linear (HR_max × √dur)', 1, r2_score(y, y_pred)))
    
    # 2. Linear: HR_mean × √duration
    X = (df['hr_mean'] * np.sqrt(df['duration'])).values.reshape(-1, 1)
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
    results.append(('Linear (HR_mean × √dur)', 1, r2_score(y, y_pred)))
    
    # 3. Linear: HR_max + duration
    X = df[['hr_max', 'duration']].values
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
    results.append(('Linear (HR_max + dur)', 2, r2_score(y, y_pred)))
    
    # 4. XGBoost: HR_max × √duration
    X = (df['hr_max'] * np.sqrt(df['duration'])).values.reshape(-1, 1)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    y_pred = cross_val_predict(model, X, y, cv=loo)
    results.append(('XGBoost (HR_max × √dur)', 1, r2_score(y, y_pred)))
    
    # 5. XGBoost: HR_max + duration
    X = df[['hr_max', 'duration']].values
    y_pred = cross_val_predict(model, X, y, cv=loo)
    results.append(('XGBoost (HR_max + dur)', 2, r2_score(y, y_pred)))
    
    return results

print("="*80)
print("COMPARISON ACROSS ALL SUBJECTS: Linear vs XGBoost (same features)")
print("="*80)

all_results = []

for subject in SUBJECTS:
    print(f"\n{'='*80}")
    print(f"SUBJECT: {subject}")
    print("="*80)
    
    df, error = load_and_process_subject(subject)
    
    if df is None:
        print(f"  SKIP: {error}")
        continue
    
    print(f"  Activities: {len(df)}")
    print(f"  Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
    print(f"  HR max range: {df['hr_max'].min():.0f} - {df['hr_max'].max():.0f} bpm")
    
    results = evaluate_models(df, subject)
    
    print(f"\n  {'Method':<30} {'CV R²':>10}")
    print("  " + "-"*45)
    for name, n_feat, r2 in sorted(results, key=lambda x: -x[2]):
        print(f"  {name:<30} {r2:>10.3f}")
        all_results.append({
            'Subject': subject,
            'Method': name,
            'n_features': n_feat,
            'CV_R2': r2,
            'n_activities': len(df),
            'borg_range': df['borg'].max() - df['borg'].min()
        })

# Summary
print("\n" + "="*80)
print("SUMMARY: Best method per subject")
print("="*80)

results_df = pd.DataFrame(all_results)

for subject in SUBJECTS:
    subj_data = results_df[results_df['Subject'] == subject]
    if len(subj_data) == 0:
        continue
    
    best = subj_data.sort_values('CV_R2', ascending=False).iloc[0]
    linear_best = subj_data[subj_data['Method'].str.startswith('Linear')].sort_values('CV_R2', ascending=False).iloc[0]
    xgb_best = subj_data[subj_data['Method'].str.startswith('XGBoost')].sort_values('CV_R2', ascending=False).iloc[0]
    
    print(f"\n{subject} (n={best['n_activities']}, Borg range={best['borg_range']:.1f}):")
    print(f"  Best Linear:  {linear_best['Method']:<25} CV R² = {linear_best['CV_R2']:.3f}")
    print(f"  Best XGBoost: {xgb_best['Method']:<25} CV R² = {xgb_best['CV_R2']:.3f}")
    
    diff = linear_best['CV_R2'] - xgb_best['CV_R2']
    if abs(diff) < 0.05:
        print(f"  → Similar performance (diff = {diff:.3f})")
    elif diff > 0:
        print(f"  → Linear wins by {diff:.3f}")
    else:
        print(f"  → XGBoost wins by {-diff:.3f}")

# Cross-subject average
print("\n" + "="*80)
print("AVERAGE ACROSS SUBJECTS")
print("="*80)

for method in results_df['Method'].unique():
    method_data = results_df[results_df['Method'] == method]
    if len(method_data) > 1:
        avg_r2 = method_data['CV_R2'].mean()
        print(f"{method:<30} Avg CV R² = {avg_r2:.3f} (n={len(method_data)} subjects)")
