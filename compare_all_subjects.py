#!/usr/bin/env python3
"""
Compare linear formula vs XGBoost across ALL subjects
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
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")
SUBJECTS = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']

def parse_time(t):
    try:
        dt = datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None

def load_subject_data(subject):
    """Load ADL and HR data for a subject"""
    subj_path = BASE_PATH / subject
    
    # Load ADL - try different file patterns
    # elderly3 has ADLs_1.csv, others have ADLs_1-3.csv with Borg
    adl_path = None
    for fname in ['ADLs_1.csv', 'ADLs_1-3.csv', 'ADLs_1-2.csv', 'ADLs_1.csv.gz']:
        test_path = subj_path / "scai_app" / fname
        if test_path.exists():
            # Check if it has the Effort column
            try:
                test_df = pd.read_csv(test_path, skiprows=2, nrows=1)
                if len(test_df.columns) >= 3:  # Has Time, ADLs, Effort
                    adl_path = test_path
                    break
            except:
                continue
    
    if adl_path is None:
        print(f"  No ADL file with Borg ratings found for {subject}")
        return None
    
    print(f"  Using ADL file: {adl_path.name}")
    
    adl = pd.read_csv(adl_path, skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
    # Load HR
    hr_path = subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz"
    if not hr_path.exists():
        return None
    
    hr = pd.read_csv(hr_path)
    hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
    
    # Filter valid HR
    hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]
    
    # Compute offset
    hr_start = hr['timestamp'].min()
    adl_start = adl['timestamp'].min()
    offset = adl_start - hr_start
    
    # Get resting HR
    hr_rest = hr['heart_rate'].quantile(0.05)
    hr_max_obs = hr['heart_rate'].max()
    
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
                    'hr_reserve': hr_vals.mean() - hr_rest,
                    'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
                })
            current = None
    
    return pd.DataFrame(activities).dropna()

def evaluate_formulas(df, subject_name):
    """Evaluate different formulas with LOO-CV"""
    if len(df) < 5:
        return None
    
    y = df['borg'].values
    loo = LeaveOneOut()
    
    results = {}
    
    # Best formulas from optimization
    formulas = {
        'HR_max × √dur': df['hr_max'] * np.sqrt(df['duration']),
        'Duration only': df['duration'],
        'HR_mean × dur': df['hr_mean'] * df['duration'],
        'HR_mean × √dur': df['hr_mean'] * np.sqrt(df['duration']),
    }
    
    for name, formula in formulas.items():
        X = formula.values.reshape(-1, 1)
        model = LinearRegression()
        y_pred = cross_val_predict(model, X, y, cv=loo)
        results[name] = {
            'CV R²': r2_score(y, y_pred),
            'CV MAE': mean_absolute_error(y, y_pred)
        }
    
    # Multi-feature: HR_mean + duration
    X = df[['hr_mean', 'duration']].values
    model = Ridge(alpha=1.0)
    y_pred = cross_val_predict(model, X, y, cv=loo)
    results['HR_mean + dur'] = {
        'CV R²': r2_score(y, y_pred),
        'CV MAE': mean_absolute_error(y, y_pred)
    }
    
    return results

def evaluate_xgboost_per_activity(subject):
    """Run XGBoost with per-activity aggregation for a subject"""
    subj_path = BASE_PATH / subject / "effort_estimation_output" / f"parsingsim3_{subject}"
    fused_path = subj_path / "fused_aligned_10.0s.csv"
    
    if not fused_path.exists():
        return None
    
    # Load fused data
    df = pd.read_csv(fused_path)
    
    # Load ADL for activity assignment - use same logic as load_subject_data
    adl_path = None
    for fname in ['ADLs_1.csv', 'ADLs_1-3.csv', 'ADLs_1-2.csv']:
        test_path = BASE_PATH / subject / "scai_app" / fname
        if test_path.exists():
            try:
                test_df = pd.read_csv(test_path, skiprows=2, nrows=1)
                if len(test_df.columns) >= 3:
                    adl_path = test_path
                    break
            except:
                continue
    
    if adl_path is None:
        return None
    
    adl = pd.read_csv(adl_path, skiprows=2)
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
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
            current = None
    
    activities_df = pd.DataFrame(activities).dropna(subset=['borg'])
    
    # Compute offset
    fused_start = df['t_center'].min()
    adl_start = activities_df['t_start'].min()
    offset = adl_start - fused_start
    
    activities_df['t_start_adj'] = activities_df['t_start'] - offset
    activities_df['t_end_adj'] = activities_df['t_end'] - offset
    
    # Assign activities to windows
    def assign_activity(t_center):
        for i, act in activities_df.iterrows():
            if act['t_start_adj'] <= t_center <= act['t_end_adj']:
                return i
        return None
    
    df['activity_idx'] = df['t_center'].apply(assign_activity)
    df = df[df['activity_idx'].notna()].copy()
    
    if len(df) < 10:
        return None
    
    # Get feature columns
    meta_cols = ['t_center', 'borg', 'borg_cr10', 'activity', 'activity_idx', 'time_diff',
                 'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 'win_sec_r', 
                 't_start', 't_end', 'window_id', 'modality', 'start_idx', 'end_idx']
    feature_cols = [c for c in df.columns if c not in meta_cols 
                    and not c.startswith('Unnamed') 
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Aggregate per activity
    agg_dict = {col: 'mean' for col in feature_cols}
    activity_features = df.groupby('activity_idx').agg(agg_dict).reset_index()
    activity_features['activity_idx'] = activity_features['activity_idx'].astype(int)
    activity_df = activity_features.merge(
        activities_df[['borg']].reset_index().rename(columns={'index': 'activity_idx'}),
        on='activity_idx'
    )
    
    X = activity_df[feature_cols].fillna(0)
    y = activity_df['borg'].astype(float)
    
    # Remove constant columns
    X = X.loc[:, X.std() > 1e-6]
    
    if len(y) < 5:
        return None
    
    # LOO-CV with XGBoost
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.5, random_state=42, n_jobs=-1
    )
    
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
    
    return {
        'CV R²': r2_score(y, y_pred),
        'CV MAE': mean_absolute_error(y, y_pred),
        'n_activities': len(y),
        'n_features': X.shape[1]
    }


print("="*80)
print("COMPARISON ACROSS ALL SUBJECTS")
print("="*80)

all_results = []

for subject in SUBJECTS:
    print(f"\n{'='*80}")
    print(f"SUBJECT: {subject}")
    print("="*80)
    
    # Load data
    df = load_subject_data(subject)
    if df is None:
        print(f"  Could not load data for {subject}")
        continue
    
    print(f"  Activities: {len(df)}")
    print(f"  Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
    
    # Evaluate linear formulas
    formula_results = evaluate_formulas(df, subject)
    if formula_results:
        print(f"\n  LINEAR FORMULAS (LOO-CV):")
        for name, metrics in sorted(formula_results.items(), key=lambda x: -x[1]['CV R²']):
            print(f"    {name:20s}  R²={metrics['CV R²']:.3f}  MAE={metrics['CV MAE']:.2f}")
            all_results.append({
                'Subject': subject,
                'Method': f'Linear: {name}',
                'CV R²': metrics['CV R²'],
                'CV MAE': metrics['CV MAE']
            })
    
    # Evaluate XGBoost
    xgb_results = evaluate_xgboost_per_activity(subject)
    if xgb_results:
        print(f"\n  XGBOOST ({xgb_results['n_features']} features, {xgb_results['n_activities']} activities):")
        print(f"    XGBoost per-activity   R²={xgb_results['CV R²']:.3f}  MAE={xgb_results['CV MAE']:.2f}")
        all_results.append({
            'Subject': subject,
            'Method': 'XGBoost (287 features)',
            'CV R²': xgb_results['CV R²'],
            'CV MAE': xgb_results['CV MAE']
        })

# Summary table
print("\n" + "="*80)
print("SUMMARY: Best Linear Formula vs XGBoost per Subject")
print("="*80)

results_df = pd.DataFrame(all_results)
for subject in SUBJECTS:
    subj_data = results_df[results_df['Subject'] == subject]
    if len(subj_data) == 0:
        print(f"\n{subject}: No data available")
        continue
    
    linear_data = subj_data[subj_data['Method'].str.startswith('Linear')]
    xgb_data = subj_data[subj_data['Method'].str.startswith('XGBoost')]
    
    print(f"\n{subject}:")
    
    if len(linear_data) > 0:
        best_linear = linear_data.sort_values('CV R²', ascending=False).iloc[0]
        print(f"  Best Linear: {best_linear['Method'].replace('Linear: ', ''):20s} CV R² = {best_linear['CV R²']:.3f}")
    else:
        best_linear = None
        print(f"  Best Linear: No data")
    
    if len(xgb_data) > 0:
        xgb_row = xgb_data.iloc[0]
        print(f"  XGBoost:     {'(287 features)':20s} CV R² = {xgb_row['CV R²']:.3f}")
    else:
        xgb_row = None
        print(f"  XGBoost:     No fused data")
    
    if best_linear is not None and xgb_row is not None:
        improvement = best_linear['CV R²'] - xgb_row['CV R²']
        print(f"  → Linear is {'BETTER' if improvement > 0 else 'WORSE'} by {abs(improvement):.3f}")
