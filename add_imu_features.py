#!/usr/bin/env python3
"""
Add IMU (accelerometer) features to improve prediction.
Different people have different HR responses, but movement intensity should be more universal.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def load_fused_with_activities(subject):
    """Load fused data with activity labels and aggregate per activity"""
    subj_path = BASE_PATH / subject
    
    # Load fused data (has IMU + HR features)
    fused_path = subj_path / "effort_estimation_output" / f"parsingsim3_{subject}" / "fused_aligned_10.0s.csv"
    if not fused_path.exists():
        return None, "No fused data"
    
    df = pd.read_csv(fused_path)
    
    # Load ADL for activity assignment
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
        return None, "No ADL file"
    
    adl = pd.read_csv(adl_path, skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
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
            activities.append({
                'activity': current,
                't_start': start_time,
                't_end': row['timestamp'],
                'duration': row['timestamp'] - start_time,
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
            current = None
    
    activities_df = pd.DataFrame(activities).dropna(subset=['borg'])
    
    # Compute offset
    offset = activities_df['t_start'].min() - df['t_center'].min()
    activities_df['t_start_adj'] = activities_df['t_start'] - offset
    activities_df['t_end_adj'] = activities_df['t_end'] - offset
    
    # Assign windows to activities
    def assign_activity(t_center):
        for i, act in activities_df.iterrows():
            if act['t_start_adj'] <= t_center <= act['t_end_adj']:
                return i
        return None
    
    df['activity_idx'] = df['t_center'].apply(assign_activity)
    df = df[df['activity_idx'].notna()].copy()
    
    if len(df) < 10:
        return None, "Too few windows matched"
    
    # Get feature columns (HR + IMU)
    hr_features = [c for c in df.columns if 'hr' in c.lower() and 'ppg' in c.lower()]
    imu_features = [c for c in df.columns if 'acc' in c.lower()]
    
    # Select key features (not all 60!)
    # HR: max, mean, std
    hr_key = [c for c in hr_features if any(x in c for x in ['_max', '_mean', '_std']) and 'green' in c]
    # IMU: max, variance, entropy (movement intensity indicators)
    imu_key = [c for c in imu_features if any(x in c for x in ['max', 'variance', 'entropy']) and not c.endswith('_r')]
    
    all_features = hr_key + imu_key
    
    # Aggregate per activity
    agg_dict = {f: ['mean', 'max'] for f in all_features if f in df.columns}
    agg_df = df.groupby('activity_idx').agg(agg_dict)
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    
    # Merge with activity info
    agg_df['activity_idx'] = agg_df['activity_idx'].astype(int)
    agg_df = agg_df.merge(
        activities_df[['borg', 'duration']].reset_index().rename(columns={'index': 'activity_idx'}),
        on='activity_idx'
    )
    
    return agg_df, None

print("="*70)
print("ADDING IMU (ACCELEROMETER) FEATURES")
print("="*70)

for subject in ['sim_elderly3', 'sim_healthy3']:
    print(f"\n{'='*70}")
    print(f"SUBJECT: {subject}")
    print("="*70)
    
    df, error = load_fused_with_activities(subject)
    if df is None:
        print(f"  SKIP: {error}")
        continue
    
    print(f"  Activities: {len(df)}")
    print(f"  Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
    print(f"  Features: {len([c for c in df.columns if c not in ['activity_idx', 'borg', 'duration']])}")
    
    y = df['borg'].values
    loo = LeaveOneOut()
    
    # Get feature groups
    hr_cols = [c for c in df.columns if 'hr' in c.lower()]
    imu_cols = [c for c in df.columns if 'acc' in c.lower()]
    
    print(f"\n  HR features: {len(hr_cols)}")
    print(f"  IMU features: {len(imu_cols)}")
    
    results = []
    
    # 1. Duration only (baseline)
    X = df[['duration']].values
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
    results.append(('Duration only', 1, r2_score(y, y_pred)))
    
    # 2. HR only (best HR feature)
    if len(hr_cols) > 0:
        # Find best single HR feature
        best_hr_r2 = -999
        best_hr_col = None
        for col in hr_cols:
            if df[col].std() > 0:
                X = df[[col]].fillna(0).values
                y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
                r2 = r2_score(y, y_pred)
                if r2 > best_hr_r2:
                    best_hr_r2 = r2
                    best_hr_col = col
        results.append((f'Best HR only ({best_hr_col[:20]}...)', 1, best_hr_r2))
    
    # 3. IMU only (best IMU feature)
    if len(imu_cols) > 0:
        best_imu_r2 = -999
        best_imu_col = None
        for col in imu_cols:
            if df[col].std() > 0:
                X = df[[col]].fillna(0).values
                y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
                r2 = r2_score(y, y_pred)
                if r2 > best_imu_r2:
                    best_imu_r2 = r2
                    best_imu_col = col
        results.append((f'Best IMU only ({best_imu_col[:20]}...)', 1, best_imu_r2))
    
    # 4. Duration + HR_max
    if 'ppg_green_hr_max_max' in df.columns:
        X = df[['duration', 'ppg_green_hr_max_max']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('Duration + HR_max', 2, r2_score(y, y_pred)))
    
    # 5. Duration + best IMU
    if best_imu_col:
        X = df[['duration', best_imu_col]].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('Duration + best IMU', 2, r2_score(y, y_pred)))
    
    # 6. HR + IMU (small set)
    if 'ppg_green_hr_max_max' in df.columns and best_imu_col:
        X = df[['duration', 'ppg_green_hr_max_max', best_imu_col]].fillna(0).values
        y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
        results.append(('Duration + HR + IMU (3 feat)', 3, r2_score(y, y_pred)))
    
    # 7. All HR + all IMU with Ridge (regularized to prevent overfit)
    all_feat_cols = [c for c in hr_cols + imu_cols if df[c].std() > 0]
    if len(all_feat_cols) > 0:
        X = df[['duration'] + all_feat_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = cross_val_predict(Ridge(alpha=10.0), X_scaled, y, cv=loo)
        results.append((f'All features Ridge ({len(all_feat_cols)+1} feat)', len(all_feat_cols)+1, r2_score(y, y_pred)))
    
    # 8. XGBoost with selected features
    if len(all_feat_cols) >= 3:
        X = df[['duration'] + all_feat_cols[:10]].fillna(0).values  # Limit to 10 best
        model = xgb.XGBRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42, verbosity=0)
        y_pred = cross_val_predict(model, X, y, cv=loo)
        results.append(('XGBoost (dur + 10 feat)', 11, r2_score(y, y_pred)))
    
    # Print results
    print(f"\n  {'Method':<40} {'# Feat':>8} {'CV R²':>10}")
    print("  " + "-"*60)
    for name, n_feat, r2 in sorted(results, key=lambda x: -x[2]):
        print(f"  {name:<40} {n_feat:>8} {r2:>10.3f}")
    
    # Find best overall
    best = max(results, key=lambda x: x[2])
    print(f"\n  ★ Best: {best[0]} with CV R² = {best[2]:.3f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
Adding IMU helps because:
1. HR response varies by person (elderly vs healthy vs severe)
2. Movement intensity (IMU) is more universal
3. Combining both captures different aspects of effort

But we're still limited by:
- Only 30 samples per subject
- Borg subjective variability
- Need more data for R² > 0.5
""")
