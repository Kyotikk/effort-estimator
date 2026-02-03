#!/usr/bin/env python3
"""
Use WRIST sensors (Corsano) for effort estimation:
- corsano_wrist_acc: Accelerometer from wrist
- corsano_bioz_emography: EDA/skin conductance
- corsano_wrist_ppg: PPG-derived HR from wrist
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
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def load_wrist_data(subject):
    """Load wrist sensor data: ACC, EDA, PPG"""
    subj_path = BASE_PATH / subject
    data = {}
    
    # 1. Wrist accelerometer
    acc_path = subj_path / "corsano_wrist_acc"
    acc_files = list(acc_path.glob("*.csv.gz")) + list(acc_path.glob("*.csv"))
    if acc_files:
        acc_dfs = []
        for f in acc_files:
            try:
                df = pd.read_csv(f)
                acc_dfs.append(df)
            except:
                pass
        if acc_dfs:
            acc = pd.concat(acc_dfs, ignore_index=True)
            acc = acc.rename(columns={'time': 'timestamp'})
            data['acc'] = acc
            print(f"  Wrist ACC: {len(acc)} samples")
    
    # 2. EDA (emography)
    eda_path = subj_path / "corsano_bioz_emography"
    eda_files = list(eda_path.glob("*.csv.gz")) + list(eda_path.glob("*.csv"))
    if eda_files:
        eda_dfs = []
        for f in eda_files:
            try:
                df = pd.read_csv(f)
                eda_dfs.append(df)
            except:
                pass
        if eda_dfs:
            eda = pd.concat(eda_dfs, ignore_index=True)
            eda = eda.rename(columns={'time': 'timestamp'})
            data['eda'] = eda
            print(f"  EDA: {len(eda)} samples")
    
    # 3. Wrist PPG (for HR)
    ppg_path = subj_path / "corsano_wrist_ppg2_green_6"
    ppg_files = list(ppg_path.glob("*.csv.gz")) + list(ppg_path.glob("*.csv"))
    if ppg_files:
        ppg_dfs = []
        for f in ppg_files:
            try:
                df = pd.read_csv(f)
                ppg_dfs.append(df)
            except:
                pass
        if ppg_dfs:
            ppg = pd.concat(ppg_dfs, ignore_index=True)
            ppg = ppg.rename(columns={'time': 'timestamp'})
            data['ppg'] = ppg
            print(f"  Wrist PPG: {len(ppg)} samples")
    
    # 4. Also load chest HR for comparison
    hr_path = subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz"
    if hr_path.exists():
        hr = pd.read_csv(hr_path)
        hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
        hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]
        data['hr'] = hr
        print(f"  Chest HR: {len(hr)} samples")
    
    return data

def load_adl(subject):
    """Load ADL with Borg ratings"""
    subj_path = BASE_PATH / subject
    
    for fname in ['ADLs_1.csv', 'ADLs_1-3.csv', 'ADLs_1-2.csv']:
        test_path = subj_path / "scai_app" / fname
        if test_path.exists():
            try:
                test_df = pd.read_csv(test_path, skiprows=2, nrows=1)
                if len(test_df.columns) >= 3:
                    adl = pd.read_csv(test_path, skiprows=2)
                    adl.columns = ['Time', 'ADLs', 'Effort']
                    adl['timestamp'] = adl['Time'].apply(parse_time)
                    return adl
            except:
                continue
    return None

def compute_activity_features(data, activities):
    """Compute features per activity from wrist sensors"""
    
    # Get time offset (ADL vs sensor timestamps)
    adl_start = activities['t_start'].min()
    
    offsets = {}
    for key in ['acc', 'eda', 'hr']:
        if key in data:
            sensor_start = data[key]['timestamp'].min()
            offsets[key] = adl_start - sensor_start
    
    results = []
    
    for _, act in activities.iterrows():
        row = {
            'activity': act['activity'],
            'duration': act['duration'],
            'borg': act['borg']
        }
        
        # ACC features
        if 'acc' in data:
            offset = offsets['acc']
            t_start = act['t_start'] - offset
            t_end = act['t_end'] - offset
            mask = (data['acc']['timestamp'] >= t_start) & (data['acc']['timestamp'] <= t_end)
            acc_vals = data['acc'].loc[mask]
            
            if len(acc_vals) > 0:
                # Compute magnitude
                if 'accX' in acc_vals.columns:
                    mag = np.sqrt(acc_vals['accX']**2 + acc_vals['accY']**2 + acc_vals['accZ']**2)
                    row['acc_mag_mean'] = mag.mean()
                    row['acc_mag_max'] = mag.max()
                    row['acc_mag_std'] = mag.std()
                    # Movement intensity (variance)
                    row['acc_intensity'] = mag.var()
        
        # EDA features
        if 'eda' in data:
            offset = offsets['eda']
            t_start = act['t_start'] - offset
            t_end = act['t_end'] - offset
            mask = (data['eda']['timestamp'] >= t_start) & (data['eda']['timestamp'] <= t_end)
            eda_vals = data['eda'].loc[mask]
            
            if len(eda_vals) > 0:
                if 'cc' in eda_vals.columns:  # Skin conductance
                    row['eda_mean'] = eda_vals['cc'].mean()
                    row['eda_max'] = eda_vals['cc'].max()
                if 'stress_skin' in eda_vals.columns:
                    row['stress_skin_mean'] = eda_vals['stress_skin'].mean()
                    row['stress_skin_max'] = eda_vals['stress_skin'].max()
        
        # HR features (from chest)
        if 'hr' in data:
            offset = offsets['hr']
            t_start = act['t_start'] - offset
            t_end = act['t_end'] - offset
            mask = (data['hr']['timestamp'] >= t_start) & (data['hr']['timestamp'] <= t_end)
            hr_vals = data['hr'].loc[mask, 'heart_rate']
            
            if len(hr_vals) > 0:
                row['hr_mean'] = hr_vals.mean()
                row['hr_max'] = hr_vals.max()
                row['hr_std'] = hr_vals.std() if len(hr_vals) > 1 else 0
        
        results.append(row)
    
    return pd.DataFrame(results)

print("="*70)
print("EFFORT ESTIMATION WITH WRIST SENSORS (ACC + EDA)")
print("="*70)

for subject in ['sim_elderly3', 'sim_healthy3', 'sim_severe3']:
    print(f"\n{'='*70}")
    print(f"SUBJECT: {subject}")
    print("="*70)
    
    # Load sensor data
    data = load_wrist_data(subject)
    if not data:
        print("  SKIP: No sensor data")
        continue
    
    # Load ADL
    adl = load_adl(subject)
    if adl is None:
        print("  SKIP: No ADL file")
        continue
    
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
    print(f"  Activities: {len(activities_df)}")
    
    if len(activities_df) < 5:
        print("  SKIP: Too few activities")
        continue
    
    # Compute features
    df = compute_activity_features(data, activities_df)
    df = df.dropna(subset=['borg'])
    
    print(f"  Features computed: {len(df)} activities")
    print(f"  Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
    
    # Check available features
    feature_cols = [c for c in df.columns if c not in ['activity', 'borg']]
    print(f"  Available features: {feature_cols}")
    
    if len(df) < 5:
        print("  SKIP: Too few matched activities")
        continue
    
    y = df['borg'].values
    loo = LeaveOneOut()
    
    # Test different feature combinations
    results = []
    
    # 1. Duration only
    X = df[['duration']].fillna(0).values
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
    results.append(('Duration only', 1, r2_score(y, y_pred)))
    
    # 2. HR only
    if 'hr_max' in df.columns and df['hr_max'].notna().sum() > 5:
        X = df[['hr_max']].fillna(df['hr_max'].mean()).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('HR_max only', 1, r2_score(y, y_pred)))
    
    # 3. ACC only
    if 'acc_mag_mean' in df.columns and df['acc_mag_mean'].notna().sum() > 5:
        X = df[['acc_mag_mean']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('ACC_mag_mean only', 1, r2_score(y, y_pred)))
        
        X = df[['acc_intensity']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('ACC_intensity only', 1, r2_score(y, y_pred)))
    
    # 4. EDA only
    if 'eda_mean' in df.columns and df['eda_mean'].notna().sum() > 5:
        X = df[['eda_mean']].fillna(df['eda_mean'].mean()).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('EDA_mean only', 1, r2_score(y, y_pred)))
    
    if 'stress_skin_mean' in df.columns and df['stress_skin_mean'].notna().sum() > 5:
        X = df[['stress_skin_mean']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('Stress_skin only', 1, r2_score(y, y_pred)))
    
    # 5. HR + Duration (baseline from before)
    if 'hr_max' in df.columns:
        X = df[['hr_max', 'duration']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('HR + Duration', 2, r2_score(y, y_pred)))
    
    # 6. ACC + Duration
    if 'acc_mag_mean' in df.columns:
        X = df[['acc_mag_mean', 'duration']].fillna(0).values
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=loo)
        results.append(('ACC + Duration', 2, r2_score(y, y_pred)))
    
    # 7. HR + ACC + Duration
    if 'hr_max' in df.columns and 'acc_mag_mean' in df.columns:
        X = df[['hr_max', 'acc_mag_mean', 'duration']].fillna(0).values
        y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
        results.append(('HR + ACC + Duration', 3, r2_score(y, y_pred)))
    
    # 8. All available features
    avail_feat = [c for c in feature_cols if df[c].notna().sum() > 5]
    if len(avail_feat) >= 3:
        X = df[avail_feat].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = cross_val_predict(Ridge(alpha=5.0), X_scaled, y, cv=loo)
        results.append((f'All features ({len(avail_feat)})', len(avail_feat), r2_score(y, y_pred)))
    
    # Print results
    print(f"\n  {'Method':<35} {'# Feat':>8} {'CV R²':>10}")
    print("  " + "-"*55)
    for name, n_feat, r2 in sorted(results, key=lambda x: -x[2]):
        print(f"  {name:<35} {n_feat:>8} {r2:>10.3f}")
    
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\n  ★ Best: {best[0]} with CV R² = {best[2]:.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
