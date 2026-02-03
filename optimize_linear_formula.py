#!/usr/bin/env python3
"""
Try different physiological formula variations to find the best predictor.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

print("="*70)
print("OPTIMIZING LINEAR FORMULA")
print("="*70)

# ============================================================================
# Load data
# ============================================================================
def parse_time(t):
    try:
        dt = datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None

adl = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv', skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

hr = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_heart_rate/data_1.csv.gz')
hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})

# Get resting HR (5th percentile as proxy)
hr_rest = hr['heart_rate'].quantile(0.05)
hr_max_observed = hr['heart_rate'].max()
print(f"HR range: {hr['heart_rate'].min():.0f} - {hr_max_observed:.0f} bpm")
print(f"HR rest (5th %ile): {hr_rest:.0f} bpm")

# Offset between ADL and HR timestamps
hr_start = hr['timestamp'].min()
adl_start = adl['timestamp'].min()
offset = adl_start - hr_start

# Parse activities with detailed HR features
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
                'hr_min': hr_vals.min(),
                'hr_std': hr_vals.std() if len(hr_vals) > 1 else 0,
                'hr_delta': hr_vals.max() - hr_vals.min(),
                'hr_reserve': hr_vals.mean() - hr_rest,  # Above resting
                'hr_pct_max': hr_vals.mean() / hr_max_observed * 100,
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
        current = None

df = pd.DataFrame(activities).dropna()
print(f"Activities: {len(df)}")

# ============================================================================
# Try different formulas
# ============================================================================
print("\n" + "="*70)
print("FORMULA VARIATIONS (all with LOO-CV)")
print("="*70)

formulas = {
    # Original literature formulas
    'HR_mean × √duration': df['hr_mean'] * np.sqrt(df['duration']),
    'HR_mean × duration': df['hr_mean'] * df['duration'],
    'HR_mean × log(duration)': df['hr_mean'] * np.log(df['duration'] + 1),
    
    # Heart rate reserve (above resting)
    'HR_reserve × √duration': df['hr_reserve'] * np.sqrt(df['duration']),
    'HR_reserve × duration': df['hr_reserve'] * df['duration'],
    
    # Peak HR based
    'HR_max × √duration': df['hr_max'] * np.sqrt(df['duration']),
    'HR_delta × √duration': df['hr_delta'] * np.sqrt(df['duration']),
    
    # Simple features
    'HR_mean only': df['hr_mean'],
    'Duration only': df['duration'],
    'HR_max only': df['hr_max'],
    'HR_reserve only': df['hr_reserve'],
    
    # %HRmax based (TRIMP-like)
    '%HR_max × duration': df['hr_pct_max'] * df['duration'],
    '%HR_max × √duration': df['hr_pct_max'] * np.sqrt(df['duration']),
}

results = []
loo = LeaveOneOut()
y = df['borg'].values

for name, formula in formulas.items():
    X = formula.values.reshape(-1, 1)
    model = LinearRegression()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    
    cv_r2 = r2_score(y, y_pred)
    cv_mae = mean_absolute_error(y, y_pred)
    cv_r, _ = pearsonr(y, y_pred)
    
    results.append({
        'Formula': name,
        'CV R²': cv_r2,
        'CV r': cv_r,
        'CV MAE': cv_mae
    })

results_df = pd.DataFrame(results).sort_values('CV R²', ascending=False)
print("\nRanked by CV R²:")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{row['Formula']:30s}  R²={row['CV R²']:.3f}  r={row['CV r']:.3f}  MAE={row['CV MAE']:.2f}")

# ============================================================================
# Try multi-feature models
# ============================================================================
print("\n" + "="*70)
print("MULTI-FEATURE MODELS (LOO-CV)")
print("="*70)

feature_sets = {
    'HR_mean + duration': ['hr_mean', 'duration'],
    'HR_mean + duration + HR×dur': ['hr_mean', 'duration', 'hr_x_dur'],
    'HR_mean + HR_max + duration': ['hr_mean', 'hr_max', 'duration'],
    'HR_reserve + duration': ['hr_reserve', 'duration'],
    'All HR features + duration': ['hr_mean', 'hr_max', 'hr_min', 'hr_std', 'hr_delta', 'duration'],
    'HR_mean + HR_std + duration': ['hr_mean', 'hr_std', 'duration'],
}

# Add derived features
df['hr_x_dur'] = df['hr_mean'] * df['duration']
df['hr_x_sqrt_dur'] = df['hr_mean'] * np.sqrt(df['duration'])
df['sqrt_dur'] = np.sqrt(df['duration'])
df['log_dur'] = np.log(df['duration'] + 1)

multi_results = []

for name, features in feature_sets.items():
    try:
        X = df[features].values
        # Use Ridge to handle potential multicollinearity
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X, y, cv=loo)
        
        cv_r2 = r2_score(y, y_pred)
        cv_mae = mean_absolute_error(y, y_pred)
        cv_r, _ = pearsonr(y, y_pred)
        
        multi_results.append({
            'Features': name,
            'CV R²': cv_r2,
            'CV r': cv_r,
            'CV MAE': cv_mae
        })
    except Exception as e:
        print(f"Error with {name}: {e}")

multi_df = pd.DataFrame(multi_results).sort_values('CV R²', ascending=False)
print("\nRanked by CV R²:")
print("-"*70)
for _, row in multi_df.iterrows():
    print(f"{row['Features']:35s}  R²={row['CV R²']:.3f}  r={row['CV r']:.3f}  MAE={row['CV MAE']:.2f}")

# ============================================================================
# Best model summary
# ============================================================================
print("\n" + "="*70)
print("BEST RESULTS SUMMARY")
print("="*70)

best_single = results_df.iloc[0]
best_multi = multi_df.iloc[0]

print(f"""
Best single formula: {best_single['Formula']}
  CV R² = {best_single['CV R²']:.3f}
  CV MAE = {best_single['CV MAE']:.2f}

Best multi-feature: {best_multi['Features']}
  CV R² = {best_multi['CV R²']:.3f}
  CV MAE = {best_multi['CV MAE']:.2f}

Comparison:
  XGBoost (287 features):  CV R² = 0.066, MAE = 1.25
  Best linear formula:     CV R² = {max(best_single['CV R²'], best_multi['CV R²']):.3f}, MAE = {min(best_single['CV MAE'], best_multi['CV MAE']):.2f}
""")
