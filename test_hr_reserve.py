#!/usr/bin/env python3
"""
Test HR Reserve Normalization Approaches

When you don't know true HR_max, use:
1. Age-predicted: 220 - age or 208 - 0.7*age (Tanaka formula)
2. Session max: max HR observed during recording
3. Percentage of range: (HR - HR_min) / (HR_max - HR_min)

For elderly subjects, assume age ~70-80 (adjust if you know actual ages)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load all data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

print('='*70)
print('HR RESERVE NORMALIZATION TEST')
print('='*70)

# Check HR data
hr_col = 'ppg_green_hr_mean'
print(f"\nRaw HR stats per subject:")
for subj in sorted(df_all['subject'].unique()):
    subj_df = df_all[df_all['subject'] == subj]
    hr = subj_df[hr_col].dropna()
    print(f"  {subj}: HR mean={hr.mean():.1f}, min={hr.min():.1f}, max={hr.max():.1f}, range={hr.max()-hr.min():.1f}")

# =============================================================================
# CREATE NORMALIZED HR FEATURES
# =============================================================================

print("\n" + "="*70)
print("CREATING NORMALIZED HR FEATURES")
print("="*70)

# Assume elderly age ~75 (adjust if known)
assumed_age = 75
hr_max_age_predicted = 220 - assumed_age  # Classic: 145 bpm for 75yo
hr_max_tanaka = 208 - 0.7 * assumed_age   # Tanaka: 155.5 bpm for 75yo

print(f"\nAssuming age = {assumed_age}")
print(f"  Classic HR_max (220-age): {hr_max_age_predicted} bpm")
print(f"  Tanaka HR_max (208-0.7*age): {hr_max_tanaka:.1f} bpm")

# Create normalized features for each subject
df_norm = df_all.copy()

for subj in df_all['subject'].unique():
    mask = df_norm['subject'] == subj
    hr = df_norm.loc[mask, hr_col]
    
    # Get subject's observed HR range
    hr_min_session = hr.min()
    hr_max_session = hr.max()
    hr_rest = hr.quantile(0.05)  # Approximate resting HR (5th percentile)
    
    # Method 1: % of age-predicted max
    df_norm.loc[mask, 'hr_pct_age_max'] = hr / hr_max_age_predicted * 100
    
    # Method 2: HR reserve with age-predicted max
    # HR_reserve = (HR - HR_rest) / (HR_max - HR_rest)
    df_norm.loc[mask, 'hr_reserve_age'] = (hr - hr_rest) / (hr_max_age_predicted - hr_rest) * 100
    
    # Method 3: HR reserve with session-observed max (more practical!)
    df_norm.loc[mask, 'hr_reserve_session'] = (hr - hr_rest) / (hr_max_session - hr_rest) * 100
    
    # Method 4: Simple % of session range (0-100%)
    df_norm.loc[mask, 'hr_pct_range'] = (hr - hr_min_session) / (hr_max_session - hr_min_session) * 100
    
    # Method 5: Z-score per subject (mean=0, std=1)
    df_norm.loc[mask, 'hr_zscore'] = (hr - hr.mean()) / hr.std()
    
    print(f"\n{subj}: HR_rest(5%)={hr_rest:.1f}, HR_max_session={hr_max_session:.1f}")

# =============================================================================
# TEST NORMALIZED HR FEATURES
# =============================================================================

print("\n" + "="*70)
print("TESTING NORMALIZED HR FEATURES")
print("="*70)

# Also get IMU for comparison
imu_features = [c for c in df_all.columns if 'acc' in c.lower()]

def evaluate_random_cal(df, features, model, cal_frac=0.2):
    subjects = sorted(df['subject'].unique())
    valid_features = [f for f in features if f in df.columns 
                      and df[f].notna().mean() > 0.5 
                      and df[f].std() > 1e-10]
    
    if len(valid_features) == 0:
        return float('nan'), {}
    
    per_subject = {}
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub]
        test_df = df[df['subject'] == test_sub]
        
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        cal_idx = idx[:n_cal]
        eval_idx = idx[n_cal:]
        
        if len(eval_idx) < 5:
            continue
        
        X_train = train_df[valid_features].values
        y_train = train_df['borg'].values
        X_test = test_df[valid_features].values
        y_test = test_df['borg'].values
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cal_offset = y_test[cal_idx].mean() - y_pred[cal_idx].mean()
        y_pred_cal = y_pred + cal_offset
        
        r, _ = pearsonr(y_pred_cal[eval_idx], y_test[eval_idx])
        per_subject[test_sub] = r
    
    return np.mean(list(per_subject.values())), per_subject

# Test different HR normalization methods
hr_norm_features = {
    'Raw HR mean': [hr_col],
    '% of age-predicted max': ['hr_pct_age_max'],
    'HR reserve (age max)': ['hr_reserve_age'],
    'HR reserve (session max)': ['hr_reserve_session'],
    '% of session range': ['hr_pct_range'],
    'HR z-score': ['hr_zscore'],
}

print(f"\n{'Method':<30} | {'Per-sub r':>10} | {'Pooled r':>10} | Breakdown")
print('-'*90)

for name, features in hr_norm_features.items():
    # Check correlation with Borg first
    valid = df_norm[features[0]].notna()
    pooled_r, _ = pearsonr(df_norm.loc[valid, features[0]], df_norm.loc[valid, 'borg'])
    
    # LOSO evaluation
    avg_r, per_sub = evaluate_random_cal(df_norm, features, Ridge(alpha=1.0), cal_frac=0.2)
    breakdown = ' '.join([f"{s[-1]}:{r:.2f}" for s, r in per_sub.items()])
    print(f"{name:<30} | {avg_r:>10.3f} | {pooled_r:>10.3f} | {breakdown}")

# =============================================================================
# TEST COMBINING NORMALIZED HR WITH IMU
# =============================================================================

print("\n" + "="*70)
print("COMBINING NORMALIZED HR WITH IMU")
print("="*70)

print(f"\n{'Method':<35} | {'Per-sub r':>10} | Breakdown")
print('-'*90)

configs = [
    ('IMU only (baseline)', imu_features),
    ('IMU + raw HR', imu_features + [hr_col]),
    ('IMU + HR reserve (session)', imu_features + ['hr_reserve_session']),
    ('IMU + HR % range', imu_features + ['hr_pct_range']),
    ('IMU + HR z-score', imu_features + ['hr_zscore']),
    ('IMU + all HR norm', imu_features + ['hr_reserve_session', 'hr_pct_range', 'hr_zscore']),
]

for name, features in configs:
    avg_r, per_sub = evaluate_random_cal(df_norm, features, 
                                          RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
                                          cal_frac=0.2)
    breakdown = ' '.join([f"{s[-1]}:{r:.2f}" for s, r in per_sub.items()])
    print(f"{name:<35} | {avg_r:>10.3f} | {breakdown}")

# =============================================================================
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
HR RESERVE NORMALIZATION OPTIONS:
─────────────────────────────────
1. Age-predicted max: 220 - age (classic) or 208 - 0.7*age (Tanaka)
   - Requires knowing age
   - May overestimate max for elderly
   
2. Session-observed max: max(HR) from recording
   - No age needed
   - May underestimate true max if no high-intensity activity
   
3. % of session range: (HR - HR_min) / (HR_max - HR_min)
   - Simple and robust
   - Captures relative effort within session

4. Z-score: (HR - mean) / std per subject
   - Standardizes across subjects
   - Good for ML models

WHICH IS BEST?
──────────────
Check results above - typically session-based normalization works
as well as age-predicted because we're measuring RELATIVE changes
within a session, not absolute fitness levels.
""")
