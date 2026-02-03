"""
GRANULAR FEATURE SELECTION
===========================
Selects INDIVIDUAL features (not groups) based on LOSO generalization
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject_id'] = f'elderly{i}'
        all_dfs.append(df)
df = pd.concat(all_dfs, ignore_index=True)

# Get all valid features
exclude = ['subject', 'borg', 'time', 'window', 'activity', 'source', 't_center', 't_start', 't_end']
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                if not any(x in c.lower() for x in exclude)]
valid_cols = [c for c in feature_cols if df[c].notna().all() and np.isfinite(df[c]).all() and df[c].std() > 1e-10]

df_clean = df.dropna(subset=['borg'] + valid_cols)
subjects = df_clean['subject_id'].values
y = df_clean['borg'].values

print('='*70)
print('GRANULAR FEATURE SELECTION (Individual Features)')
print('='*70)
print(f'Testing {len(valid_cols)} individual features...')

def loso_score(feature_list):
    X = df_clean[feature_list].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    per_subj_r = []
    for test_subj in df_clean['subject_id'].unique():
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X_scaled[train_mask], y[train_mask])
        pred = model.predict(X_scaled[test_mask])
        r = np.corrcoef(y[test_mask], pred)[0,1]
        per_subj_r.append(r)
    return np.mean(per_subj_r)

def get_modality(feat):
    if feat.startswith(('acc_', 'gyr_')): return 'IMU'
    if feat.startswith('ppg_'): return 'PPG'
    if feat.startswith('eda_'): return 'EDA'
    return '???'

# Greedy forward selection
selected = []
best_score = -1
remaining = valid_cols.copy()

print('\nGreedy forward selection (adds feature only if improves LOSO r):')
print('-'*70)

for iteration in range(20):  # Max 20 features
    best_feat = None
    best_new_score = best_score
    
    for feat in remaining:
        candidate = selected + [feat]
        score = loso_score(candidate)
        if score > best_new_score:
            best_new_score = score
            best_feat = feat
    
    if best_feat is None:
        print(f'\nStopped: No feature improves beyond r={best_score:.3f}')
        break
    
    selected.append(best_feat)
    remaining.remove(best_feat)
    best_score = best_new_score
    mod = get_modality(best_feat)
    print(f'  + {best_feat[:42]:42} [{mod:3}] -> r = {best_score:.3f}')

print()
print('='*70)
print('FINAL SELECTED FEATURES:')
print('='*70)
modality_count = {}
for feat in selected:
    mod = get_modality(feat)
    modality_count[mod] = modality_count.get(mod, 0) + 1
    print(f'  [{mod:3}] {feat}')

print(f'\nModality mix: {modality_count}')
print(f'Final LOSO r = {best_score:.3f}')

print()
print('='*70)
print('COMPARISON:')
print('='*70)
imu_cols = [c for c in valid_cols if c.startswith(('acc_', 'gyr_'))]
ppg_cols = [c for c in valid_cols if c.startswith('ppg_')]
print(f'  All IMU ({len(imu_cols)} features):    LOSO r = {loso_score(imu_cols):.3f}')
print(f'  All PPG ({len(ppg_cols)} features):   LOSO r = {loso_score(ppg_cols):.3f}')
print(f'  Granular ({len(selected)} features): LOSO r = {best_score:.3f}')

print('''
INTERPRETATION:
───────────────
The granular selection picks INDIVIDUAL features from ANY modality
based on whether they improve LOSO generalization.

If some PPG features are meaningful, they will be selected!
If only IMU features generalize, only IMU will be selected.

→ This is DATA-DRIVEN, not hardcoded to any modality.
''')
