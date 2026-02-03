"""
IMPROVED GRANULAR SELECTION
============================
Better algorithm: Start from best modality, then try adding/removing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject_id'] = f'elderly{i}'
        all_dfs.append(df)
df = pd.concat(all_dfs, ignore_index=True)

# Get features
exclude = ['subject', 'borg', 'time', 'window', 'activity', 'source', 't_center', 't_start', 't_end']
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                if not any(x in c.lower() for x in exclude)]
valid_cols = [c for c in feature_cols if df[c].notna().all() and np.isfinite(df[c]).all() and df[c].std() > 1e-10]

imu_cols = [c for c in valid_cols if c.startswith(('acc_', 'gyr_'))]
ppg_cols = [c for c in valid_cols if c.startswith('ppg_')]
eda_cols = [c for c in valid_cols if c.startswith('eda_')]

df_clean = df.dropna(subset=['borg'] + valid_cols)
subjects = df_clean['subject_id'].values
y = df_clean['borg'].values

def loso_score(feature_list):
    if len(feature_list) == 0:
        return -1
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
        if np.std(pred) > 0:
            r = np.corrcoef(y[test_mask], pred)[0,1]
            per_subj_r.append(r)
    return np.mean(per_subj_r) if per_subj_r else -1

def get_modality(feat):
    if feat.startswith(('acc_', 'gyr_')): return 'IMU'
    if feat.startswith('ppg_'): return 'PPG'
    if feat.startswith('eda_'): return 'EDA'
    return '???'

print('='*70)
print('FINDING THE ABSOLUTE BEST FEATURE COMBINATION')
print('='*70)

# Step 1: Baseline scores for each modality
print('\nStep 1: Baseline modality scores')
print('-'*50)
imu_score = loso_score(imu_cols)
ppg_score = loso_score(ppg_cols)
eda_score = loso_score(eda_cols)
print(f'  IMU ({len(imu_cols)} features): r = {imu_score:.3f}')
print(f'  PPG ({len(ppg_cols)} features): r = {ppg_score:.3f}')
print(f'  EDA ({len(eda_cols)} features): r = {eda_score:.3f}')

# Step 2: Start from IMU (best modality), try REMOVING features
print('\nStep 2: Start from IMU, try removing features (backward elimination)')
print('-'*70)

selected = imu_cols.copy()
best_score = imu_score
improved = True
iteration = 0

while improved and len(selected) > 5:
    improved = False
    iteration += 1
    worst_feat = None
    best_after_removal = best_score
    
    # Try removing each feature
    for feat in selected:
        candidate = [f for f in selected if f != feat]
        score = loso_score(candidate)
        if score > best_after_removal:
            best_after_removal = score
            worst_feat = feat
    
    if worst_feat:
        selected.remove(worst_feat)
        best_score = best_after_removal
        print(f'  Removed: {worst_feat[:45]:45} → r = {best_score:.3f}')
        improved = True
    
    if iteration >= 20:  # Limit iterations
        break

print(f'\nAfter backward elimination: {len(selected)} IMU features, r = {best_score:.3f}')

# Step 3: Try ADDING features from PPG/EDA
print('\nStep 3: Try adding best PPG/EDA features')
print('-'*70)

other_features = ppg_cols + eda_cols
for feat in other_features:
    candidate = selected + [feat]
    score = loso_score(candidate)
    if score > best_score:
        selected.append(feat)
        best_score = score
        print(f'  + {feat[:45]:45} [{get_modality(feat):3}] → r = {best_score:.3f}')

print()
print('='*70)
print('FINAL OPTIMAL FEATURE SET')
print('='*70)

modality_count = {'IMU': 0, 'PPG': 0, 'EDA': 0}
for feat in selected:
    mod = get_modality(feat)
    modality_count[mod] += 1

print(f'\nTotal features: {len(selected)}')
print(f'Modality mix: {modality_count}')
print(f'LOSO r = {best_score:.3f}')

print('\nSelected features:')
for feat in sorted(selected, key=get_modality):
    print(f'  [{get_modality(feat):3}] {feat}')

print()
print('='*70)
print('COMPARISON SUMMARY')
print('='*70)
print(f'  All IMU (58):     r = {imu_score:.3f}')
print(f'  Optimized ({len(selected)}): r = {best_score:.3f}')
print(f'  Improvement:      {best_score - imu_score:+.3f}')

if best_score <= imu_score + 0.01:
    print('\n→ CONCLUSION: IMU alone is essentially optimal!')
    print('  Adding PPG/EDA does not improve generalization.')
else:
    print(f'\n→ CONCLUSION: Optimal mix outperforms IMU alone by {best_score - imu_score:.3f}')
