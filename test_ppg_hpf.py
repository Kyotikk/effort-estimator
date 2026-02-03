#!/usr/bin/env python3
"""
Re-run PPG preprocessing with HPF on green channel for all subjects,
then test if LOSO improves.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Parameters
FS = 32.0
HPF_CUTOFF = 0.5

def apply_hpf(values, fs=32.0, cutoff=0.5):
    """Apply 4th order Butterworth high-pass filter"""
    sos = signal.butter(4, cutoff, 'hp', fs=fs, output='sos')
    return signal.sosfilt(sos, values)

print("=" * 70)
print("RE-PROCESSING PPG GREEN WITH HPF (0.5 Hz)")
print("=" * 70)

# Process each subject
subjects_data = []

for i in range(1, 6):
    subj = f'P{i}'
    base_path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}')
    
    # Check if green PPG exists
    green_path = base_path / 'ppg_green' / 'ppg_green_preprocessed.csv'
    if not green_path.exists():
        print(f"{subj}: ppg_green not found, skipping")
        continue
    
    # Load and apply HPF
    df = pd.read_csv(green_path)
    original_mean = df['value'].mean()
    
    # Apply HPF
    df['value_hpf'] = apply_hpf(df['value'].values, fs=FS, cutoff=HPF_CUTOFF)
    new_mean = df['value_hpf'].mean()
    
    print(f"\n{subj} ppg_green:")
    print(f"  Original: mean={original_mean:.1f}, std={df['value'].std():.1f}")
    print(f"  After HPF: mean={new_mean:.1f}, std={df['value_hpf'].std():.1f}")
    
    # Save filtered version
    df_out = df[['t_unix', 't_sec']].copy()
    df_out['value'] = df['value_hpf']
    
    hpf_path = base_path / 'ppg_green' / 'ppg_green_preprocessed_hpf.csv'
    df_out.to_csv(hpf_path, index=False)
    print(f"  Saved: {hpf_path}")

print("\n" + "=" * 70)
print("NOW TESTING PPG LOSO WITH HPF-FILTERED GREEN")
print("=" * 70)

# We need to re-extract features from HPF green
# For now, let's just test with existing features but note the limitation

# Load fused data for PPG features test
print("\nLoading existing PPG features (note: these use OLD green without HPF)...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        tmp = pd.read_csv(path)
        tmp['subject'] = f'P{i}'
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

# Get PPG columns (excluding correlation columns)
ppg_cols = [c for c in df.columns if 'ppg_' in c and '_r' not in c]
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]

print(f"Data: {len(df)} windows, {df['subject'].nunique()} subjects")
print(f"PPG features: {len(ppg_cols)}")
print(f"IMU features: {len(imu_cols)}")

# Run LOSO for both modalities
def run_loso(df, feat_cols, name):
    subjects = sorted(df['subject'].unique())
    results = []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        X_train = train[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_train = train['borg'].values
        X_test = test[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y_test = test['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r, _ = pearsonr(y_test, y_pred)
        results.append({'subject': test_subj, 'r': r})
    
    results_df = pd.DataFrame(results)
    mean_r = results_df['r'].mean()
    print(f"\n{name} LOSO Results:")
    for _, row in results_df.iterrows():
        print(f"  {row['subject']}: r = {row['r']:.2f}")
    print(f"  Mean r = {mean_r:.2f}")
    return mean_r

imu_r = run_loso(df, imu_cols, "IMU")
ppg_r = run_loso(df, ppg_cols, "PPG (old, without HPF on green)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"IMU LOSO r = {imu_r:.2f}")
print(f"PPG LOSO r = {ppg_r:.2f} (NOTE: features still from old preprocessing)")
print("\nTo fully test HPF effect, need to re-run feature extraction pipeline.")
print("HPF-filtered green PPG files have been saved for each subject.")
