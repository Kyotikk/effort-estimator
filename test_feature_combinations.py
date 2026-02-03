#!/usr/bin/env python3
"""Test combining top features from each modality."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

# Load all subjects
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = i
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Define modality groups
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]
eda_cols = [c for c in df.columns if 'eda_' in c]

def loso_evaluate(feature_cols, name, verbose=False):
    """Run LOSO cross-validation for a set of features."""
    results = []
    for test_subj in sorted(df['subject'].unique()):
        train_df = df[df['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df[df['subject'] == test_subj].dropna(subset=['borg'])
        
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if len(y_test) > 2 and np.std(y_test) > 0:
            r, _ = pearsonr(y_test, y_pred)
            results.append(r)
            if verbose:
                print(f"    P{test_subj}: r = {r:.3f}")
    
    return np.mean(results)

def get_top_features_by_importance(feature_cols, n_top=10):
    """Get top N features by RF importance (trained on all data)."""
    clean_df = df.dropna(subset=['borg'])
    valid_cols = [c for c in feature_cols if c in clean_df.columns]
    X = clean_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = clean_df['borg'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.Series(rf.feature_importances_, index=valid_cols)
    top_features = importance.nlargest(n_top).index.tolist()
    return top_features

print("="*60)
print("FEATURE COMBINATION EXPERIMENT")
print("="*60)

# Get top features from each modality
print("\nSelecting top features per modality...")
top_imu = get_top_features_by_importance(imu_cols, n_top=10)
top_ppg = get_top_features_by_importance(ppg_cols, n_top=10)
top_eda = get_top_features_by_importance(eda_cols, n_top=5)

print(f"\nTop 10 IMU: {top_imu}")
print(f"\nTop 10 PPG: {top_ppg}")
print(f"\nTop 5 EDA: {top_eda}")

# Test combinations
print("\n" + "="*60)
print("LOSO RESULTS")
print("="*60)

experiments = [
    ("All IMU (30)", imu_cols),
    ("All PPG (183)", ppg_cols),
    ("All EDA (47)", eda_cols),
    ("Top 10 IMU", top_imu),
    ("Top 10 PPG", top_ppg),
    ("Top 10 IMU + Top 10 PPG", top_imu + top_ppg),
    ("Top 10 IMU + Top 5 EDA", top_imu + top_eda),
    ("Top 10 IMU + Top 10 PPG + Top 5 EDA", top_imu + top_ppg + top_eda),
    ("All modalities combined", imu_cols + ppg_cols + eda_cols),
]

results = []
for name, cols in experiments:
    r = loso_evaluate(cols, name)
    results.append((name, len(cols), r))
    print(f"  {name:<40}: r = {r:.3f} ({len(cols)} features)")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"\n{'Approach':<45} | {'N feat':>6} | {'LOSO r':>6}")
print("-"*65)
for name, n, r in sorted(results, key=lambda x: -x[2]):
    print(f"{name:<45} | {n:>6} | {r:>6.3f}")

best = max(results, key=lambda x: x[2])
print(f"\n✓ Best: {best[0]} with r = {best[2]:.3f}")
