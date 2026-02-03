#!/usr/bin/env python3
"""LOSO evaluation with Top 10 features per modality."""

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
        print(f"P{i}: {len(df)} windows, {df['borg'].notna().sum()} with Borg")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df)} windows")

# Define modality groups
imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]
eda_cols = [c for c in df.columns if 'eda_' in c]

print(f"\nAll features: IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)}")

def get_top_k_features(feature_cols, k=10):
    """Get top K features by correlation with Borg (pooled)."""
    df_labeled = df.dropna(subset=['borg'])
    correlations = []
    for col in feature_cols:
        if col in df_labeled.columns:
            x = df_labeled[col].fillna(0).replace([np.inf, -np.inf], 0)
            y = df_labeled['borg']
            if x.std() > 0:
                r, _ = pearsonr(x, y)
                correlations.append((col, abs(r)))
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in correlations[:k]]

def loso_evaluate(feature_cols, name):
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
    
    mean_r = np.mean(results)
    return mean_r, results

# Get top 10 features per modality
print("\n" + "="*60)
print("SELECTING TOP 10 FEATURES PER MODALITY")
print("="*60)

top10_imu = get_top_k_features(imu_cols, 10)
top10_ppg = get_top_k_features(ppg_cols, 10)
top10_eda = get_top_k_features(eda_cols, 10)

print(f"\nTop 10 IMU features:")
for i, f in enumerate(top10_imu, 1):
    print(f"  {i}. {f}")

print(f"\nTop 10 PPG features:")
for i, f in enumerate(top10_ppg, 1):
    print(f"  {i}. {f}")

print(f"\nTop 10 EDA features:")
for i, f in enumerate(top10_eda, 1):
    print(f"  {i}. {f}")

# Run LOSO with all features
print("\n" + "="*60)
print("LOSO WITH ALL FEATURES PER MODALITY")
print("="*60)

print("\nAll IMU features (30):")
r_imu_all, _ = loso_evaluate(imu_cols, "IMU")
print(f"  → Mean LOSO r = {r_imu_all:.3f}")

print("\nAll PPG features (183):")
r_ppg_all, _ = loso_evaluate(ppg_cols, "PPG")
print(f"  → Mean LOSO r = {r_ppg_all:.3f}")

print("\nAll EDA features (47):")
r_eda_all, _ = loso_evaluate(eda_cols, "EDA")
print(f"  → Mean LOSO r = {r_eda_all:.3f}")

# Run LOSO with top 10 features
print("\n" + "="*60)
print("LOSO WITH TOP 10 FEATURES PER MODALITY")
print("="*60)

print("\nTop 10 IMU features:")
r_imu_10, results_imu = loso_evaluate(top10_imu, "IMU Top 10")
for i, r in enumerate(results_imu, 1):
    print(f"    P{i}: r = {r:.3f}")
print(f"  → Mean LOSO r = {r_imu_10:.3f}")

print("\nTop 10 PPG features:")
r_ppg_10, results_ppg = loso_evaluate(top10_ppg, "PPG Top 10")
for i, r in enumerate(results_ppg, 1):
    print(f"    P{i}: r = {r:.3f}")
print(f"  → Mean LOSO r = {r_ppg_10:.3f}")

print("\nTop 10 EDA features:")
r_eda_10, results_eda = loso_evaluate(top10_eda, "EDA Top 10")
for i, r in enumerate(results_eda, 1):
    print(f"    P{i}: r = {r:.3f}")
print(f"  → Mean LOSO r = {r_eda_10:.3f}")

# Combined Top 10 from each
print("\nCombined Top 10 from each modality (30 total):")
combined_top = top10_imu + top10_ppg + top10_eda
r_combined, _ = loso_evaluate(combined_top, "Combined")
print(f"  → Mean LOSO r = {r_combined:.3f}")

# Summary table
print("\n" + "="*60)
print("FINAL SUMMARY TABLE")
print("="*60)
print(f"""
┌────────────┬──────────┬─────────────┬─────────────┐
│ Modality   │ Features │ All (LOSO r)│ Top10 (LOSO)│
├────────────┼──────────┼─────────────┼─────────────┤
│ IMU        │ {len(imu_cols):>3}      │ {r_imu_all:.2f}        │ {r_imu_10:.2f}        │
│ PPG        │ {len(ppg_cols):>3}      │ {r_ppg_all:.2f}        │ {r_ppg_10:.2f}        │
│ EDA        │ {len(eda_cols):>3}      │ {r_eda_all:.2f}        │ {r_eda_10:.2f}        │
├────────────┼──────────┼─────────────┼─────────────┤
│ Combined   │  30      │    -        │ {r_combined:.2f}        │
└────────────┴──────────┴─────────────┴─────────────┘
""")

print("\nKEY INSIGHT:")
print(f"  - Top 10 IMU ({r_imu_10:.2f}) vs All IMU ({r_imu_all:.2f}): {'Better' if r_imu_10 > r_imu_all else 'Similar/Worse'}")
print(f"  - Top 10 PPG ({r_ppg_10:.2f}) vs All PPG ({r_ppg_all:.2f}): {'Better' if r_ppg_10 > r_ppg_all else 'Similar/Worse'}")
print(f"  - PPG still doesn't generalize even with top features")
