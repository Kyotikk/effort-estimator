#!/usr/bin/env python3
"""
Test the data-driven feature selection approach.

This script:
1. Loads data from all subjects
2. For each LOSO fold, automatically selects best features from ANY modality
3. Trains and evaluates, reporting honest per-subject metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/Users/pascalschlegel/effort-estimator')
from ml.best_feature_selection import select_best_features_loso, get_feature_columns, analyze_feature_selection

np.random.seed(42)

print("="*80)
print("DATA-DRIVEN FEATURE SELECTION TEST")
print("="*80)

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)
        print(f"  Loaded elderly{i}: {len(df)} windows")

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"\nTotal: {len(df_all)} windows from {df_all['subject'].nunique()} subjects")

# Get all feature columns
feature_cols = get_feature_columns(df_all)
imu_count = sum(1 for c in feature_cols if c.startswith('acc_'))
ppg_count = sum(1 for c in feature_cols if c.startswith('ppg_'))
eda_count = sum(1 for c in feature_cols if c.startswith('eda_'))
print(f"Features available: {len(feature_cols)} total (IMU={imu_count}, PPG={ppg_count}, EDA={eda_count})")

# =============================================================================
# ANALYZE FEATURE SELECTION
# =============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION ANALYSIS")
print("="*80)

# Analyze what features get selected for each fold
print("\nUsing min_subject_ratio=0.75 (at least 3 of 4 training subjects must agree)")
all_selected, common_features = analyze_feature_selection(df_all, min_corr=0.10, verbose=True)

# =============================================================================
# EVALUATE WITH DATA-DRIVEN FEATURES
# =============================================================================
print("\n" + "="*80)
print("LOSO EVALUATION WITH DATA-DRIVEN FEATURES")
print("="*80)

subjects = sorted(df_all['subject'].unique())
cal_fraction = 0.3

all_preds = []
all_true = []
per_subject_metrics = {}

for test_subj in subjects:
    # DATA-DRIVEN: Select best features using ONLY training data
    feature_cols, feature_info = select_best_features_loso(
        df_all, test_subj, min_corr=0.10, min_subject_ratio=0.75, top_n=50
    )
    
    if len(feature_cols) == 0:
        print(f"⚠️ No features for {test_subj}")
        continue
    
    mod = feature_info['modality_breakdown']
    print(f"\nTest {test_subj}: {len(feature_cols)} features "
          f"(IMU={mod['IMU']}, PPG={mod['PPG']}, EDA={mod['EDA']})")
    
    # Show top 3 features selected
    if 'details' in feature_info and len(feature_info['details']) > 0:
        print(f"  Top features: ", end="")
        top3 = [f['feature'].split('_')[-1] if '_' in f['feature'] else f['feature'] 
                for f in feature_info['details'][:3]]
        print(", ".join([feature_info['details'][i]['feature'] for i in range(min(3, len(feature_info['details'])))]))
    
    # Split data
    train_df = df_all[df_all['subject'] != test_subj]
    test_df = df_all[df_all['subject'] == test_subj]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['borg'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['borg'].values
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_raw = model.predict(X_test_scaled)
    
    # Calibration
    n_cal = max(1, int(len(y_test) * cal_fraction))
    cal_offset = y_test[:n_cal].mean() - y_pred_raw[:n_cal].mean()
    y_pred_cal = y_pred_raw + cal_offset
    
    # Evaluate on non-calibration samples
    test_idx = np.arange(n_cal, len(y_test))
    if len(test_idx) > 5:
        y_pred_test = y_pred_cal[test_idx]
        y_true_test = y_test[test_idx]
        
        r, _ = pearsonr(y_pred_test, y_true_test)
        mae = np.mean(np.abs(y_pred_test - y_true_test))
        within_1 = np.mean(np.abs(y_pred_test - y_true_test) <= 1) * 100
        
        per_subject_metrics[test_subj] = {'r': r, 'mae': mae, 'within_1': within_1}
        print(f"  Result: r={r:.3f}, MAE={mae:.2f}, ±1 Borg={within_1:.1f}%")
        
        all_preds.extend(y_pred_test)
        all_true.extend(y_true_test)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_preds = np.array(all_preds)
all_true = np.array(all_true)

pooled_r, _ = pearsonr(all_preds, all_true)
per_subject_r = np.mean([m['r'] for m in per_subject_metrics.values()])
avg_mae = np.mean([m['mae'] for m in per_subject_metrics.values()])
avg_within_1 = np.mean([m['within_1'] for m in per_subject_metrics.values()])

print(f"\nPooled r = {pooled_r:.3f} (misleading!)")
print(f"Per-subject r = {per_subject_r:.3f} ← HONEST METRIC")
print(f"Average MAE = {avg_mae:.2f}")
print(f"Average ±1 Borg = {avg_within_1:.1f}%")

print(f"\nPer-subject breakdown:")
for subj, m in per_subject_metrics.items():
    print(f"  {subj}: r={m['r']:.3f}, MAE={m['mae']:.2f}, ±1 Borg={m['within_1']:.1f}%")

# Compare to IMU-only baseline
print("\n" + "-"*40)
print("COMPARISON: Data-driven vs IMU-only")
print("-"*40)
print(f"Data-driven: per-subject r = {per_subject_r:.3f}")
print(f"IMU-only (from earlier): per-subject r ≈ 0.56")
print(f"\nIf data-driven gives mostly IMU features → IMU is genuinely best")
print(f"If data-driven includes PPG/EDA → multimodal could help with more data")
