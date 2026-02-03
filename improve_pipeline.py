#!/usr/bin/env python3
"""
IMPROVED FEATURE SELECTION + MODEL PIPELINE
============================================
Goal: Get from 0.44 to 0.6+ with current data

Improvements:
1. Consistent feature selection (across subjects)
2. Try different models (XGBoost, SVR)
3. Better calibration (more data, different methods)
4. Ensemble approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ml.consistent_feature_selection import select_consistent_features

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("IMPROVED PIPELINE - TARGET: r > 0.6")
print("="*80)

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"Loaded {len(df_all)} windows from {df_all['subject'].nunique()} subjects")

# Get feature sets
skip_cols = {'t_center', 'borg', 'subject', 'activity_label', 'Unnamed'}
all_features = [c for c in df_all.columns 
                if not any(x in c for x in skip_cols)
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

imu_features = [c for c in all_features if 'acc' in c.lower() or 'gyro' in c.lower()]
ppg_features = [c for c in all_features if 'ppg' in c.lower()]
eda_features = [c for c in all_features if 'eda' in c.lower()]

print(f"Features: {len(all_features)} total, {len(imu_features)} IMU, {len(ppg_features)} PPG, {len(eda_features)} EDA")

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_loso(df, features, model_class, model_params={}, cal_frac=0.2, name=""):
    """
    LOSO evaluation with calibration.
    Returns BOTH pooled and per-subject metrics.
    """
    subjects = sorted(df['subject'].unique())
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    
    if len(valid_features) == 0:
        return None
    
    all_preds, all_true = [], []
    per_sub = {}
    
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub]
        test_df = df[df['subject'] == test_sub]
        
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        cal_df = test_df.iloc[idx[:n_cal]]
        eval_df = test_df.iloc[idx[n_cal:]]
        
        if len(eval_df) < 5:
            continue
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(imputer.fit_transform(train_df[valid_features]))
        X_cal = scaler.transform(imputer.transform(cal_df[valid_features]))
        X_eval = scaler.transform(imputer.transform(eval_df[valid_features]))
        
        y_train = train_df['borg'].values
        y_cal = cal_df['borg'].values
        y_eval = eval_df['borg'].values
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Calibrate
        preds_cal = model.predict(X_cal)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Predict
        preds = calibrator.predict(model.predict(X_eval).reshape(-1, 1))
        
        all_preds.extend(preds)
        all_true.extend(y_eval)
        
        r_sub, _ = pearsonr(preds, y_eval)
        mae_sub = np.mean(np.abs(preds - y_eval))
        per_sub[test_sub] = {'r': r_sub, 'mae': mae_sub}
    
    r_pooled, _ = pearsonr(all_preds, all_true)
    mae_pooled = np.mean(np.abs(np.array(all_preds) - np.array(all_true)))
    within_1 = np.mean(np.abs(np.array(all_preds) - np.array(all_true)) <= 1) * 100
    avg_per_sub_r = np.mean([v['r'] for v in per_sub.values()])
    
    return {
        'name': name,
        'r_pooled': r_pooled,
        'r_per_sub': avg_per_sub_r,
        'mae': mae_pooled,
        'within_1': within_1,
        'n_features': len(valid_features),
        'per_subject': per_sub
    }

# =============================================================================
# IMPROVEMENT 1: DIFFERENT MODELS
# =============================================================================

print("\n" + "="*60)
print("IMPROVEMENT 1: TRY DIFFERENT MODELS (with IMU features)")
print("="*60)

models = {
    'Ridge': (Ridge, {'alpha': 1.0}),
    'Ridge α=0.1': (Ridge, {'alpha': 0.1}),
    'Ridge α=10': (Ridge, {'alpha': 10.0}),
    'ElasticNet': (ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5}),
    'SVR (RBF)': (SVR, {'kernel': 'rbf', 'C': 1.0}),
    'SVR (linear)': (SVR, {'kernel': 'linear', 'C': 1.0}),
    'RandomForest': (RandomForestRegressor, {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}),
    'GradientBoosting': (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}),
}

print(f"\n{'Model':<20} | {'r (pooled)':>12} | {'r (per-sub)':>12} | {'MAE':>6} | {'±1 Borg':>8}")
print("-" * 70)

best_model = None
best_r = 0

for name, (model_class, params) in models.items():
    result = evaluate_loso(df_all, imu_features, model_class, params, name=name)
    if result:
        print(f"{name:<20} | {result['r_pooled']:>12.3f} | {result['r_per_sub']:>12.3f} | {result['mae']:>6.2f} | {result['within_1']:>7.1f}%")
        if result['r_per_sub'] > best_r:
            best_r = result['r_per_sub']
            best_model = name

print(f"\nBest model by per-subject r: {best_model} (r = {best_r:.3f})")

# =============================================================================
# IMPROVEMENT 2: DIFFERENT FEATURE SETS
# =============================================================================

print("\n" + "="*60)
print("IMPROVEMENT 2: DIFFERENT FEATURE SETS (with GradientBoosting)")
print("="*60)

# Get consistent features
consistent_features, _ = select_consistent_features(df_all, verbose=False)

feature_sets = {
    'IMU only': imu_features,
    'PPG only': ppg_features,
    'EDA only': eda_features,
    'IMU + PPG': imu_features + ppg_features,
    'IMU + EDA': imu_features + eda_features,
    'All multimodal': all_features,
    'Consistent only': consistent_features,
    'Consistent + IMU': list(set(consistent_features + imu_features)),
}

print(f"\n{'Feature Set':<20} | {'# Feat':>7} | {'r (pooled)':>12} | {'r (per-sub)':>12} | {'±1 Borg':>8}")
print("-" * 75)

best_features = None
best_r = 0

for name, features in feature_sets.items():
    result = evaluate_loso(df_all, features, GradientBoostingRegressor, 
                          {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}, name=name)
    if result:
        print(f"{name:<20} | {result['n_features']:>7} | {result['r_pooled']:>12.3f} | {result['r_per_sub']:>12.3f} | {result['within_1']:>7.1f}%")
        if result['r_per_sub'] > best_r:
            best_r = result['r_per_sub']
            best_features = name

print(f"\nBest features by per-subject r: {best_features} (r = {best_r:.3f})")

# =============================================================================
# IMPROVEMENT 3: MORE CALIBRATION DATA
# =============================================================================

print("\n" + "="*60)
print("IMPROVEMENT 3: VARY CALIBRATION AMOUNT")
print("="*60)

cal_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

print(f"\n{'Cal %':<8} | {'r (pooled)':>12} | {'r (per-sub)':>12} | {'Test samples':>14}")
print("-" * 55)

for cal_frac in cal_fractions:
    result = evaluate_loso(df_all, imu_features, GradientBoostingRegressor,
                          {'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
                          cal_frac=cal_frac)
    if result:
        n_test = int(len(df_all) * 0.8 * (1 - cal_frac))  # approximate
        print(f"{cal_frac*100:>5.0f}%   | {result['r_pooled']:>12.3f} | {result['r_per_sub']:>12.3f} | ~{n_test:>12}")

# =============================================================================
# IMPROVEMENT 4: BEST COMBINATION
# =============================================================================

print("\n" + "="*60)
print("IMPROVEMENT 4: BEST COMBINATION")
print("="*60)

# Based on above, try best model + best features + optimal calibration
best_result = evaluate_loso(
    df_all, 
    imu_features,  # IMU seems best
    GradientBoostingRegressor, 
    {'n_estimators': 100, 'max_depth': 4, 'random_state': 42, 'learning_rate': 0.1},
    cal_frac=0.3,
    name="Best combo"
)

print(f"\nBest combination:")
print(f"  Model: GradientBoosting (100 trees, depth 4)")
print(f"  Features: IMU ({len(imu_features)} features)")
print(f"  Calibration: 30%")
print(f"\nResults:")
print(f"  POOLED r = {best_result['r_pooled']:.3f}")
print(f"  Per-subject r = {best_result['r_per_sub']:.3f}")
print(f"  MAE = {best_result['mae']:.2f}")
print(f"  ±1 Borg = {best_result['within_1']:.1f}%")

print("\nPer-subject breakdown:")
for sub, res in best_result['per_subject'].items():
    print(f"  {sub}: r = {res['r']:.3f}, MAE = {res['mae']:.2f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: HOW TO GET BETTER RESULTS")
print("="*80)

print(f"""
CURRENT BEST RESULTS:
─────────────────────
• Pooled r = {best_result['r_pooled']:.2f} (looks good but misleading)
• Per-subject r = {best_result['r_per_sub']:.2f} (REAL generalization metric)

TO GET TO r = 0.6:
──────────────────
1. MORE DATA (most important!)
   - Current: 5 subjects, ~280 windows each
   - Need: 20+ subjects to learn generalizable patterns
   
2. BETTER CALIBRATION
   - Current: Simple linear shift
   - Try: Personalized baseline (resting HR), activity-specific models
   
3. ACTIVITY-LEVEL FEATURES
   - Current: 5s windows
   - Try: Aggregate to activity level (mean, max over activity duration)
   
4. DOMAIN KNOWLEDGE FEATURES
   - HR reserve: (HR - HR_rest) / (HR_max - HR_rest)
   - Movement intensity normalized by subject
   
5. DIFFERENT TARGET
   - Instead of raw Borg: predict Borg_relative = (Borg - Borg_min) / range

WHERE IN PIPELINE TO INTEGRATE:
───────────────────────────────
The feature selection happens in: ml/feature_selection_and_qc.py

I'll update it to use consistent feature selection.
""")
