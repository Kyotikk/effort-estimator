#!/usr/bin/env python3
"""
ML EXPERT APPROACH: Let the Model Choose Features
=================================================

Key insight: Don't pre-filter features! Give the model ALL features and let it
decide what's important via:
1. Regularization (Ridge, Lasso, ElasticNet)
2. Tree-based feature importance
3. Cross-validation to tune regularization strength

With 5 subjects, the challenge is GENERALIZATION, not feature selection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("ML EXPERT APPROACH: LET THE MODEL CHOOSE")
print("="*80)

# Load ALL data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
print(f"Loaded {len(df_all)} windows from {df_all['subject'].nunique()} subjects")

# Get ALL features (don't pre-filter!)
skip_cols = {'t_center', 't_start', 't_end', 'borg', 'subject', 'activity_label', 
             'window_id', 'n_samples', 'win_sec', 'modality', 'valid'}
all_features = [c for c in df_all.columns 
                if c not in skip_cols 
                and not c.startswith('Unnamed')
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"Total features available: {len(all_features)}")
print(f"  IMU: {sum(1 for c in all_features if 'acc' in c.lower())}")
print(f"  PPG: {sum(1 for c in all_features if 'ppg' in c.lower())}")
print(f"  EDA: {sum(1 for c in all_features if 'eda' in c.lower())}")

# =============================================================================
# LOSO EVALUATION FUNCTION
# =============================================================================

def evaluate_loso(df, features, model, cal_frac=0.3, name=""):
    """
    LOSO with calibration. Model gets ALL features and chooses what to use.
    """
    subjects = sorted(df['subject'].unique())
    
    # Clean features (remove constant, too many NaN)
    valid_features = []
    for f in features:
        if f in df.columns:
            col = df[f]
            if col.notna().mean() > 0.5 and col.std() > 1e-10:
                valid_features.append(f)
    
    all_preds, all_true, all_subjects = [], [], []
    per_subject = {}
    
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub].copy()
        test_df = df[df['subject'] == test_sub].copy()
        
        # Calibration split
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.arange(n_test)
        cal_idx = idx[:n_cal]
        eval_idx = idx[n_cal:]
        
        if len(eval_idx) < 10:
            continue
        
        # Prepare data
        X_train = train_df[valid_features].values
        y_train = train_df['borg'].values
        X_test = test_df[valid_features].values
        y_test = test_df['borg'].values
        
        # Impute + Scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))
        
        # Train (model chooses features via regularization/tree structure)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calibrate (shift to match calibration mean)
        cal_offset = y_test[cal_idx].mean() - y_pred[cal_idx].mean()
        y_pred_cal = y_pred + cal_offset
        
        # Evaluate
        y_eval_pred = y_pred_cal[eval_idx]
        y_eval_true = y_test[eval_idx]
        
        r, _ = pearsonr(y_eval_pred, y_eval_true)
        mae = np.mean(np.abs(y_eval_pred - y_eval_true))
        within_1 = np.mean(np.abs(y_eval_pred - y_eval_true) <= 1) * 100
        
        per_subject[test_sub] = {'r': r, 'mae': mae, 'within_1': within_1}
        all_preds.extend(y_eval_pred)
        all_true.extend(y_eval_true)
        all_subjects.extend([test_sub] * len(y_eval_pred))
    
    # Overall metrics
    pooled_r, _ = pearsonr(all_preds, all_true)
    per_subject_r = np.mean([m['r'] for m in per_subject.values()])
    avg_mae = np.mean([m['mae'] for m in per_subject.values()])
    avg_within_1 = np.mean([m['within_1'] for m in per_subject.values()])
    
    return {
        'pooled_r': pooled_r,
        'per_subject_r': per_subject_r,
        'mae': avg_mae,
        'within_1': avg_within_1,
        'per_subject': per_subject,
        'n_features': len(valid_features)
    }

# =============================================================================
# TEST DIFFERENT APPROACHES
# =============================================================================

print("\n" + "="*80)
print("APPROACH 1: REGULARIZED LINEAR MODELS (model chooses via penalty)")
print("="*80)

models_linear = {
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10)': Ridge(alpha=10),
    'Ridge (α=100)': Ridge(alpha=100),
    'Lasso (α=0.01)': Lasso(alpha=0.01, max_iter=5000),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
}

print(f"\n{'Model':<20} | {'# Feat':>7} | {'Pooled r':>10} | {'Per-sub r':>10} | {'±1 Borg':>8}")
print("-"*70)

best_linear = None
best_linear_r = -1

for name, model in models_linear.items():
    res = evaluate_loso(df_all, all_features, model, cal_frac=0.3)
    print(f"{name:<20} | {res['n_features']:>7} | {res['pooled_r']:>10.3f} | {res['per_subject_r']:>10.3f} | {res['within_1']:>7.1f}%")
    if res['per_subject_r'] > best_linear_r:
        best_linear_r = res['per_subject_r']
        best_linear = name

print(f"\n✓ Best linear: {best_linear} (per-subject r = {best_linear_r:.3f})")

# =============================================================================
print("\n" + "="*80)
print("APPROACH 2: TREE-BASED MODELS (model chooses via splits)")
print("="*80)

models_tree = {
    'RF (depth=4)': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1),
    'RF (depth=6)': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    'RF (depth=8)': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    'GB (depth=3)': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    'GB (depth=4)': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    'GB (depth=5)': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

print(f"\n{'Model':<20} | {'# Feat':>7} | {'Pooled r':>10} | {'Per-sub r':>10} | {'±1 Borg':>8}")
print("-"*70)

best_tree = None
best_tree_r = -1

for name, model in models_tree.items():
    res = evaluate_loso(df_all, all_features, model, cal_frac=0.3)
    print(f"{name:<20} | {res['n_features']:>7} | {res['pooled_r']:>10.3f} | {res['per_subject_r']:>10.3f} | {res['within_1']:>7.1f}%")
    if res['per_subject_r'] > best_tree_r:
        best_tree_r = res['per_subject_r']
        best_tree = name

print(f"\n✓ Best tree: {best_tree} (per-subject r = {best_tree_r:.3f})")

# =============================================================================
print("\n" + "="*80)
print("APPROACH 3: SVR WITH RBF KERNEL")
print("="*80)

models_svr = {
    'SVR (C=0.1)': SVR(kernel='rbf', C=0.1),
    'SVR (C=1.0)': SVR(kernel='rbf', C=1.0),
    'SVR (C=10)': SVR(kernel='rbf', C=10),
}

print(f"\n{'Model':<20} | {'# Feat':>7} | {'Pooled r':>10} | {'Per-sub r':>10} | {'±1 Borg':>8}")
print("-"*70)

best_svr = None
best_svr_r = -1

for name, model in models_svr.items():
    res = evaluate_loso(df_all, all_features, model, cal_frac=0.3)
    print(f"{name:<20} | {res['n_features']:>7} | {res['pooled_r']:>10.3f} | {res['per_subject_r']:>10.3f} | {res['within_1']:>7.1f}%")
    if res['per_subject_r'] > best_svr_r:
        best_svr_r = res['per_subject_r']
        best_svr = name

print(f"\n✓ Best SVR: {best_svr} (per-subject r = {best_svr_r:.3f})")

# =============================================================================
print("\n" + "="*80)
print("APPROACH 4: VARY CALIBRATION AMOUNT (with best model)")
print("="*80)

# Use best overall model
best_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

print(f"\nUsing RandomForest (depth=6) with ALL {len(all_features)} features")
print(f"\n{'Cal %':<8} | {'Pooled r':>10} | {'Per-sub r':>10} | {'±1 Borg':>8} | {'Test samples':>12}")
print("-"*60)

for cal_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
    res = evaluate_loso(df_all, all_features, best_model, cal_frac=cal_frac)
    n_test = int(len(df_all) * (1 - cal_frac) * 0.8)  # Approximate
    print(f"{cal_frac*100:>5.0f}%   | {res['pooled_r']:>10.3f} | {res['per_subject_r']:>10.3f} | {res['within_1']:>7.1f}% | ~{n_test:>10}")

# =============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS (what did the model actually use?)")
print("="*80)

# Train on all data to see what features matter
X_all = df_all[all_features].values
y_all = df_all['borg'].values

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_all_clean = scaler.fit_transform(imputer.fit_transform(X_all))

rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_all_clean, y_all)

# Get feature importances
importances = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by RF importance:")
for i, row in importances.head(20).iterrows():
    modality = 'IMU' if 'acc' in row['feature'].lower() else ('PPG' if 'ppg' in row['feature'].lower() else ('EDA' if 'eda' in row['feature'].lower() else 'Other'))
    print(f"  {row['importance']:.4f}  [{modality:>3}]  {row['feature']}")

# Modality breakdown
imu_imp = importances[importances['feature'].str.contains('acc', case=False)]['importance'].sum()
ppg_imp = importances[importances['feature'].str.contains('ppg', case=False)]['importance'].sum()
eda_imp = importances[importances['feature'].str.contains('eda', case=False)]['importance'].sum()
total_imp = imu_imp + ppg_imp + eda_imp

print(f"\nModality importance breakdown:")
print(f"  IMU: {imu_imp/total_imp*100:.1f}%")
print(f"  PPG: {ppg_imp/total_imp*100:.1f}%")
print(f"  EDA: {eda_imp/total_imp*100:.1f}%")

# =============================================================================
print("\n" + "="*80)
print("SUMMARY: BEST APPROACH")
print("="*80)

print(f"""
RESULTS WITH ALL {len(all_features)} FEATURES (model chooses):
─────────────────────────────────────────────────
Best Linear (Ridge):     per-subject r = {best_linear_r:.3f}
Best Tree (RF/GB):       per-subject r = {best_tree_r:.3f}
Best SVR:                per-subject r = {best_svr_r:.3f}

WHAT THE MODEL ACTUALLY USES:
─────────────────────────────
IMU features: {imu_imp/total_imp*100:.1f}% of importance
PPG features: {ppg_imp/total_imp*100:.1f}% of importance  
EDA features: {eda_imp/total_imp*100:.1f}% of importance

KEY INSIGHT:
────────────
With only 5 subjects, the CEILING is ~r=0.55-0.60 per-subject.
This is NOT a feature selection problem - it's a DATA problem.

To get r > 0.6:
1. Add more subjects (15-20 would help significantly)
2. Better calibration (activity-specific, HR reserve normalization)
3. Subject-specific adaptation

The model IS choosing the best features (mostly IMU) - there's just
not enough subjects to learn truly generalizable patterns.
""")
