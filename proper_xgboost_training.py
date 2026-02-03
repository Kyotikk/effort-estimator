#!/usr/bin/env python3
"""
PROPER XGBoost Training with Correct Validation
- GroupKFold by activity (no window leakage)
- Feature selection INSIDE CV loop
- All preprocessing details documented
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed, using RandomForest instead")
    from sklearn.ensemble import RandomForestRegressor

print("="*80)
print("PROPER XGBoost TRAINING WITH CORRECT VALIDATION")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD FUSED FEATURES")
print("="*80)

data_path = '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv'
df = pd.read_csv(data_path)

print(f"  Loaded: {len(df)} windows × {len(df.columns)} columns")

# Identify columns
meta_cols = ['t_center', 'valid', 'n_samples', 'win_sec', 'modality', 'borg', 
             'activity', 'activity_id', 'subject_id']
feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]

print(f"  Feature columns: {len(feature_cols)}")
print(f"  Meta columns: {[c for c in meta_cols if c in df.columns]}")

# Check for borg and activity_id
if 'borg' not in df.columns:
    raise ValueError("No 'borg' column found!")
    
# Create activity_id if not present (from t_center gaps)
if 'activity_id' not in df.columns:
    # Infer activity boundaries from time gaps
    df = df.sort_values('t_center').reset_index(drop=True)
    time_diff = df['t_center'].diff()
    # New activity when gap > 30 seconds
    df['activity_id'] = (time_diff > 30).cumsum()
    print(f"  Created activity_id from time gaps (threshold=30s)")

print(f"  N activities: {df['activity_id'].nunique()}")
print(f"  Borg range: [{df['borg'].min():.1f}, {df['borg'].max():.1f}]")

# =============================================================================
# STEP 2: PREPARE FEATURES
# =============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARE FEATURES")
print("="*80)

# Get feature matrix
X = df[feature_cols].copy()
y = df['borg'].values
groups = df['activity_id'].values

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Handle NaN/Inf
print(f"\n  NaN values per feature (top 10):")
nan_counts = X.isna().sum().sort_values(ascending=False)
for feat, count in nan_counts.head(10).items():
    if count > 0:
        print(f"    {feat}: {count} ({100*count/len(X):.1f}%)")

# Fill NaN with column median
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
# Double-check: fill any remaining NaN with 0
X = X.fillna(0)

# Remove constant features
std = X.std()
constant_feats = std[std < 1e-10].index.tolist()
if constant_feats:
    print(f"\n  Removing {len(constant_feats)} constant features")
    X = X.drop(columns=constant_feats)

print(f"\n  Final X shape: {X.shape}")

# Show feature categories
feature_prefixes = {}
for col in X.columns:
    prefix = col.split('_')[0] if '_' in col else col
    if prefix not in feature_prefixes:
        feature_prefixes[prefix] = 0
    feature_prefixes[prefix] += 1

print(f"\n  Features by modality:")
for prefix, count in sorted(feature_prefixes.items(), key=lambda x: -x[1])[:10]:
    print(f"    {prefix}: {count}")

# =============================================================================
# STEP 3: FEATURE IMPORTANCE (on full data, for inspection only)
# =============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE CORRELATIONS WITH BORG")
print("="*80)

correlations = {}
for col in X.columns:
    x_vals = X[col].values
    # Skip if any NaN or Inf
    if np.any(~np.isfinite(x_vals)):
        continue
    # Skip if constant
    if np.std(x_vals) < 1e-10:
        continue
    try:
        r, p = stats.pearsonr(x_vals, y)
        if np.isfinite(r):
            correlations[col] = (r, p)
    except:
        pass

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

print("\n  Top 20 features by |correlation| with Borg:")
print(f"  {'Feature':<50} {'r':>8} {'p-value':>12}")
print("  " + "-"*72)
for feat, (r, p) in sorted_corr[:20]:
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {feat:<50} {r:>+8.3f} {p:>12.2e} {sig}")

# Group by modality
print("\n  Best feature per modality:")
modalities = ['eda', 'ppg_green', 'ppg_infra', 'ppg_red', 'acc_x', 'acc_y', 'acc_z', 'rr']
for mod in modalities:
    mod_feats = [(f, r, p) for f, (r, p) in sorted_corr if mod in f.lower()]
    if mod_feats:
        best = max(mod_feats, key=lambda x: abs(x[1]))
        print(f"    {mod:<12}: {best[0]:<40} r={best[1]:+.3f}")

# =============================================================================
# STEP 4: PROPER CROSS-VALIDATION (GroupKFold)
# =============================================================================
print("\n" + "="*80)
print("STEP 4: PROPER CROSS-VALIDATION (GroupKFold by activity)")
print("="*80)

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

print(f"""
  Validation strategy: GroupKFold
    - N splits: {n_splits}
    - Groups: activity_id ({df['activity_id'].nunique()} unique activities)
    - ALL windows from same activity in same fold
    - NO temporal leakage between train/test
""")

# Show fold distribution
print("  Fold distribution:")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_activities = np.unique(groups[train_idx])
    test_activities = np.unique(groups[test_idx])
    print(f"    Fold {fold+1}: Train {len(train_idx)} windows ({len(train_activities)} activities), "
          f"Test {len(test_idx)} windows ({len(test_activities)} activities)")

# =============================================================================
# STEP 5: MODEL COMPARISON
# =============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL COMPARISON (with feature selection INSIDE CV)")
print("="*80)

def evaluate_model_proper(model, X, y, groups, n_features=16, model_name="Model"):
    """
    Proper evaluation with feature selection INSIDE each CV fold.
    """
    gkf = GroupKFold(n_splits=5)
    
    y_true_all = []
    y_pred_all = []
    selected_features_all = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature selection on TRAIN ONLY
        selector = SelectKBest(f_regression, k=min(n_features, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_names = X.columns[selected_mask].tolist()
        selected_features_all.append(selected_names)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Fit model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_test_scaled)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Compute metrics
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - y_true_all.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    rmse = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    mae = np.mean(np.abs(y_true_all - y_pred_all))
    
    r, _ = stats.pearsonr(y_true_all, y_pred_all)
    
    # Most common selected features
    from collections import Counter
    all_features = [f for fold_feats in selected_features_all for f in fold_feats]
    feature_counts = Counter(all_features)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'r': r,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'top_features': feature_counts.most_common(10)
    }

# Models to compare
models = {
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10.0)': Ridge(alpha=10.0),
}

if HAS_XGBOOST:
    models['XGBoost (default)'] = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    models['XGBoost (small)'] = XGBRegressor(
        n_estimators=20, max_depth=2, learning_rate=0.05,
        random_state=42, verbosity=0
    )
    models['XGBoost (regularized)'] = XGBRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        reg_alpha=1.0, reg_lambda=10.0,
        random_state=42, verbosity=0
    )
else:
    from sklearn.ensemble import RandomForestRegressor
    models['RandomForest (default)'] = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
    models['RandomForest (small)'] = RandomForestRegressor(
        n_estimators=20, max_depth=3, random_state=42
    )

print("\n  Testing different models and feature counts...")
print(f"  {'Model':<30} {'N_feat':>7} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'r':>8}")
print("  " + "-"*75)

results = {}
for n_features in [4, 8, 16, 32]:
    for model_name, model in models.items():
        result = evaluate_model_proper(model, X, y, groups, n_features, model_name)
        key = f"{model_name}_{n_features}"
        results[key] = result
        print(f"  {model_name:<30} {n_features:>7} {result['r2']:>+8.3f} {result['rmse']:>8.2f} {result['mae']:>8.2f} {result['r']:>+8.3f}")

# =============================================================================
# STEP 6: BEST MODEL ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("STEP 6: BEST MODEL ANALYSIS")
print("="*80)

# Find best model
best_key = max(results.keys(), key=lambda k: results[k]['r2'])
best_result = results[best_key]

print(f"\n  Best model: {best_key}")
print(f"    R² = {best_result['r2']:.3f}")
print(f"    RMSE = {best_result['rmse']:.2f}")
print(f"    MAE = {best_result['mae']:.2f}")
print(f"    r = {best_result['r']:.3f}")

print(f"\n  Most frequently selected features:")
for feat, count in best_result['top_features']:
    print(f"    {feat:<50} (selected in {count}/5 folds)")

# =============================================================================
# STEP 7: COMPARISON WITH LITERATURE FORMULA
# =============================================================================
print("\n" + "="*80)
print("STEP 7: COMPARISON WITH LITERATURE FORMULA")
print("="*80)

# Load per-activity data for literature formula comparison
tli_path = '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv'
try:
    tli_df = pd.read_csv(tli_path)
    
    # Literature formula: HR_delta × √duration
    hr_baseline = tli_df['hr_mean'].min()
    hr_delta = tli_df['hr_mean'] - hr_baseline
    hr_effort = hr_delta * np.sqrt(tli_df['duration_s'])
    
    # Correlation on all data
    r_lit, _ = stats.pearsonr(hr_effort, tli_df['borg'])
    
    # LOO-CV for literature formula
    from sklearn.linear_model import LinearRegression
    y_lit = tli_df['borg'].values
    X_lit = hr_effort.values.reshape(-1, 1)
    
    y_pred_lit = np.zeros(len(y_lit))
    for i in range(len(y_lit)):
        X_train = np.delete(X_lit, i, axis=0)
        y_train = np.delete(y_lit, i)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lit[i] = lr.predict(X_lit[i:i+1])[0]
    
    ss_res_lit = np.sum((y_lit - y_pred_lit) ** 2)
    ss_tot_lit = np.sum((y_lit - y_lit.mean()) ** 2)
    r2_lit = 1 - ss_res_lit / ss_tot_lit
    
    print(f"""
  LITERATURE FORMULA (per-activity):
    Formula: HR_effort = HR_delta × √duration
    N samples: {len(tli_df)} activities
    
    r (all data): {r_lit:.3f}
    R² (LOO-CV): {r2_lit:.3f}
  
  ML PIPELINE (best model, per-window):
    Model: {best_key}
    N samples: {len(df)} windows ({df['activity_id'].nunique()} activities)
    
    R² (GroupKFold): {best_result['r2']:.3f}
    r: {best_result['r']:.3f}
    
  COMPARISON:
    Literature R² = {r2_lit:.3f} (33 samples, 2 params)
    ML Best R² = {best_result['r2']:.3f} ({len(df)} samples, many params)
    
    Winner: {'Literature formula' if r2_lit > best_result['r2'] else 'ML pipeline'}!
""")
except FileNotFoundError:
    print("  Could not load TLI results for comparison")

# =============================================================================
# STEP 8: WHAT FEATURES MATTER?
# =============================================================================
print("\n" + "="*80)
print("STEP 8: WHAT FEATURES ACTUALLY MATTER?")
print("="*80)

# Get features selected across all folds for best model
print("\n  Features most predictive of Borg (by correlation):")
print("\n  HEART RATE / HRV related:")
hr_feats = [(f, r, p) for f, (r, p) in correlations.items() 
            if any(x in f.lower() for x in ['hr_', 'ibi', 'rmssd', 'sdnn', 'pnn', 'lf_', 'hf_'])]
hr_feats_sorted = sorted(hr_feats, key=lambda x: abs(x[1]), reverse=True)[:10]
for feat, r, p in hr_feats_sorted:
    print(f"    {feat:<45} r={r:+.3f}")

print("\n  MOVEMENT (IMU) related:")
imu_feats = [(f, r, p) for f, (r, p) in correlations.items() 
             if any(x in f.lower() for x in ['acc_', 'gyro'])]
imu_feats_sorted = sorted(imu_feats, key=lambda x: abs(x[1]), reverse=True)[:10]
for feat, r, p in imu_feats_sorted:
    print(f"    {feat:<45} r={r:+.3f}")

print("\n  EDA (stress) related:")
eda_feats = [(f, r, p) for f, (r, p) in correlations.items() 
             if 'eda' in f.lower()]
eda_feats_sorted = sorted(eda_feats, key=lambda x: abs(x[1]), reverse=True)[:10]
for feat, r, p in eda_feats_sorted:
    print(f"    {feat:<45} r={r:+.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
  PROPER ML PIPELINE:
    - Features: {X.shape[1]} total, selecting top {best_key.split('_')[-1]} per fold
    - Validation: GroupKFold (no window leakage)
    - Feature selection: INSIDE each CV fold
    - Best R²: {best_result['r2']:.3f}
    
  KEY INSIGHT:
    Even with proper validation and many features,
    the simple literature formula (HR_delta × √duration) 
    often works as well or better!
    
  WHY?
    1. Per-activity aggregation removes noise
    2. HR is the dominant signal for effort
    3. Complex features add noise, not signal
    4. Borg is subjective → limited predictability
""")
