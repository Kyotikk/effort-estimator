#!/usr/bin/env python3
"""
GRANULAR FEATURE SELECTION
==========================

Find the best INDIVIDUAL features - could be 10 IMU + 5 PPG + 2 EDA.
Not just "all IMU" or "all PPG" but the actual best mix.

Approaches:
1. Lasso - automatically zeros out bad features, keeps good ones
2. Feature importance from RF - rank all features, take top N
3. Recursive feature elimination
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

# Get ALL features
skip_cols = {'t_center', 't_start', 't_end', 'borg', 'subject', 'activity_label', 
             'window_id', 'n_samples', 'win_sec', 'modality', 'valid'}
all_features = [c for c in df_all.columns 
                if c not in skip_cols 
                and not c.startswith('Unnamed')
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']
                and df_all[c].notna().mean() > 0.5
                and df_all[c].std() > 1e-10]

print("="*80)
print("GRANULAR FEATURE SELECTION - FIND BEST INDIVIDUAL FEATURES")
print("="*80)
print(f"\nTotal features: {len(all_features)}")
print(f"  IMU: {sum(1 for c in all_features if 'acc' in c.lower())}")
print(f"  PPG: {sum(1 for c in all_features if 'ppg' in c.lower())}")
print(f"  EDA: {sum(1 for c in all_features if 'eda' in c.lower())}")


def get_modality(feature):
    """Get modality of a feature"""
    if 'acc' in feature.lower() or 'gyro' in feature.lower():
        return 'IMU'
    elif 'ppg' in feature.lower():
        return 'PPG'
    elif 'eda' in feature.lower():
        return 'EDA'
    return 'Other'


def evaluate_features(df, features, model, cal_frac=0.2):
    """Evaluate a specific set of features"""
    np.random.seed(42)
    subjects = sorted(df['subject'].unique())
    
    per_subject = {}
    for test_sub in subjects:
        train_df = df[df['subject'] != test_sub]
        test_df = df[df['subject'] == test_sub]
        
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]
        
        if len(eval_idx) < 5:
            continue
        
        X_train = train_df[features].values
        y_train = train_df['borg'].values
        X_test = test_df[features].values
        y_test = test_df['borg'].values
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))
        
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        
        cal_offset = y_test[cal_idx].mean() - y_pred[cal_idx].mean()
        y_pred_cal = y_pred + cal_offset
        
        r, _ = pearsonr(y_pred_cal[eval_idx], y_test[eval_idx])
        per_subject[test_sub] = r
    
    return np.mean(list(per_subject.values())), per_subject


# =============================================================================
print("\n" + "="*80)
print("APPROACH 1: LASSO - LET IT AUTOMATICALLY SELECT FEATURES")
print("="*80)

# Prepare data
X = df_all[all_features].values
y = df_all['borg'].values

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_clean = scaler.fit_transform(imputer.fit_transform(X))

# Use LassoCV to find optimal alpha
print("\nFinding optimal Lasso alpha...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_clean, y)
print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")

# Get selected features (non-zero coefficients)
lasso_coefs = pd.DataFrame({
    'feature': all_features,
    'coef': lasso_cv.coef_,
    'abs_coef': np.abs(lasso_cv.coef_)
}).sort_values('abs_coef', ascending=False)

lasso_selected = lasso_coefs[lasso_coefs['abs_coef'] > 0.001]['feature'].tolist()
print(f"\nLasso selected {len(lasso_selected)} features:")

# Modality breakdown
lasso_modalities = {}
for f in lasso_selected:
    mod = get_modality(f)
    lasso_modalities[mod] = lasso_modalities.get(mod, 0) + 1
print(f"  Modality mix: {lasso_modalities}")

# Show top features
print("\nTop 15 Lasso features:")
for _, row in lasso_coefs.head(15).iterrows():
    if row['abs_coef'] > 0.001:
        mod = get_modality(row['feature'])
        print(f"  [{mod:>3}] {row['feature']:<45} coef={row['coef']:.4f}")

# Evaluate Lasso-selected features
if len(lasso_selected) > 0:
    r_lasso, per_sub = evaluate_features(df_all, lasso_selected, 
                                          RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    print(f"\nLasso features + RF: per-subject r = {r_lasso:.3f}")


# =============================================================================
print("\n" + "="*80)
print("APPROACH 2: RANDOM FOREST IMPORTANCE - RANK ALL FEATURES")
print("="*80)

# Train RF on all data to get feature importances
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_clean, y)

importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 30 features by RF importance:")
for i, row in importance_df.head(30).iterrows():
    mod = get_modality(row['feature'])
    print(f"  {row['importance']:.4f}  [{mod:>3}]  {row['feature']}")

# Modality breakdown of top 30
top30_modalities = {}
for f in importance_df.head(30)['feature']:
    mod = get_modality(f)
    top30_modalities[mod] = top30_modalities.get(mod, 0) + 1
print(f"\nTop 30 modality mix: {top30_modalities}")


# =============================================================================
print("\n" + "="*80)
print("APPROACH 3: EVALUATE TOP N FEATURES (FIND OPTIMAL N)")
print("="*80)

# Test different numbers of top features
print(f"\n{'Top N':<8} | {'Modality Mix':<25} | {'Per-sub r':>10}")
print("-"*55)

best_n = 0
best_r = 0
best_features = []

for n in [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
    top_n_features = importance_df.head(n)['feature'].tolist()
    
    # Modality mix
    mix = {}
    for f in top_n_features:
        mod = get_modality(f)
        mix[mod] = mix.get(mod, 0) + 1
    mix_str = ", ".join([f"{k}:{v}" for k, v in sorted(mix.items())])
    
    r, _ = evaluate_features(df_all, top_n_features,
                              RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    print(f"{n:<8} | {mix_str:<25} | {r:>10.3f}")
    
    if r > best_r:
        best_r = r
        best_n = n
        best_features = top_n_features

print(f"\n✓ Best: Top {best_n} features → per-subject r = {best_r:.3f}")


# =============================================================================
print("\n" + "="*80)
print("APPROACH 4: GREEDY FORWARD SELECTION (BEST OF EACH)")
print("="*80)

# Start with top feature, add one at a time if it improves
print("\nGreedy forward selection (add feature if it improves r)...")

# Start with empty set
selected = []
remaining = importance_df['feature'].tolist()
current_r = 0

while len(remaining) > 0 and len(selected) < 50:
    best_addition = None
    best_new_r = current_r
    
    # Try adding each remaining feature
    for feat in remaining[:30]:  # Only try top 30 remaining (for speed)
        test_features = selected + [feat]
        r, _ = evaluate_features(df_all, test_features,
                                  RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42))
        if r > best_new_r:
            best_new_r = r
            best_addition = feat
    
    if best_addition is None or best_new_r <= current_r + 0.001:
        break
    
    selected.append(best_addition)
    remaining.remove(best_addition)
    current_r = best_new_r
    
    mod = get_modality(best_addition)
    print(f"  +{best_addition:<45} [{mod}] → r = {current_r:.3f}")

print(f"\nGreedy selected {len(selected)} features:")
greedy_mix = {}
for f in selected:
    mod = get_modality(f)
    greedy_mix[mod] = greedy_mix.get(mod, 0) + 1
print(f"  Modality mix: {greedy_mix}")

# Final evaluation with greedy features
r_greedy, per_sub = evaluate_features(df_all, selected,
                                       RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
print(f"  Final per-subject r = {r_greedy:.3f}")


# =============================================================================
print("\n" + "="*80)
print("SUMMARY: BEST GRANULAR FEATURE SELECTION")
print("="*80)

print(f"""
RESULTS COMPARISON:
───────────────────
All IMU features (60):       per-subject r ≈ 0.56
Lasso-selected ({len(lasso_selected):>3}):        per-subject r = {r_lasso:.3f}
Top {best_n} by importance:        per-subject r = {best_r:.3f}
Greedy selection ({len(selected):>2}):        per-subject r = {r_greedy:.3f}

OPTIMAL FEATURE MIX:
────────────────────
Top {best_n} features include: {dict(sorted(top30_modalities.items()) if best_n <= 30 else greedy_mix)}

KEY INSIGHT:
────────────
The granular selection shows which SPECIFIC features matter most.
With more data (subjects 6-20), this mix could change!
""")

# Save the best features
print("\nBest individual features (by RF importance):")
for i, row in importance_df.head(best_n).iterrows():
    mod = get_modality(row['feature'])
    print(f"  {row['importance']:.4f}  [{mod:>3}]  {row['feature']}")
