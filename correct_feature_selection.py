#!/usr/bin/env python3
"""
CORRECT FEATURE SELECTION: On Training Data Only
=================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FEATURE SELECTION COMPARISON")
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

exclude_cols = ['t_center', 'borg', 'subject', 'Unnamed', 'activity_label', 'source', 'fused']
all_features = [c for c in df_all.columns if not any(x in c for x in exclude_cols) 
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"Total features: {len(all_features)}")
print(f"Total windows: {len(df_all)}")

# =============================================================================
# METHOD 1: WRONG - Select features on ALL data (including test)
# =============================================================================

print("\n" + "="*80)
print("METHOD 1: WRONG - Select top features on ALL data (data leakage!)")
print("="*80)

# Select top 30 on ALL data
correlations_all = []
for col in all_features:
    valid = df_all[[col, 'borg']].dropna()
    if len(valid) > 100:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations_all.append((col, abs(r), r))
correlations_all.sort(key=lambda x: x[1], reverse=True)
top_30_all = [c[0] for c in correlations_all[:30]]

print(f"\nTop 10 features (selected on ALL data - WRONG):")
for feat, abs_r, r in correlations_all[:10]:
    print(f"  {feat}: r = {r:.3f}")

# =============================================================================
# METHOD 2: CORRECT - Select features on TRAINING data only
# =============================================================================

print("\n" + "="*80)
print("METHOD 2: CORRECT - Select top features on TRAINING data only")
print("="*80)

def select_features_on_train(df, features, test_subject, n_top=30):
    """Select top features using ONLY training subjects."""
    train_df = df[df['subject'] != test_subject]
    
    correlations = []
    for col in features:
        valid = train_df[[col, 'borg']].dropna()
        if len(valid) > 50:
            r, _ = pearsonr(valid[col], valid['borg'])
            correlations.append((col, abs(r), r))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in correlations[:n_top]], correlations[:n_top]

# Show which features are selected for each test subject
print("\nTop 5 features selected when training WITHOUT each subject:")
for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    top_feats, corrs = select_features_on_train(df_all, all_features, test_sub, n_top=5)
    print(f"\n{test_sub} held out:")
    for feat, abs_r, r in corrs:
        print(f"    {feat}: r = {r:.3f}")

# =============================================================================
# METHOD 3: Select features that are CONSISTENT across subjects
# =============================================================================

print("\n" + "="*80)
print("METHOD 3: Select features CONSISTENTLY correlated across ALL subjects")
print("="*80)

def get_per_subject_correlations(df, features):
    """Get correlation per subject for each feature."""
    results = {}
    subjects = df['subject'].unique()
    
    for feat in features:
        per_sub = {}
        for sub in subjects:
            sub_df = df[df['subject'] == sub][[feat, 'borg']].dropna()
            if len(sub_df) > 20:
                r, _ = pearsonr(sub_df[feat], sub_df['borg'])
                per_sub[sub] = r
        
        if len(per_sub) >= 4:  # At least 4 subjects have data
            results[feat] = per_sub
    
    return results

per_sub_corrs = get_per_subject_correlations(df_all, all_features)

# Find features with CONSISTENT direction across subjects
print("\nFeatures with CONSISTENT correlation direction (same sign for all subjects):")

consistent_features = []
for feat, corrs in per_sub_corrs.items():
    values = list(corrs.values())
    if len(values) >= 4:
        # Check if all same sign
        all_positive = all(v > 0 for v in values)
        all_negative = all(v < 0 for v in values)
        
        if all_positive or all_negative:
            avg_r = np.mean(values)
            min_r = min(values) if all_positive else max(values)
            consistent_features.append((feat, avg_r, min_r, values))

# Sort by minimum absolute correlation (most consistently strong)
consistent_features.sort(key=lambda x: min(abs(v) for v in x[3]), reverse=True)

print(f"\nTop 15 CONSISTENTLY correlated features (same direction for all subjects):")
print(f"{'Feature':<50} | {'Avg r':>7} | {'Min |r|':>7} | Per-subject r")
print("-" * 110)

for feat, avg_r, min_r, values in consistent_features[:15]:
    vals_str = ", ".join([f"{v:.2f}" for v in values])
    print(f"{feat:<50} | {avg_r:>7.3f} | {min(abs(v) for v in values):>7.3f} | {vals_str}")

# =============================================================================
# COMPARE METHODS
# =============================================================================

print("\n" + "="*80)
print("COMPARISON: Train on 4, Test on 1")
print("="*80)

def train_test_with_features(df, features, test_subject):
    """Train on 4, test on 1 with given features."""
    train_df = df[df['subject'] != test_subject]
    test_df = df[df['subject'] == test_subject]
    
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    if len(valid_features) == 0:
        return None
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[valid_features]))
    X_test = scaler.transform(imputer.transform(test_df[valid_features]))
    y_train = train_df['borg'].values
    y_test = test_df['borg'].values
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r, _ = pearsonr(preds, y_test)
    mae = np.mean(np.abs(preds - y_test))
    
    return {'r': r, 'mae': mae, 'n_features': len(valid_features)}

# Get consistent features
consistent_30 = [f[0] for f in consistent_features[:30]]

print("\n--- RESULTS: Train on 4, Test on 1 (NO CALIBRATION) ---\n")

methods = {
    'WRONG: Top30 on ALL data': top_30_all,
    'CORRECT: Top30 on TRAIN only': None,  # Will select per fold
    'CONSISTENT: Same sign all subjects': consistent_30,
    'ALL features': all_features,
}

print(f"{'Method':<35} | {'Test Sub':<10} | {'r':>7} | {'MAE':>6}")
print("-" * 70)

results_by_method = {m: [] for m in methods.keys()}

for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    # Method: CORRECT - select on train only
    top_30_train, _ = select_features_on_train(df_all, all_features, test_sub, n_top=30)
    
    for method_name, features in methods.items():
        if method_name == 'CORRECT: Top30 on TRAIN only':
            features = top_30_train
        
        result = train_test_with_features(df_all, features, test_sub)
        if result:
            print(f"{method_name:<35} | {test_sub:<10} | {result['r']:>7.3f} | {result['mae']:>6.2f}")
            results_by_method[method_name].append(result)
    print()

# Average results
print("\n--- AVERAGE RESULTS ---\n")
print(f"{'Method':<35} | {'Avg r':>8} | {'Avg MAE':>8}")
print("-" * 60)

for method_name, results in results_by_method.items():
    if results:
        avg_r = np.mean([r['r'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        print(f"{method_name:<35} | {avg_r:>8.3f} | {avg_mae:>8.2f}")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
FEATURE SELECTION METHODS:

1. WRONG: Select on ALL data
   - This is DATA LEAKAGE - you're peeking at the test subject
   - Features that correlate well in the test subject will be selected
   - Leads to overly optimistic results

2. CORRECT: Select on TRAINING data only
   - Each fold, select features using ONLY the 4 training subjects
   - This is honest cross-validation
   - But features may vary between folds

3. BEST: Select CONSISTENTLY correlated features
   - Find features that correlate in the SAME DIRECTION for all subjects
   - These are most likely to generalize to a new person
   - Example: If HR correlates positively with Borg for ALL 5 subjects,
     it's likely to work for a 6th subject too

THE INSIGHT:
- Many features have HIGH pooled correlation but INCONSISTENT per-subject
- Example: ppg_green_signal_rms has r=-0.44 pooled, but ranges from -0.13 to -0.49 per subject
- Features with CONSISTENT correlation (even if weaker) may generalize better
""")

# Show the best consistent features
print("\n--- RECOMMENDED FEATURES (consistent across all subjects) ---\n")
for feat, avg_r, min_r, values in consistent_features[:20]:
    modality = 'PPG' if 'ppg' in feat.lower() else ('EDA' if 'eda' in feat.lower() else 'IMU')
    direction = "+" if avg_r > 0 else "-"
    print(f"  {direction} {feat} ({modality}): avg r = {avg_r:.3f}")
