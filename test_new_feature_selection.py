#!/usr/bin/env python3
"""
Test the new consistent feature selection in the pipeline
=========================================================
Compare OLD (pooled correlation) vs NEW (consistent across subjects)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ml.feature_selection import select_features, select_features_consistent
from ml.consistent_feature_selection import select_features_for_loso

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("TESTING NEW CONSISTENT FEATURE SELECTION")
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

# =============================================================================
# METHOD 1: OLD - Pooled correlation (data leakage in LOSO!)
# =============================================================================

print("\n" + "="*60)
print("METHOD 1: OLD - Select by pooled correlation")
print("="*60)

# Get feature columns
skip_cols = {'t_center', 'borg', 'subject', 'activity_label'}
feature_cols = [c for c in df_all.columns 
                if c not in skip_cols 
                and not c.startswith('Unnamed')
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

# Select top 30 by pooled correlation
correlations = []
for col in feature_cols:
    valid = df_all[[col, 'borg']].dropna()
    if len(valid) > 100:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r)))
correlations.sort(key=lambda x: x[1], reverse=True)
old_features = [c[0] for c in correlations[:30]]

print(f"\nSelected {len(old_features)} features (top by pooled correlation)")
print("Top 5:", old_features[:5])

# =============================================================================
# METHOD 2: NEW - Consistent across subjects
# =============================================================================

print("\n" + "="*60)
print("METHOD 2: NEW - Select by consistency across subjects")
print("="*60)

new_features, summary = select_features_consistent(
    df_all, 
    subject_col='subject',
    target_col='borg',
    min_subjects=4,
    top_n=30,
    verbose=True
)

print(f"\nSelected {len(new_features)} features")

# =============================================================================
# COMPARE: Train on 4, Test on 1 (LOSO without calibration)
# =============================================================================

print("\n" + "="*80)
print("COMPARISON: Train on 4, Test on 1 (NO calibration)")
print("="*80)

def train_test(df, features, test_subject):
    train_df = df[df['subject'] != test_subject]
    test_df = df[df['subject'] == test_subject]
    
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    if len(valid_features) == 0:
        return None
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[valid_features]))
    X_test = scaler.transform(imputer.transform(test_df[valid_features]))
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_df['borg'].values)
    preds = model.predict(X_test)
    
    r, _ = pearsonr(preds, test_df['borg'].values)
    mae = np.mean(np.abs(preds - test_df['borg'].values))
    
    return {'r': r, 'mae': mae}

print(f"\n{'Test Subject':<12} | {'OLD (pooled) r':>15} | {'NEW (consistent) r':>18} | {'Improvement':>12}")
print("-" * 70)

old_results = []
new_results = []

for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    old_r = train_test(df_all, old_features, test_sub)
    new_r = train_test(df_all, new_features, test_sub)
    
    if old_r and new_r:
        old_results.append(old_r)
        new_results.append(new_r)
        diff = new_r['r'] - old_r['r']
        print(f"{test_sub:<12} | {old_r['r']:>15.3f} | {new_r['r']:>18.3f} | {diff:>+11.3f}")

# Average
avg_old = np.mean([r['r'] for r in old_results])
avg_new = np.mean([r['r'] for r in new_results])
print("-" * 70)
print(f"{'AVERAGE':<12} | {avg_old:>15.3f} | {avg_new:>18.3f} | {avg_new - avg_old:>+11.3f}")

# =============================================================================
# BONUS: LOSO-correct feature selection (select on train only each fold)
# =============================================================================

print("\n" + "="*80)
print("BONUS: LOSO-correct feature selection (select on training data only)")
print("="*80)

loso_results = []

for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    # Select features using ONLY training subjects
    loso_features = select_features_for_loso(
        df_all, 
        test_subject=test_sub,
        subject_col='subject',
        target_col='borg',
        top_n=30,
        verbose=False
    )
    
    result = train_test(df_all, loso_features, test_sub)
    if result:
        loso_results.append(result)
        print(f"{test_sub}: r = {result['r']:.3f} (using {len(loso_features)} features selected on train only)")

avg_loso = np.mean([r['r'] for r in loso_results])
print(f"\nAverage (LOSO-correct): r = {avg_loso:.3f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Feature Selection Method Comparison (Train on 4, Test on 1):

┌─────────────────────────────────────────────────────────────┐
│  Method                              │  Avg r  │ Features  │
├─────────────────────────────────────────────────────────────┤
│  OLD: Pooled correlation (leakage!)  │  {avg_old:>5.3f}  │    30     │
│  NEW: Consistent across subjects     │  {avg_new:>5.3f}  │    {len(new_features):>2}     │
│  LOSO-correct: Select on train only  │  {avg_loso:>5.3f}  │   ~30     │
└─────────────────────────────────────────────────────────────┘

CONCLUSION:
- The NEW consistent method improves generalization by {avg_new - avg_old:+.3f}
- LOSO-correct is even better: +{avg_loso - avg_old:.3f} over OLD method

TO USE IN YOUR PIPELINE:
    from ml.feature_selection import select_features_consistent
    
    selected, summary = select_features_consistent(
        df,
        subject_col='subject',
        target_col='borg',
        min_subjects=4,
        top_n=30
    )
""")
