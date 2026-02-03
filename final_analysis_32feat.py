#!/usr/bin/env python3
"""
Final analysis using the automatically selected 32 features from feature_selection_and_qc.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Load the filtered dataset (with 32 selected features)
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/qc_5.0s/features_filtered_5.0s.csv')
print(f'Loaded: {len(df)} samples, {len(df.columns)} columns')
print(f'Subjects: {sorted(df["subject"].unique())}')

# Get feature columns
feat_cols = [c for c in df.columns if c not in ['subject', 'borg', 't_center', 't_start', 't_end']]
print(f'Features: {len(feat_cols)}')
print(f'Feature list: {feat_cols}')

# Category mapping
def to_cat(borg):
    if borg <= 2: return 0  # LOW
    elif borg <= 4: return 1  # MOD
    else: return 2  # HIGH

X = df[feat_cols].values
y = df['borg'].values
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Method 1: Raw Features (no calibration) - LOSO
print("\n" + "="*60)
print("METHOD 1: Raw Features (LOSO, no calibration)")
print("="*60)
method1_preds = np.full(len(df), np.nan)
for subj in df['subject'].unique():
    test_mask = df['subject'] == subj
    train_mask = ~test_mask
    model = Ridge(alpha=1.0).fit(X_scaled[train_mask], y[train_mask])
    method1_preds[test_mask] = model.predict(X_scaled[test_mask])

# Method 3: Calibration (mean shift only)
print("\n" + "="*60)
print("METHOD 3: LOSO + Subject Mean Calibration")
print("="*60)
method3_preds = method1_preds.copy()
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    bias = np.mean(y[mask]) - np.mean(method1_preds[mask])
    method3_preds[mask] += bias

# Method 4: Within-subject train (20% train, 80% test)
print("\n" + "="*60)
print("METHOD 4: Within-Subject (20% train, 80% test)")
print("="*60)
np.random.seed(42)
method4_preds = np.full(len(df), np.nan)
for subj in df['subject'].unique():
    mask = df['subject'] == subj
    subj_idx = np.where(mask)[0]
    np.random.shuffle(subj_idx)
    train_idx = subj_idx[:int(len(subj_idx)*0.2)]
    test_idx = subj_idx[int(len(subj_idx)*0.2):]
    
    # Fit on subject's training data
    scaler_s = StandardScaler().fit(X[train_idx])
    X_train = scaler_s.transform(X[train_idx])
    X_test = scaler_s.transform(X[test_idx])
    
    model = Ridge(alpha=1.0).fit(X_train, y[train_idx])
    method4_preds[test_idx] = model.predict(X_test)
    method4_preds[train_idx] = y[train_idx]  # Training samples = perfect

# Results table
print("\n" + "="*60)
print("RESULTS SUMMARY (32 Features, Auto-Selected)")
print("="*60)

print("\n{:<30} {:>8} {:>8} {:>10} {:>10} {:>12}".format(
    "Method", "r", "MAE", "Exact%", "±1 Cat%", "Danger%"))
print("-"*80)

for name, preds in [('Method 1 (Raw)', method1_preds), 
                     ('Method 3 (Calibration)', method3_preds),
                     ('Method 4 (Within-Subject)', method4_preds)]:
    valid = ~np.isnan(preds)
    y_true = y[valid]
    y_pred = preds[valid]
    
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Category accuracy
    cat_true = np.array([to_cat(b) for b in y_true])
    cat_pred = np.array([to_cat(b) for b in np.round(y_pred)])
    
    exact = np.mean(cat_true == cat_pred) * 100
    within1 = np.mean(np.abs(cat_true - cat_pred) <= 1) * 100
    dangerous = np.mean(np.abs(cat_true - cat_pred) >= 2) * 100
    
    print("{:<30} {:>8.2f} {:>8.2f} {:>10.1f} {:>10.1f} {:>12.1f}".format(
        name, r, mae, exact, within1, dangerous))

print("\nCategory definitions: LOW (0-2), MOD (3-4), HIGH (5+)")
