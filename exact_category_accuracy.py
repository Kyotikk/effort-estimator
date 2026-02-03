#!/usr/bin/env python3
"""Exact category accuracy for all 4 methods"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, confusion_matrix

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')

exclude_cols = ['subject', 'borg', 't_center', 'window_start', 'window_end', 'unix_time', 'Unnamed: 0', 'index']
feature_cols = [c for c in df.columns if c not in exclude_cols 
                and df[c].dtype in ['float64', 'int64']
                and df[c].notna().sum() > 100]
valid_features = [c for c in feature_cols if df[c].isna().mean() < 0.5]

df_model = df.dropna(subset=['borg'])[['subject', 'borg'] + valid_features].dropna()

X_raw = df_model[valid_features].values
y = df_model['borg'].values
groups = df_model['subject'].values

def to_cat(b):
    if b <= 2: return 0
    elif b <= 4: return 1
    else: return 2

y_cat = np.array([to_cat(b) for b in y])

print("="*70)
print("EXACT CATEGORY ACCURACY: LOW / MODERATE / HIGH")
print("="*70)
print("\nCategories: LOW (Borg 0-2), MODERATE (Borg 3-4), HIGH (Borg 5+)\n")

logo = LeaveOneGroupOut()
model = Ridge(alpha=1.0)

# METHOD 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y_pred_1 = cross_val_predict(model, X_scaled, y, cv=logo, groups=groups)
y_pred_1_cat = np.array([to_cat(b) for b in y_pred_1])

# METHOD 2
df_norm = df_model.copy()
for feat in valid_features:
    for subj in df_model['subject'].unique():
        mask = df_model['subject'] == subj
        m, s = df_model.loc[mask, feat].mean(), df_model.loc[mask, feat].std()
        df_norm.loc[mask, feat] = (df_model.loc[mask, feat] - m) / s if s > 0 else 0
X_norm = df_norm[valid_features].values
y_pred_2 = cross_val_predict(model, X_norm, y, cv=logo, groups=groups)
y_pred_2_cat = np.array([to_cat(b) for b in y_pred_2])

# METHOD 3
borg_stats = {}
df_norm['borg_norm'] = 0.0
for subj in df_model['subject'].unique():
    mask = df_model['subject'] == subj
    m, s = df_model.loc[mask, 'borg'].mean(), df_model.loc[mask, 'borg'].std()
    borg_stats[subj] = {'mean': m, 'std': s}
    df_norm.loc[mask, 'borg_norm'] = (df_model.loc[mask, 'borg'] - m) / s if s > 0 else 0
y_norm = df_norm['borg_norm'].values
y_pred_3_norm = cross_val_predict(model, X_norm, y_norm, cv=logo, groups=groups)
y_pred_3 = np.zeros_like(y_pred_3_norm)
for subj in df_model['subject'].unique():
    mask = groups == subj
    y_pred_3[mask] = y_pred_3_norm[mask] * borg_stats[subj]['std'] + borg_stats[subj]['mean']
y_pred_3_cat = np.array([to_cat(b) for b in y_pred_3])

# METHOD 4
y_pred_4 = np.zeros_like(y)
for subj in df_model['subject'].unique():
    mask = groups == subj
    X_subj, y_subj = X_raw[mask], y[mask]
    scaler_subj = StandardScaler()
    X_subj_scaled = scaler_subj.fit_transform(X_subj)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_4[mask] = cross_val_predict(model, X_subj_scaled, y_subj, cv=kf)
y_pred_4_cat = np.array([to_cat(b) for b in y_pred_4])

# Results
results = [
    ("1. Cross-subject (raw)", y_pred_1, y_pred_1_cat),
    ("2. Features normalized", y_pred_2, y_pred_2_cat),
    ("3. WITH CALIBRATION", y_pred_3, y_pred_3_cat),
    ("4. Within-subject", y_pred_4, y_pred_4_cat),
]

for name, y_pred, y_pred_cat in results:
    r, _ = pearsonr(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    exact = (y_cat == y_pred_cat).mean()
    big_miss = (np.abs(y_cat - y_pred_cat) == 2).mean()
    cm = confusion_matrix(y_cat, y_pred_cat)
    
    print(f"\n{name}")
    print(f"  r = {r:.2f}, MAE = {mae:.2f}")
    print(f"  EXACT CATEGORY: {exact:.0%}")
    print(f"  DANGEROUS (LOW<->HIGH): {big_miss:.0%}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"            LOW   MOD  HIGH")
    print(f"  Actual LOW  {cm[0,0]:3d}   {cm[0,1]:3d}   {cm[0,2]:3d}   -> {cm[0,0]/cm[0,:].sum()*100:.0f}% correct")
    print(f"  Actual MOD  {cm[1,0]:3d}   {cm[1,1]:3d}   {cm[1,2]:3d}   -> {cm[1,1]/cm[1,:].sum()*100:.0f}% correct")
    print(f"  Actual HIGH {cm[2,0]:3d}   {cm[2,1]:3d}   {cm[2,2]:3d}   -> {cm[2,2]/cm[2,:].sum()*100:.0f}% correct")
    print("-"*70)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nMethod                        | r    | MAE  | Exact | Dangerous")
print("------------------------------|------|------|-------|----------")
for name, y_pred, y_pred_cat in results:
    r, _ = pearsonr(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    exact = (y_cat == y_pred_cat).mean()
    big_miss = (np.abs(y_cat - y_pred_cat) == 2).mean()
    print(f"{name:29s} | {r:.2f} | {mae:.2f} | {exact:5.0%} | {big_miss:5.0%}")

print("\n" + "="*70)
print("WHAT 'FULL DATASET' MEANS FOR METHOD 4:")
print("="*70)
print("""
Method 4 uses 5-FOLD CV WITHIN each subject:

  For P1 (114 samples):
    Fold 1: Train on 91 samples, test on 23
    Fold 2: Train on 91 samples, test on 23
    ... (5 times)
    
  Each subject needs ~100 samples to train their model.
  
  In PRACTICE, this means:
    - User must do activities for ~10-15 minutes
    - Provide ~100 Borg ratings during that time
    - THEN the model can predict for them
    
  vs METHOD 3 (calibration):
    - User does ~8 minutes calibration (~20 samples)
    - Uses PRE-TRAINED model from other people
    - Only needs calibration data, not training data!
    
  That's why Method 3 is more PRACTICAL even though Method 4 is slightly better.
""")
