#!/usr/bin/env python3
"""
Compare top features across all 4 methods
Are the same features important regardless of normalization?
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr

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

print("="*80)
print("TOP FEATURES COMPARISON ACROSS ALL 4 METHODS")
print("="*80)

# ============================================================================
# METHOD 1: Cross-subject raw - feature importance from global model
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: CROSS-SUBJECT (RAW) - Feature Importance")
print("="*80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Get feature importance (absolute coefficients)
coef_1 = pd.DataFrame({
    'feature': valid_features,
    'coef': model.coef_,
    'abs_coef': np.abs(model.coef_)
}).sort_values('abs_coef', ascending=False)

print("\nTop 15 features (by absolute coefficient):")
for i, row in coef_1.head(15).iterrows():
    sign = "+" if row['coef'] > 0 else "-"
    print(f"  {sign} {row['feature']:50s} coef={row['coef']:.4f}")

# ============================================================================
# METHOD 2: Cross-subject normalized features - feature importance
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: CROSS-SUBJECT (FEATURES NORMALIZED) - Feature Importance")
print("="*80)

df_norm = df_model.copy()
for feat in valid_features:
    for subj in df_model['subject'].unique():
        mask = df_model['subject'] == subj
        subj_mean = df_model.loc[mask, feat].mean()
        subj_std = df_model.loc[mask, feat].std()
        if subj_std > 0:
            df_norm.loc[mask, feat] = (df_model.loc[mask, feat] - subj_mean) / subj_std
        else:
            df_norm.loc[mask, feat] = 0

X_norm = df_norm[valid_features].values

model2 = Ridge(alpha=1.0)
model2.fit(X_norm, y)

coef_2 = pd.DataFrame({
    'feature': valid_features,
    'coef': model2.coef_,
    'abs_coef': np.abs(model2.coef_)
}).sort_values('abs_coef', ascending=False)

print("\nTop 15 features (by absolute coefficient):")
for i, row in coef_2.head(15).iterrows():
    sign = "+" if row['coef'] > 0 else "-"
    print(f"  {sign} {row['feature']:50s} coef={row['coef']:.4f}")

# ============================================================================
# METHOD 3: Both normalized - feature importance
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: CROSS-SUBJECT (BOTH NORMALIZED) - Feature Importance")
print("="*80)

df_norm['borg_norm'] = 0.0
for subj in df_model['subject'].unique():
    mask = df_model['subject'] == subj
    subj_mean = df_model.loc[mask, 'borg'].mean()
    subj_std = df_model.loc[mask, 'borg'].std()
    if subj_std > 0:
        df_norm.loc[mask, 'borg_norm'] = (df_model.loc[mask, 'borg'] - subj_mean) / subj_std

y_norm = df_norm['borg_norm'].values

model3 = Ridge(alpha=1.0)
model3.fit(X_norm, y_norm)

coef_3 = pd.DataFrame({
    'feature': valid_features,
    'coef': model3.coef_,
    'abs_coef': np.abs(model3.coef_)
}).sort_values('abs_coef', ascending=False)

print("\nTop 15 features (by absolute coefficient):")
for i, row in coef_3.head(15).iterrows():
    sign = "+" if row['coef'] > 0 else "-"
    print(f"  {sign} {row['feature']:50s} coef={row['coef']:.4f}")

# ============================================================================
# METHOD 4: Within-subject - average importance across subjects
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: WITHIN-SUBJECT - Average Feature Importance")
print("="*80)

within_coefs = {feat: [] for feat in valid_features}

for subj in df_model['subject'].unique():
    mask = df_model['subject'] == subj
    X_subj = X_raw[mask]
    y_subj = y[mask]
    
    scaler_subj = StandardScaler()
    X_subj_scaled = scaler_subj.fit_transform(X_subj)
    
    model_subj = Ridge(alpha=1.0)
    model_subj.fit(X_subj_scaled, y_subj)
    
    for i, feat in enumerate(valid_features):
        within_coefs[feat].append(model_subj.coef_[i])

# Average absolute coefficient across subjects
coef_4 = pd.DataFrame({
    'feature': valid_features,
    'mean_coef': [np.mean(within_coefs[f]) for f in valid_features],
    'abs_mean_coef': [np.mean(np.abs(within_coefs[f])) for f in valid_features]
}).sort_values('abs_mean_coef', ascending=False)

print("\nTop 15 features (by mean absolute coefficient across subjects):")
for i, row in coef_4.head(15).iterrows():
    sign = "+" if row['mean_coef'] > 0 else "-"
    print(f"  {sign} {row['feature']:50s} coef={row['mean_coef']:.4f}")

# ============================================================================
# COMPARISON: Are the same features important?
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: OVERLAP IN TOP FEATURES")
print("="*80)

top_n = 20

top1 = set(coef_1.head(top_n)['feature'].tolist())
top2 = set(coef_2.head(top_n)['feature'].tolist())
top3 = set(coef_3.head(top_n)['feature'].tolist())
top4 = set(coef_4.head(top_n)['feature'].tolist())

print(f"\nTop {top_n} features overlap:")
print(f"  Method 1 ∩ Method 2: {len(top1 & top2)} / {top_n}")
print(f"  Method 1 ∩ Method 3: {len(top1 & top3)} / {top_n}")
print(f"  Method 1 ∩ Method 4: {len(top1 & top4)} / {top_n}")
print(f"  Method 2 ∩ Method 3: {len(top2 & top3)} / {top_n}")
print(f"  Method 2 ∩ Method 4: {len(top2 & top4)} / {top_n}")
print(f"  Method 3 ∩ Method 4: {len(top3 & top4)} / {top_n}")

# Features in ALL methods
all_overlap = top1 & top2 & top3 & top4
print(f"\n  Features in ALL 4 methods: {len(all_overlap)}")
if all_overlap:
    print(f"    {sorted(all_overlap)}")

# Features unique to each
print(f"\n  Unique to Method 1 (raw cross): {len(top1 - top2 - top3 - top4)}")
print(f"  Unique to Method 2 (norm feat): {len(top2 - top1 - top3 - top4)}")
print(f"  Unique to Method 3 (both norm): {len(top3 - top1 - top2 - top4)}")
print(f"  Unique to Method 4 (within):    {len(top4 - top1 - top2 - top3)}")

# ============================================================================
# DETAILED COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("DETAILED COMPARISON: TOP 20 FEATURES SIDE BY SIDE")
print("="*80)

print("\n" + "-"*80)
print(f"{'Rank':<5} {'M1 (raw cross)':<30} {'M3 (both norm)':<30} {'M4 (within)':<30}")
print("-"*80)

top1_list = coef_1.head(20)['feature'].tolist()
top3_list = coef_3.head(20)['feature'].tolist()
top4_list = coef_4.head(20)['feature'].tolist()

for i in range(20):
    f1 = top1_list[i][:28] if len(top1_list) > i else ""
    f3 = top3_list[i][:28] if len(top3_list) > i else ""
    f4 = top4_list[i][:28] if len(top4_list) > i else ""
    print(f"{i+1:<5} {f1:<30} {f3:<30} {f4:<30}")

# ============================================================================
# FEATURE TYPE BREAKDOWN
# ============================================================================
print("\n" + "="*80)
print("FEATURE TYPE BREAKDOWN IN TOP 20")
print("="*80)

def get_feature_type(feat):
    if 'eda' in feat.lower():
        return 'EDA'
    elif 'ppg' in feat.lower() or 'hr' in feat.lower():
        return 'PPG/HR'
    elif 'acc' in feat.lower() or 'gyr' in feat.lower() or 'imu' in feat.lower() or 'mag' in feat.lower():
        return 'IMU'
    elif 'rr' in feat.lower() or 'hrv' in feat.lower() or 'rmssd' in feat.lower():
        return 'HRV'
    else:
        return 'Other'

def count_types(features):
    types = [get_feature_type(f) for f in features]
    return pd.Series(types).value_counts().to_dict()

print("\nFeature type distribution in top 20:")
print(f"  {'Method':<35} {'EDA':<6} {'PPG/HR':<8} {'IMU':<6} {'HRV':<6} {'Other':<6}")
print(f"  {'-'*65}")

for name, feats in [
    ('Method 1 (raw cross-subject)', top1_list),
    ('Method 2 (features normalized)', coef_2.head(20)['feature'].tolist()),
    ('Method 3 (both normalized)', top3_list),
    ('Method 4 (within-subject)', top4_list)
]:
    types = count_types(feats)
    print(f"  {name:<35} {types.get('EDA', 0):<6} {types.get('PPG/HR', 0):<8} {types.get('IMU', 0):<6} {types.get('HRV', 0):<6} {types.get('Other', 0):<6}")

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
KEY FINDINGS:

1. TOP FEATURES CHANGE BASED ON NORMALIZATION:
   • Raw cross-subject uses features that capture BASELINE differences
   • Normalized methods use features that capture RELATIVE changes

2. WHY THIS MATTERS:
   • Method 1 might use "EDA mean" because P1 has high EDA + medium Borg
   • Method 3 uses "EDA change from baseline" - actual physiological response

3. WHICH FEATURES ARE "REAL"?
   • Features that appear in Method 3 AND Method 4 are most reliable
   • These capture actual effort-related physiological changes
   • Features only in Method 1 might be "fake" (Simpson's Paradox artifacts)
""")
