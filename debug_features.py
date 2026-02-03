#!/usr/bin/env python3
"""
Debug: What features are being used and why MI fails
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')
df = df.dropna(subset=['borg']).reset_index(drop=True)
y = df['borg'].values
print(f"Data: {len(df)} windows, Borg range [{y.min():.1f}, {y.max():.1f}]")

# Features
meta_cols = ['t_center', 'borg', 'activity', 'activity_id', 'subject_id', 'valid', 'n_samples', 'win_sec', 'modality']
X = df[[c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]].copy()
X = X.loc[:, X.nunique() > 1]
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

print("="*70)
print("WHAT FEATURES ARE BEING USED?")
print("="*70)

# What MI selects
print("\n--- MI SELECTION (top 15) picks these features: ---")
mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)
mi_df = pd.DataFrame({'feature': X.columns, 'mi': mi_scores}).sort_values('mi', ascending=False)
for _, row in mi_df.head(15).iterrows():
    print(f"  {row['mi']:.4f}  {row['feature']}")

# Physiological features
hr_features = [c for c in X.columns if 'hr_mean' in c or 'hr_max' in c]
rmssd_features = [c for c in X.columns if 'rmssd' in c]
ibi_features = [c for c in X.columns if 'ibi' in c]

print(f"\n--- PHYSIOLOGICAL FEATURES (Option B uses these): ---")
print(f"HR features ({len(hr_features)}): {hr_features}")
print(f"RMSSD features ({len(rmssd_features)}): {rmssd_features}")
print(f"IBI features ({len(ibi_features)}): {ibi_features}")

# Compare correlations
print("\n--- RAW CORRELATIONS WITH BORG ---")
print("\nEDA features:")
for col in ['eda_scl_mean', 'eda_cc_mean', 'eda_scl_auc']:
    if col in X.columns:
        r, p = pearsonr(X[col], y)
        print(f"  {col}: r={r:.3f} {'***' if p<0.001 else ''}")

print("\nHR features:")
for col in ['ppg_green_hr_mean', 'ppg_green_hr_max', 'ppg_infra_hr_mean']:
    if col in X.columns:
        r, p = pearsonr(X[col], y)
        print(f"  {col}: r={r:.3f} {'***' if p<0.001 else ''}")
        
print("\nRMSSD features:")
for col in rmssd_features:
    if col in X.columns:
        r, p = pearsonr(X[col], y)
        print(f"  {col}: r={r:.3f} {'***' if p<0.001 else ''}")

print("\n--- WHY MI SELECTS EDA BUT IT FAILS IN CV ---")
print("""
MI measures dependency WITHIN the training set.
EDA (skin conductance) changes slowly over time → high autocorrelation.
Adjacent windows have similar EDA values AND similar Borg (same activity).
So MI sees: "EDA predicts Borg!" but it's really just time proximity.

When you do GroupKFold by activity:
- Train on activities 1-20, test on activities 21-26
- EDA values in test activities are DIFFERENT from training
- Model fails because EDA doesn't generalize

HR features work better because:
- HR responds quickly to effort (physiological relationship)
- HR during high-effort activities is high regardless of WHICH activity
""")

# Show this explicitly
print("\n--- PROOF: EDA is correlated with TIME, not effort ---")
from scipy.stats import spearmanr

t_center = df['t_center'].values
for col in ['eda_scl_mean', 'ppg_green_hr_mean']:
    if col in X.columns:
        r_borg, _ = pearsonr(X[col], y)
        r_time, _ = pearsonr(X[col], t_center)
        print(f"  {col}:")
        print(f"    Correlation with Borg: r={r_borg:.3f}")
        print(f"    Correlation with TIME: r={r_time:.3f}")
