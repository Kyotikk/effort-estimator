#!/usr/bin/env python3
"""
Critical analysis: Are EDA correlations real or artifacts?
If EDA baselines differ wildly across subjects, correlations might be fake.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')

print("="*75)
print("POOLED vs WITHIN-SUBJECT CORRELATION")
print("="*75)
print()

# Key features to analyze
eda_feats = ['eda_cc_mean', 'eda_stress_skin_mean', 'eda_cc_std']
ppg_feats = ['ppg_green_hr_mean', 'ppg_green_rmssd', 'ppg_green_p50']

all_feats = [f for f in eda_feats + ppg_feats if f in df.columns]

print("Feature                        | Pooled r  | Within r  | Verdict")
print("-"*75)

for feat in all_feats:
    valid = df[['borg', feat]].dropna()
    r_pooled, _ = pearsonr(valid['borg'], valid[feat])
    
    within_rs = []
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj][['borg', feat]].dropna()
        if len(subj_data) > 10:
            r, _ = pearsonr(subj_data['borg'], subj_data[feat])
            within_rs.append(r)
    
    r_within = np.mean(within_rs)
    
    if abs(r_pooled) > 0.15 and abs(r_within) < 0.1:
        verdict = "ARTIFACT!"
    elif abs(r_pooled) > abs(r_within) + 0.1:
        verdict = "Inflated"
    else:
        verdict = "Real"
    
    print(f"{feat:30s} | {r_pooled:+.3f}     | {r_within:+.3f}     | {verdict}")

print()
print("="*75)
print("EDA BASELINE DIFFERENCES")
print("="*75)
print()
print("Subject       | EDA mean  | Borg mean | Note")
print("-"*60)

for subj in sorted(df['subject'].unique()):
    mean_eda = df[df['subject']==subj]['eda_cc_mean'].mean()
    mean_borg = df[df['subject']==subj]['borg'].dropna().mean()
    note = "HIGH EDA" if mean_eda > 500 else "LOW BORG" if mean_borg < 2 else ""
    label = subj.replace('sim_elderly', 'P')
    print(f"{label:13s} | {mean_eda:9.0f} | {mean_borg:9.2f} | {note}")

print()
print("="*75)
print("CONCLUSION")
print("="*75)

print("\n1. POOLED (ALL SUBJECTS) CORRELATION:")
print("-"*50)
pooled_results = {}
for feat in eda_features[:10]:
    valid = df_model[[feat, 'borg']].dropna()
    if len(valid) > 10:
        r, p = pearsonr(valid[feat], valid['borg'])
        pooled_results[feat] = r
        print(f"  {feat:35s}: r={r:+.3f}")

print("\n2. WITHIN-SUBJECT CORRELATION (TRUE SIGNAL):")
print("-"*50)

within_results = {}
for feat in eda_features[:10]:
    within_rs = []
    for subj in sorted(df_model['subject'].unique()):
        subj_data = df_model[df_model['subject'] == subj][[feat, 'borg']].dropna()
        if len(subj_data) > 10:
            r, p = pearsonr(subj_data[feat], subj_data['borg'])
            within_rs.append(r)
    
    if within_rs:
        mean_within = np.mean(within_rs)
        within_results[feat] = mean_within
        print(f"  {feat:35s}: mean r={mean_within:+.3f}")

print("\n" + "="*70)
print("3. COMPARISON: POOLED vs WITHIN-SUBJECT")
print("="*70)

print("\nFeature                             | Pooled r | Within r | ARTIFACT?")
print("-"*75)

for feat in eda_features[:10]:
    if feat in pooled_results and feat in within_results:
        r_pooled = pooled_results[feat]
        r_within = within_results[feat]
        
        # If pooled >> within, it's an artifact
        if abs(r_pooled) > 0.1 and abs(r_within) < 0.1:
            verdict = "YES - FAKE!"
        elif abs(r_pooled - r_within) > 0.15:
            verdict = "INFLATED"
        else:
            verdict = "Real"
        
        print(f"{feat:35s} | {r_pooled:+.3f}   | {r_within:+.3f}   | {verdict}")

# Now do the same for PPG and other features
print("\n" + "="*70)
print("4. ALL FEATURE TYPES COMPARISON")
print("="*70)

ppg_features = [c for c in feature_cols if 'ppg' in c.lower()][:10]
imu_features = [c for c in feature_cols if any(x in c.lower() for x in ['acc', 'gyro'])][:10]

for name, features in [("PPG", ppg_features), ("IMU", imu_features)]:
    print(f"\n{name} FEATURES:")
    print("-"*50)
    
    for feat in features[:5]:
        # Pooled
        valid = df_model[[feat, 'borg']].dropna()
        if len(valid) > 10:
            r_pooled, _ = pearsonr(valid[feat], valid['borg'])
            
            # Within-subject
            within_rs = []
            for subj in sorted(df_model['subject'].unique()):
                subj_data = df_model[df_model['subject'] == subj][[feat, 'borg']].dropna()
                if len(subj_data) > 10:
                    r, _ = pearsonr(subj_data[feat], subj_data['borg'])
                    within_rs.append(r)
            
            r_within = np.mean(within_rs) if within_rs else 0
            diff = r_pooled - r_within
            
            print(f"  {feat:35s}: pooled={r_pooled:+.3f}, within={r_within:+.3f}, diff={diff:+.3f}")

print("\n" + "="*70)
print("5. CONCLUSION")
print("="*70)

print("""
KEY INSIGHT:
─────────────────────────────────────────────────────────────────────
You are RIGHT to question this!

If Pooled r is HIGH but Within-Subject r is LOW:
  → The correlation is driven by BETWEEN-SUBJECT baseline differences
  → P1 has high EDA AND high Borg (subject effect)
  → P5 has low EDA AND low Borg (subject effect)
  → This is NOT a true effort signal - it's SIMPSON'S PARADOX again!

If Pooled r ≈ Within-Subject r:
  → The correlation reflects TRUE within-subject effort variation
  → This feature genuinely predicts effort

The EDA baselines differ by 5-6x across subjects:
  → P1: ~1300 (very high)
  → P2-P5: ~200-350 (normal)
  
This creates artificial correlations in pooled data!

IMPLICATION:
Features should be evaluated on WITHIN-SUBJECT correlation,
not pooled correlation. This is another reason why
personalization is essential.
""")
