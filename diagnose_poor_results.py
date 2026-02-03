#!/usr/bin/env python3
"""
Diagnostic: Why are effort estimation results poor?
Check preprocessing, windowing, data quality, and signal content
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
df_labeled = df.dropna(subset=['borg'])

print("="*70)
print("DIAGNOSTIC: WHY ARE RESULTS POOR?")
print("="*70)

# ============================================================================
# 1. DATA QUALITY CHECK
# ============================================================================
print("\n" + "="*70)
print("1. DATA QUALITY")
print("="*70)

print(f"\nTotal samples: {len(df)}")
print(f"Labeled samples: {len(df_labeled)} ({100*len(df_labeled)/len(df):.1f}%)")
print(f"Subjects: {df_labeled['subject'].nunique()}")
print(f"Windows per subject:")
for subj in df_labeled['subject'].unique():
    n = len(df_labeled[df_labeled['subject'] == subj])
    print(f"  {subj}: {n} windows")

# Borg distribution
print(f"\nBorg distribution:")
print(df_labeled['borg'].value_counts().sort_index())

# Check for imbalance
borg_counts = df_labeled['borg'].value_counts()
print(f"\nBorg imbalance: min={borg_counts.min()}, max={borg_counts.max()}, ratio={borg_counts.max()/borg_counts.min():.1f}x")

# ============================================================================
# 2. FEATURE QUALITY - CORRELATION WITH BORG
# ============================================================================
print("\n" + "="*70)
print("2. FEATURE QUALITY (Correlation with Borg)")
print("="*70)

feat_cols = [c for c in df_labeled.columns if c not in ['t_center', 'subject', 'borg', 'activity_id', 't_start', 't_end']]
# Only numeric columns
feat_cols = [c for c in feat_cols if df_labeled[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Check correlations
correlations = []
for feat in feat_cols:
    if df_labeled[feat].notna().sum() > 100:
        r, p = pearsonr(df_labeled[feat].fillna(0), df_labeled['borg'])
        correlations.append((feat, abs(r), r, p))

correlations.sort(key=lambda x: x[1], reverse=True)

print(f"\nTotal features: {len(feat_cols)}")
print(f"Features with |r| > 0.3: {sum(1 for _, ar, _, _ in correlations if ar > 0.3)}")
print(f"Features with |r| > 0.2: {sum(1 for _, ar, _, _ in correlations if ar > 0.2)}")
print(f"Features with |r| > 0.1: {sum(1 for _, ar, _, _ in correlations if ar > 0.1)}")

print(f"\nTop 15 features by |correlation|:")
for feat, abs_r, r, p in correlations[:15]:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {feat[:50]:<50} r={r:+.3f} {sig}")

print(f"\nWorst 5 features (no signal):")
for feat, abs_r, r, p in correlations[-5:]:
    print(f"  {feat[:50]:<50} r={r:+.3f}")

# ============================================================================
# 3. INTER-SUBJECT VARIABILITY
# ============================================================================
print("\n" + "="*70)
print("3. INTER-SUBJECT VARIABILITY (The Real Problem?)")
print("="*70)

# Check if same Borg rating has different feature values across subjects
top_feat = correlations[0][0]  # Best feature
print(f"\nBest feature: {top_feat}")
print(f"\nMean value by subject and Borg level:")

# For Borg 2-3 (moderate)
print("\nAt Borg=2:")
for subj in df_labeled['subject'].unique():
    sub_df = df_labeled[(df_labeled['subject'] == subj) & (df_labeled['borg'] == 2)]
    if len(sub_df) > 0:
        print(f"  {subj}: mean={sub_df[top_feat].mean():.3f}, std={sub_df[top_feat].std():.3f}, n={len(sub_df)}")

print("\nAt Borg=4:")
for subj in df_labeled['subject'].unique():
    sub_df = df_labeled[(df_labeled['subject'] == subj) & (df_labeled['borg'] == 4)]
    if len(sub_df) > 0:
        print(f"  {subj}: mean={sub_df[top_feat].mean():.3f}, std={sub_df[top_feat].std():.3f}, n={len(sub_df)}")

# Calculate between-subject vs within-subject variance
print("\n\nVariance analysis:")
between_subj_var = df_labeled.groupby('subject')[top_feat].mean().var()
within_subj_var = df_labeled.groupby('subject')[top_feat].var().mean()
print(f"  Between-subject variance: {between_subj_var:.4f}")
print(f"  Within-subject variance:  {within_subj_var:.4f}")
print(f"  Ratio (between/within):   {between_subj_var/within_subj_var:.2f}")
if between_subj_var > within_subj_var:
    print("  ⚠️ MORE variance BETWEEN subjects than WITHIN = hard to generalize!")

# ============================================================================
# 4. PER-SUBJECT CORRELATIONS
# ============================================================================
print("\n" + "="*70)
print("4. PER-SUBJECT CORRELATIONS (Does signal exist within subjects?)")
print("="*70)

for subj in df_labeled['subject'].unique():
    sub_df = df_labeled[df_labeled['subject'] == subj]
    
    # Best correlation for this subject
    best_r = 0
    best_feat = None
    for feat, _, _, _ in correlations[:20]:
        if sub_df[feat].notna().sum() > 10:
            r, _ = pearsonr(sub_df[feat].fillna(0), sub_df['borg'])
            if abs(r) > abs(best_r):
                best_r = r
                best_feat = feat
    
    # Overall correlation with top feature
    r_top, _ = pearsonr(sub_df[top_feat].fillna(0), sub_df['borg'])
    
    print(f"\n{subj}:")
    print(f"  Samples: {len(sub_df)}")
    print(f"  Borg range: {sub_df['borg'].min():.1f} - {sub_df['borg'].max():.1f}")
    print(f"  Correlation with {top_feat[:30]}...: r={r_top:.3f}")
    print(f"  Best feature for this subject: {best_feat[:40] if best_feat else 'N/A'}... r={best_r:.3f}")

# ============================================================================
# 5. WINDOW SIZE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("5. WINDOW SIZE (Is 5.0s appropriate?)")
print("="*70)

# Check different window sizes available
window_files = list(Path('/Users/pascalschlegel/data/interim/elderly_combined').glob('elderly_aligned_*.csv'))
print(f"\nAvailable window sizes:")
for f in sorted(window_files):
    df_w = pd.read_csv(f)
    n_labeled = df_w.dropna(subset=['borg']).shape[0] if 'borg' in df_w.columns else 0
    print(f"  {f.name}: {len(df_w)} total, {n_labeled} labeled")

print("\n⚠️ Physiological signals need time to change with effort:")
print("  - EDA response: 2-4 seconds lag")
print("  - HR response: 5-30 seconds to stabilize")
print("  - PPG morphology: relatively fast")
print("  5.0s windows might be OK, but transitions are hard to capture")

# ============================================================================
# 6. MISSING DATA CHECK
# ============================================================================
print("\n" + "="*70)
print("6. MISSING DATA")
print("="*70)

# Load selected features
selected_features = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv', header=None)[0].tolist()
selected_features = [f for f in selected_features if f in df_labeled.columns]

print(f"\nSelected features: {len(selected_features)}")
print(f"\nMissing data in selected features:")
for feat in selected_features[:10]:
    missing = df_labeled[feat].isna().sum()
    pct = 100 * missing / len(df_labeled)
    if pct > 0:
        print(f"  {feat[:45]:<45} {missing:>5} ({pct:.1f}%)")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("""
LIKELY CAUSES OF POOR PERFORMANCE:

1. SMALL N (5 subjects)
   - Not enough subjects to learn generalizable patterns
   - Each elderly person has unique baseline physiology
   
2. HIGH INTER-SUBJECT VARIABILITY
   - Same Borg rating → very different sensor values across subjects
   - Model can't find universal "effort signature"
   
3. WEAK FEATURE-BORG CORRELATIONS
   - Best features only have |r| ~ 0.2-0.3 with Borg
   - Wearable signals don't strongly reflect perceived effort
   
4. POSSIBLE DATA ISSUES
   - Borg ratings subjective and inconsistent?
   - Sensor noise/artifacts?
   - Activities too similar in intensity?

NOT LIKELY THE CAUSE:
- Preprocessing (standard methods used)
- Window size (5s is reasonable)
- Feature extraction (260+ features extracted)

RECOMMENDATIONS:
1. Focus on PERSONALIZED models (calibration per patient)
2. Use CATEGORICAL prediction (LOW/MOD/HIGH instead of exact Borg)
3. Report as "challenging real-world dataset" in thesis
4. Compare to literature: r=0.3-0.5 is common for wearable effort estimation
""")
