#!/usr/bin/env python3
"""Quick data quality check"""
import pandas as pd

# Load current filtered data (32 features)
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/qc_5.0s/features_filtered_5.0s.csv')
feat_cols = [c for c in df.columns if c not in ['subject', 'borg', 't_center']]

print('=== CURRENT DATA: 32 Selected Features ===')
print(f'Samples: {len(df)}, Features: {len(feat_cols)}')

missing = df[feat_cols].isna().mean()
print(f'\nMissingness: {(missing == 0).sum()} features have 0% missing')
print(f'Max missing: {missing.max()*100:.1f}%')

if missing.max() > 0:
    print('\nTop missing:')
    for f, p in missing.nlargest(5).items():
        if p > 0: print(f'  {f}: {p*100:.1f}%')
else:
    print('\n✅ NO MISSING DATA in selected features!')

# Full dataset
print('\n=== FULL DATASET (260 Features) ===')
df_full = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')
feat_full = [c for c in df_full.columns if c not in ['subject', 'borg', 't_center']]
missing_full = df_full[feat_full].isna().mean()
print(f'Total features: {len(feat_full)}')
print(f'>50% missing: {(missing_full > 0.5).sum()} features')
print(f'>80% missing: {(missing_full > 0.8).sum()} features')

# Show the bad ones
bad = missing_full[missing_full > 0.5].sort_values(ascending=False)
if len(bad) > 0:
    print('\nFeatures >50% missing (excluded from selection):')
    for f, p in bad.head(15).items():
        print(f'  {f}: {p*100:.1f}%')
