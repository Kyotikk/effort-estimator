#!/usr/bin/env python3
"""Get exact pipeline statistics for presentation."""

import pandas as pd
from pathlib import Path

# Load the final training data
combined = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
selected_path = Path('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s')
selected = pd.read_csv(selected_path / 'pruned_features.csv') if (selected_path / 'pruned_features.csv').exists() else combined

print('=== Dataset Statistics ===')
print(f'Total samples: {len(combined)}')
print(f'Samples with Borg labels: {combined["borg"].notna().sum()}')
print(f'Unique subjects: {combined["subject"].nunique()}')
if "label" in combined.columns:
    print(f'Unique activities: {combined["label"].nunique()}')

print(f'\nFeatures before selection: {len([c for c in combined.columns if c not in ["t_center", "subject", "label", "borg", "modality", "valid", "n_samples", "win_sec", "valid_r", "n_samples_r", "win_sec_r"]])}')
print(f'Features after selection: {len([c for c in selected.columns if c not in ["t_center", "subject", "label", "borg", "modality", "valid", "n_samples", "win_sec", "valid_r", "n_samples_r", "win_sec_r"]])}')

# Feature breakdown by modality
feat_cols = [c for c in selected.columns if c not in ['t_center', 'subject', 'label', 'borg', 'modality', 'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 'win_sec_r']]
eda_feats = [c for c in feat_cols if 'eda' in c.lower()]
imu_feats = [c for c in feat_cols if 'acc_' in c.lower()]
ppg_feats = [c for c in feat_cols if 'ppg' in c.lower()]
print(f'\nSelected features by modality:')
print(f'  EDA: {len(eda_feats)}')
print(f'  IMU: {len(imu_feats)}')
print(f'  PPG: {len(ppg_feats)}')

# Borg distribution
print(f'\nBorg score distribution:')
borg_labeled = combined[combined['borg'].notna()]['borg']
print(f'  Range: {borg_labeled.min():.0f} - {borg_labeled.max():.0f}')
print(f'  Mean ± Std: {borg_labeled.mean():.2f} ± {borg_labeled.std():.2f}')
print(f'  Median: {borg_labeled.median():.1f}')

# Per-subject breakdown
print(f'\nPer-subject sample counts:')
for subj in combined['subject'].unique():
    subj_data = combined[combined['subject'] == subj]
    labeled = subj_data['borg'].notna().sum()
    print(f'  {subj}: {len(subj_data)} total, {labeled} labeled')

# Activity count per subject
if 'label' in combined.columns:
    print(f'\nActivities per subject:')
    for subj in combined['subject'].unique():
        subj_data = combined[combined['subject'] == subj]
        n_activities = subj_data['label'].nunique()
        print(f'  {subj}: {n_activities} unique activities')

# List all feature names
print(f'\n=== All Selected Features ({len(feat_cols)}) ===')
for i, f in enumerate(sorted(feat_cols)):
    print(f'  {i+1}. {f}')
