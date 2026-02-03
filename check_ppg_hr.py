#!/usr/bin/env python3
"""Check PPG HR correlation per subject and pooled"""

import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

# Load all subjects' fused data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)
        print(f"Loaded elderly{i}: {len(df)} windows")

combined = pd.concat(all_dfs, ignore_index=True)
print(f'\nTotal windows: {len(combined)}')

# Check PPG HR correlation with Borg across ALL subjects
print("\n=== POOLED (ALL SUBJECTS) ===")
for col in ['ppg_green_hr_mean', 'ppg_infra_hr_mean']:
    valid = combined[[col, 'borg']].dropna()
    r, p = pearsonr(valid[col], valid['borg'])
    print(f'{col}: r = {r:.3f}')

# Per-subject
print("\n=== PER-SUBJECT ===")
for sub in sorted(combined['subject'].unique()):
    sub_df = combined[combined['subject'] == sub]
    valid = sub_df[['ppg_green_hr_mean', 'borg']].dropna()
    if len(valid) > 2:
        r, _ = pearsonr(valid['ppg_green_hr_mean'], valid['borg'])
        print(f'{sub}: ppg_green_hr_mean r = {r:.3f}, n={len(valid)}')
