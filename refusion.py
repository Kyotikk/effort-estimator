#!/usr/bin/env python3
"""Re-run fusion with HRV features."""
from ml.fusion.fuse_windows import fuse_modalities
import pandas as pd
import yaml

# Load config
with open('config/pipeline.yaml') as f:
    cfg = yaml.safe_load(f)

base = '/Users/pascalschlegel/data/interim/parsingsim3'
subjects = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']

all_dfs = []
for subj in subjects:
    out_dir = f'{base}/{subj}/effort_estimation_output/parsingsim3_{subj}'
    fused = fuse_modalities(out_dir, cfg['feature_extraction']['windows'], cfg['fusion']['modalities'])
    fused['subject'] = subj
    all_dfs.append(fused)
    print(f'{subj}: {len(fused)} rows, {len(fused.columns)} cols')

combined = pd.concat(all_dfs, ignore_index=True)
print(f'\nCombined: {len(combined)} rows, {len(combined.columns)} cols')

# Check for HRV features
hrv_cols = [c for c in combined.columns if 'rmssd' in c.lower() or 'hr_mean' in c.lower() or 'mean_ibi' in c.lower()]
print(f'\nHRV columns found: {hrv_cols}')

# Save
combined.to_csv(f'{base}/multisub_combined/multisub_fused_10.0s.csv', index=False)
print('Saved to multisub_fused_10.0s.csv')
