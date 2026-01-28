#!/usr/bin/env python3
"""Re-run feature selection with HRV features and train model."""
import pandas as pd
import numpy as np
import json
from ml.feature_selection import select_features

# Load combined data with HRV
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
df_labeled = df.dropna(subset=['borg'])
print(f'Labeled samples: {len(df_labeled)}')

# Check HRV features in data
hrv_cols = [c for c in df_labeled.columns if 'rmssd' in c or 'hr_mean' in c or 'mean_ibi' in c or 'sdnn' in c]
print(f'HRV features in data: {hrv_cols}')

# Run feature selection
selected_cols, X, y = select_features(df_labeled, target_col='borg', corr_threshold=0.90, top_n=100)

# Check HRV in selected
hrv_selected = [c for c in selected_cols if 'rmssd' in c or 'hr_mean' in c or 'mean_ibi' in c or 'sdnn' in c]
print(f'\nHRV in selected features: {hrv_selected}')

# Save
out_dir = '/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models'
with open(f'{out_dir}/selected_features_with_hrv.json', 'w') as f:
    json.dump(selected_cols, f, indent=2)
print(f'Saved to {out_dir}/selected_features_with_hrv.json')
