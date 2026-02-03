#!/usr/bin/env python3
"""Debug time alignment"""
import pandas as pd
import numpy as np
from pathlib import Path

# Check one subject's data
base = Path('/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1')
green_hpf = pd.read_csv(base / 'ppg_green' / 'ppg_green_preprocessed_hpf.csv')
fused = pd.read_csv(base / 'fused_aligned_5.0s.csv')

print("Green HPF signal:")
print(f"  Rows: {len(green_hpf)}")
print(f"  t_sec range: {green_hpf['t_sec'].min():.1f} - {green_hpf['t_sec'].max():.1f}")
print(f"  Value stats: mean={green_hpf['value'].mean():.1f}, std={green_hpf['value'].std():.1f}")
print(f"  Any NaN: {green_hpf['value'].isna().sum()}")
print(f"  Any Inf: {np.isinf(green_hpf['value']).sum()}")

print("\nFused data:")
print(f"  Rows: {len(fused)}")
fused_labeled = fused.dropna(subset=['borg'])
print(f"  Labeled rows: {len(fused_labeled)}")
print(f"  t_center range: {fused_labeled['t_center'].min():.1f} - {fused_labeled['t_center'].max():.1f}")

# Quick feature test
from features.ppg_features import _basic_features
window = green_hpf['value'].values[1000:1160]  # 5s at 32Hz = 160 samples
feats = _basic_features(window, 'test_')
print(f"\nTest feature extraction:")
print(f"  Window size: {len(window)}")
print(f"  Features: {len(feats)}")
print(f"  Sample values: {list(feats.items())[:3]}")
print(f"  Any NaN in features: {sum(1 for v in feats.values() if pd.isna(v))}")
