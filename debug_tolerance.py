#!/usr/bin/env python3
"""Check actual tolerance requirements."""

import pandas as pd
import numpy as np
from pathlib import Path

base = Path('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3')

# Check the actual t_center spacing and alignment
eda = pd.read_csv(base / 'eda/eda_features_10.0s.csv')
imu_bioz = pd.read_csv(base / 'imu_bioz/imu_features_10.0s.csv')

# Get spacing
eda_spacing = eda['t_center'].diff().dropna()
imu_spacing = imu_bioz['t_center'].diff().dropna()

print('Window spacing (should be ~9s for 10% overlap):')
print(f'  EDA:      mean={eda_spacing.mean():.1f}s, std={eda_spacing.std():.1f}s')
print(f'  IMU_bioz: mean={imu_spacing.mean():.1f}s, std={imu_spacing.std():.1f}s')

# Check what tolerance we need to align
# Find the minimum distance between any EDA and IMU t_center
eda_t = eda['t_center'].values
imu_t = imu_bioz['t_center'].values

# For each IMU timestamp, find closest EDA
min_diffs = []
for t in imu_t[:50]:  # Sample first 50
    diffs = np.abs(eda_t - t)
    min_diffs.append(diffs.min())

print(f'\nMin time diff between IMU and nearest EDA window:')
print(f'  Min: {min(min_diffs):.1f}s, Max: {max(min_diffs):.1f}s, Mean: {np.mean(min_diffs):.1f}s')
print(f'  Samples with diff > 5s: {sum(d > 5 for d in min_diffs)} of {len(min_diffs)}')
print(f'  Samples with diff > 2s: {sum(d > 2 for d in min_diffs)} of {len(min_diffs)}')
