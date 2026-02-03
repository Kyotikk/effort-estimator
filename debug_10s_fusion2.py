#!/usr/bin/env python3
"""Debug why 10s fusion produces 0 rows - part 2."""

import pandas as pd
from pathlib import Path

base = Path('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3')

eda = pd.read_csv(base / 'eda/eda_features_10.0s.csv')
imu_bioz = pd.read_csv(base / 'imu_bioz/imu_features_10.0s.csv')

# Test merge_asof
eda_sorted = eda.sort_values('t_center')
imu_sorted = imu_bioz.sort_values('t_center')

# With no tolerance
merged = pd.merge_asof(eda_sorted, imu_sorted[['t_center']], on='t_center', direction='nearest')
print(f'Merged (no tolerance): {len(merged)} rows')

# With 5s tolerance
merged5 = pd.merge_asof(eda_sorted, imu_sorted[['t_center']], on='t_center', direction='nearest', tolerance=5.0)
print(f'Merged (5s tolerance): {len(merged5)} rows')

# Check the difference in t_center values
print(f'\nEDA first 3 t_center: {list(eda_sorted.t_center.head(3))}')
print(f'IMU first 3 t_center: {list(imu_sorted.t_center.head(3))}')
print(f'Diff of first matched pair: {abs(eda_sorted.t_center.iloc[0] - imu_sorted.t_center.iloc[0]):.1f}s')

# Check for the actual problem: multiple modalities need sequential fusion
# Let's replicate what run_fusion does

print("\n=== Simulating full 10s fusion ===")

from ml.fusion.fuse_windows import fuse_feature_tables

# Load all modalities - with correct paths
modalities = {}
paths = {
    'eda': 'eda/eda_features_10.0s.csv',
    'eda_advanced': 'eda/eda_advanced_features_10.0s.csv',
    'imu_bioz': 'imu_bioz/imu_features_10.0s.csv',
    'imu_wrist': 'imu_wrist/imu_features_10.0s.csv',
    'ppg_green': 'ppg_green/ppg_green_features_10.0s.csv',
    'ppg_green_hrv': 'ppg_green/ppg_green_hrv_features_10.0s.csv',
    'ppg_infra': 'ppg_infra/ppg_infra_features_10.0s.csv',
    'ppg_infra_hrv': 'ppg_infra/ppg_infra_hrv_features_10.0s.csv',
    'ppg_red': 'ppg_red/ppg_red_features_10.0s.csv',
    'ppg_red_hrv': 'ppg_red/ppg_red_hrv_features_10.0s.csv',
}

for name, path_rel in paths.items():
    path = base / path_rel
    if path.exists():
        df = pd.read_csv(path)
        modalities[name] = df
        t = df['t_center']
        print(f"Loaded {name:15s}: {len(df):4d} rows | t_center: {t.min():.0f} - {t.max():.0f}")

# Try fusion with different tolerances
for tol in [5.0, 10.0, 15.0]:
    tables = list(modalities.values())
    fused = fuse_feature_tables(tables, join_col='t_center', tolerance_sec=tol)
    print(f"\nFused with tolerance={tol}s: {len(fused)} rows")
