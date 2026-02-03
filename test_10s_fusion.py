#!/usr/bin/env python3
"""Test 10s fusion with corrected tolerance."""

import pandas as pd
from pathlib import Path
from ml.fusion.fuse_windows import fuse_feature_tables

base = Path('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3')

# Load all 10s feature tables
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

tables = [pd.read_csv(base / p) for p in paths.values()]
print('Loaded all 10 feature tables')

# Fuse with 5s tolerance (enough for 4s offset)
fused = fuse_feature_tables(tables, join_col='t_center', tolerance_sec=5.0)
print(f'Fused 10s with 5s tolerance: {len(fused)} rows, {len(fused.columns)} features')

# Compare to 5s windows
fused_5s = pd.read_csv(base / 'fused_features_5.0s.csv')
print(f'Existing 5s fusion: {len(fused_5s)} rows, {len(fused_5s.columns)} features')
