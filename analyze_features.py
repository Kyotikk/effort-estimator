#!/usr/bin/env python3
"""
Feature analysis script - shows exact features used in the pipeline
"""
import pandas as pd
from pathlib import Path

# Load one subject's data to see actual feature columns
path = Path('/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv')
df = pd.read_csv(path)

# Get all feature columns
skip = {'window_id', 'start_idx', 'end_idx', 'valid', 't_start', 't_center', 't_end', 'n_samples', 'win_sec', 'modality', 'subject', 'borg'}

all_cols = [c for c in df.columns if c not in skip]

# Categorize by modality
imu_cols = [c for c in all_cols if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in all_cols if 'ppg_' in c and '_r' not in c]
eda_cols = [c for c in all_cols if 'eda_' in c and '_r' not in c]

print("=" * 70)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 70)

print(f"\nIMU FEATURES ({len(imu_cols)} total):")
for c in sorted(imu_cols):
    print(f"   {c}")

print(f"\nPPG FEATURES ({len(ppg_cols)} total):")
for c in sorted(ppg_cols):
    print(f"   {c}")

print(f"\nEDA FEATURES ({len(eda_cols)} total):")
for c in sorted(eda_cols):
    print(f"   {c}")

print(f"\n{'='*70}")
print(f"TOTAL: {len(imu_cols)} IMU + {len(ppg_cols)} PPG + {len(eda_cols)} EDA = {len(all_cols)} features")
print("="*70)
