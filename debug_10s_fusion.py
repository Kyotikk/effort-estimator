#!/usr/bin/env python3
"""Debug why 10s fusion produces 0 rows."""

import pandas as pd
from pathlib import Path

base = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3")

print("=== t_center time ranges for 10.0s windows ===\n")

files = {
    "eda": base / "eda/eda_features_10.0s.csv",
    "eda_adv": base / "eda/eda_advanced_features_10.0s.csv", 
    "imu_bioz": base / "imu_bioz/imu_features_10.0s.csv",
    "imu_wrist": base / "imu_wrist/imu_features_10.0s.csv",
    "ppg_green": base / "ppg_green/ppg_green_features_10.0s.csv",
    "ppg_green_hrv": base / "ppg_green/ppg_green_hrv_features_10.0s.csv",
    "ppg_infra": base / "ppg_infra/ppg_infra_features_10.0s.csv",
    "ppg_red": base / "ppg_red/ppg_red_features_10.0s.csv",
}

dfs = {}
for name, path in files.items():
    if path.exists():
        df = pd.read_csv(path)
        if "t_center" in df.columns:
            t = df["t_center"]
            dfs[name] = df
            print(f"{name:15s}: min={t.min():.0f} max={t.max():.0f} n={len(t)}")

# Find intersection of all time ranges
all_mins = [df["t_center"].min() for df in dfs.values()]
all_maxs = [df["t_center"].max() for df in dfs.values()]
common_start = max(all_mins)
common_end = min(all_maxs)

print(f"\nCommon time range: {common_start:.0f} to {common_end:.0f}")
print(f"Duration of overlap: {common_end - common_start:.0f}s ({(common_end - common_start)/60:.1f} min)")

# Check the first few t_center values to see the spacing
print("\n=== First few t_center values for each modality ===")
for name, df in dfs.items():
    print(f"\n{name}: {list(df['t_center'].head(5).values)}")
