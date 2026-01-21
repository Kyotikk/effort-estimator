"""
Analyze data quality and NaN patterns
"""
import pandas as pd
import numpy as np
from pathlib import Path

reduced_df = pd.read_csv("output/hrv_recovery_reduced.csv")
print("="*70)
print("DATA QUALITY ANALYSIS")
print("="*70)

# Show all columns
exclude_cols = [
    'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
    'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
    'qc_ok', 'effort', 'subject_id'
]

feature_cols = [c for c in reduced_df.columns if c not in exclude_cols]
print(f"\nTotal rows: {len(reduced_df)}")
print(f"Total features: {len(feature_cols)}")
print(f"Target variable: delta_rmssd")

# NaN analysis
nan_counts = reduced_df[feature_cols].isna().sum()
nan_pct = 100 * nan_counts / len(reduced_df)

print("\nNaN Analysis by Feature:")
print("-" * 70)

nan_summary = pd.DataFrame({
    'Feature': feature_cols,
    'NaN_Count': nan_counts.values,
    'NaN_Percent': nan_pct.values
}).sort_values('NaN_Count', ascending=False)

print(nan_summary.to_string(index=False))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Features with NO NaNs: {(nan_counts == 0).sum()}")
print(f"Features with <20% NaNs: {(nan_counts < len(reduced_df)*0.2).sum()}")
print(f"Features with 20-50% NaNs: {((nan_counts >= len(reduced_df)*0.2) & (nan_counts < len(reduced_df)*0.5)).sum()}")
print(f"Features with >50% NaNs: {(nan_counts >= len(reduced_df)*0.5).sum()}")
print(f"\nRows with all values: {(reduced_df[feature_cols].notna().all(axis=1)).sum()}")
print(f"Rows with at least 1 NaN: {(reduced_df[feature_cols].isna().any(axis=1)).sum()}")

# Show which rows have issues
rows_with_nans = reduced_df[feature_cols].isna().any(axis=1)
print(f"\nRows with NaNs:")
print(reduced_df[rows_with_nans][['subject_id', 'task_name', 'bout_id', 'delta_rmssd']])
