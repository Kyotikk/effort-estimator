"""
Alignment module - aligns fused features with target labels.

Callable functions:
- align_fused_data()  # Main alignment function
"""

import pandas as pd
import numpy as np
from pathlib import Path


def align_fused_data(fused_csv, target_csv, tolerance_s=2.0):
    """
    Align fused features with target labels by time.
    
    Args:
        fused_csv: Path to fused features CSV
        target_csv: Path to target labels CSV  
        tolerance_s: Time tolerance for alignment (seconds)
        
    Returns:
        DataFrame with aligned features and labels
    """
    df_fused = pd.read_csv(fused_csv)
    df_targets = pd.read_csv(target_csv)
    
    # Merge on time column with tolerance
    merged = pd.merge_asof(
        df_fused.sort_values('t_center'),
        df_targets.sort_values('t_start'),
        left_on='t_center',
        right_on='t_start',
        direction='nearest',
        tolerance=tolerance_s
    )
    
    # Remove rows where alignment failed (NaN borg score)
    aligned = merged.dropna(subset=['borg']).copy()
    
    return aligned


def save_aligned_data(df, output_path, window_length):
    """Save aligned data to CSV."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"fused_aligned_{window_length:.1f}s.csv"
    df.to_csv(output_file, index=False)
    return output_file
