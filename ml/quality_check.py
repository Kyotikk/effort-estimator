"""
Quality check module - validates data integrity.

Callable functions:
- check_data_quality()  # Main QC function
"""

import pandas as pd
import numpy as np


def check_data_quality(df, features_only=False):
    """
    Perform quality checks on data.
    
    Args:
        df: DataFrame to check
        features_only: Only check feature columns (exclude metadata)
        
    Returns:
        dict: QC results
    """
    results = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'zero_variance': [],
        'high_correlation': [],
    }
    
    # Get feature columns only
    if features_only:
        skip_cols = {
            'window_id', 'start_idx', 'end_idx', 'valid',
            't_start', 't_center', 't_end', 'n_samples', 'win_sec',
            'modality', 'subject', 'borg'
        }
        feature_cols = [c for c in df.columns if c not in skip_cols and not c.endswith('_r')]
        data = df[feature_cols]
    else:
        data = df
    
    # Check zero variance
    for col in data.columns:
        if data[col].var() == 0:
            results['zero_variance'].append(col)
    
    # Check high correlation
    corr_matrix = data.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for i, col in enumerate(upper_tri.columns):
        high_corr_cols = upper_tri[col][upper_tri[col] > 0.99]
        for corr_col in high_corr_cols.index:
            high_corr_pairs.append((col, corr_col))
    
    results['high_correlation'] = high_corr_pairs[:10]  # Limit to top 10
    
    return results


def print_qc_results(qc_results):
    """Print QC results in readable format."""
    print(f"\nData Quality Check:")
    print(f"  Rows: {qc_results['n_rows']}")
    print(f"  Columns: {qc_results['n_cols']}")
    print(f"  Zero variance features: {len(qc_results['zero_variance'])}")
    print(f"  Highly correlated pairs: {len(qc_results['high_correlation'])}")
    
    if qc_results['zero_variance']:
        print(f"  ⚠ Zero variance: {qc_results['zero_variance'][:5]}")
    
    if qc_results['high_correlation']:
        print(f"  ⚠ High correlation detected")
