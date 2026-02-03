#!/usr/bin/env python3
"""
Consistent Feature Selection Module
====================================
Select features that correlate in the SAME DIRECTION across ALL subjects.
This ensures generalization to new subjects.

Usage:
    from ml.consistent_feature_selection import select_consistent_features
    
    selected_features = select_consistent_features(
        df,
        subject_col='subject',
        target_col='borg',
        min_subjects=4,
        min_correlation=0.05,
        top_n=30
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import List, Tuple, Optional


def get_per_subject_correlations(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    subject_col: str = 'subject',
    target_col: str = 'borg',
    min_samples: int = 20
) -> dict:
    """
    Calculate correlation of each feature with target FOR EACH SUBJECT.
    
    Returns:
        dict: {feature_name: {subject: correlation, ...}, ...}
    """
    subjects = df[subject_col].unique()
    results = {}
    
    for feat in feature_cols:
        per_sub = {}
        for sub in subjects:
            sub_df = df[df[subject_col] == sub][[feat, target_col]].dropna()
            if len(sub_df) >= min_samples:
                try:
                    r, p = pearsonr(sub_df[feat], sub_df[target_col])
                    if not np.isnan(r):
                        per_sub[sub] = {'r': r, 'p': p, 'n': len(sub_df)}
                except:
                    pass
        
        if len(per_sub) > 0:
            results[feat] = per_sub
    
    return results


def find_consistent_features(
    per_subject_corrs: dict,
    min_subjects: int = 4,
    min_abs_correlation: float = 0.05
) -> List[Tuple[str, float, dict]]:
    """
    Find features where correlation has SAME SIGN for all subjects.
    
    Args:
        per_subject_corrs: Output from get_per_subject_correlations()
        min_subjects: Minimum number of subjects that must have data
        min_abs_correlation: Minimum |r| for each subject
        
    Returns:
        List of (feature_name, avg_correlation, per_subject_dict)
        sorted by minimum absolute correlation (most consistent first)
    """
    consistent = []
    
    for feat, per_sub in per_subject_corrs.items():
        if len(per_sub) < min_subjects:
            continue
        
        correlations = [v['r'] for v in per_sub.values()]
        
        # Check if all same sign
        all_positive = all(r > 0 for r in correlations)
        all_negative = all(r < 0 for r in correlations)
        
        if not (all_positive or all_negative):
            continue
        
        # Check minimum correlation strength
        min_abs_r = min(abs(r) for r in correlations)
        if min_abs_r < min_abs_correlation:
            continue
        
        avg_r = np.mean(correlations)
        consistent.append((feat, avg_r, min_abs_r, per_sub))
    
    # Sort by minimum absolute correlation (most consistent first)
    consistent.sort(key=lambda x: x[2], reverse=True)
    
    return consistent


def prune_redundant_features(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'borg',
    corr_threshold: float = 0.90
) -> List[str]:
    """
    Remove highly correlated features, keeping the one with higher target correlation.
    """
    if len(features) <= 1:
        return features
    
    # Calculate feature-feature correlations
    X = df[features].dropna()
    if len(X) < 50:
        return features
    
    corr_matrix = X.corr().abs().values
    np.fill_diagonal(corr_matrix, 0.0)
    
    # Get target correlations
    target_corrs = {}
    for f in features:
        valid = df[[f, target_col]].dropna()
        if len(valid) > 20:
            r, _ = pearsonr(valid[f], valid[target_col])
            target_corrs[f] = abs(r)
        else:
            target_corrs[f] = 0
    
    # Prune
    keep = set(range(len(features)))
    while True:
        keep_list = sorted(list(keep))
        if len(keep_list) <= 1:
            break
        
        sub = corr_matrix[np.ix_(keep_list, keep_list)]
        max_val = sub.max()
        if max_val < corr_threshold:
            break
        
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        fi, fj = keep_list[i], keep_list[j]
        
        # Drop the one with lower target correlation
        if target_corrs[features[fi]] <= target_corrs[features[fj]]:
            keep.discard(fi)
        else:
            keep.discard(fj)
    
    return [features[i] for i in sorted(keep)]


def select_consistent_features(
    df: pd.DataFrame,
    subject_col: str = 'subject',
    target_col: str = 'borg',
    min_subjects: int = 4,
    min_abs_correlation: float = 0.05,
    top_n: int = 30,
    prune_redundant: bool = True,
    prune_threshold: float = 0.90,
    verbose: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Main function: Select features that correlate consistently across all subjects.
    
    Args:
        df: DataFrame with features, subject column, and target column
        subject_col: Name of subject identifier column
        target_col: Name of target column (e.g., 'borg')
        min_subjects: Minimum subjects that must show same correlation direction
        min_abs_correlation: Minimum |r| for each subject
        top_n: Maximum features to select
        prune_redundant: Whether to prune highly correlated features
        prune_threshold: Correlation threshold for pruning
        verbose: Print progress info
        
    Returns:
        Tuple of:
        - List of selected feature names
        - DataFrame with feature selection details
    """
    if verbose:
        print("="*60)
        print("CONSISTENT FEATURE SELECTION")
        print("="*60)
    
    # Get all numeric feature columns
    skip_cols = {subject_col, target_col, 't_center', 't_start', 't_end', 
                 'window_id', 'start_idx', 'end_idx', 'valid', 'n_samples',
                 'win_sec', 'modality', 'activity_label'}
    
    feature_cols = [c for c in df.columns 
                    if c not in skip_cols 
                    and not c.startswith('Unnamed')
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if verbose:
        print(f"\nTotal features to evaluate: {len(feature_cols)}")
        print(f"Subjects: {sorted(df[subject_col].unique())}")
    
    # Step 1: Get per-subject correlations
    if verbose:
        print("\n1. Computing per-subject correlations...")
    
    per_sub_corrs = get_per_subject_correlations(
        df, feature_cols, subject_col, target_col
    )
    
    if verbose:
        print(f"   Features with data: {len(per_sub_corrs)}")
    
    # Step 2: Find consistent features
    if verbose:
        print(f"\n2. Finding features with consistent direction (min {min_subjects} subjects)...")
    
    consistent = find_consistent_features(
        per_sub_corrs, 
        min_subjects=min_subjects,
        min_abs_correlation=min_abs_correlation
    )
    
    if verbose:
        print(f"   Consistent features found: {len(consistent)}")
    
    if len(consistent) == 0:
        if verbose:
            print("   WARNING: No consistent features found! Relaxing criteria...")
        # Try with lower threshold
        consistent = find_consistent_features(
            per_sub_corrs, 
            min_subjects=min_subjects - 1,
            min_abs_correlation=0.01
        )
        if verbose:
            print(f"   After relaxing: {len(consistent)} features")
    
    # Step 3: Take top N
    selected = [c[0] for c in consistent[:top_n]]
    
    if verbose:
        print(f"\n3. Selected top {len(selected)} features")
    
    # Step 4: Prune redundant
    if prune_redundant and len(selected) > 1:
        if verbose:
            print(f"\n4. Pruning redundant features (threshold={prune_threshold})...")
        
        selected = prune_redundant_features(
            df, selected, target_col, prune_threshold
        )
        
        if verbose:
            print(f"   After pruning: {len(selected)} features")
    
    # Create summary DataFrame
    summary_rows = []
    for feat, avg_r, min_r, per_sub in consistent:
        if feat in selected:
            row = {
                'feature': feat,
                'avg_r': avg_r,
                'min_abs_r': min_r,
                'direction': '+' if avg_r > 0 else '-',
                'n_subjects': len(per_sub),
                'selected': True
            }
            for sub, vals in per_sub.items():
                row[f'r_{sub}'] = vals['r']
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    if verbose:
        print("\n" + "="*60)
        print("SELECTED FEATURES:")
        print("="*60)
        for feat, avg_r, min_r, _ in consistent[:len(selected)]:
            if feat in selected:
                direction = '+' if avg_r > 0 else '-'
                modality = 'PPG' if 'ppg' in feat.lower() else ('EDA' if 'eda' in feat.lower() else 'IMU')
                print(f"  {direction} {feat} ({modality}): avg r = {avg_r:.3f}, min |r| = {min_r:.3f}")
    
    return selected, summary_df


def select_features_for_loso(
    df: pd.DataFrame,
    test_subject: str,
    subject_col: str = 'subject',
    target_col: str = 'borg',
    top_n: int = 30,
    verbose: bool = False
) -> List[str]:
    """
    Select consistent features using ONLY training subjects (LOSO-compatible).
    
    This is the correct way to do feature selection in LOSO cross-validation:
    features are selected on training data only.
    
    Args:
        df: Full DataFrame
        test_subject: Subject to hold out
        subject_col: Name of subject column
        target_col: Name of target column
        top_n: Number of features to select
        verbose: Print progress
        
    Returns:
        List of selected feature names
    """
    # Use only training subjects
    train_df = df[df[subject_col] != test_subject].copy()
    n_train_subjects = train_df[subject_col].nunique()
    
    # Select consistent features on training data
    selected, _ = select_consistent_features(
        train_df,
        subject_col=subject_col,
        target_col=target_col,
        min_subjects=max(2, n_train_subjects - 1),  # Allow 1 subject to be inconsistent
        min_abs_correlation=0.05,
        top_n=top_n,
        prune_redundant=True,
        verbose=verbose
    )
    
    return selected


# =============================================================================
# MAIN: Test the module
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    print("Testing consistent feature selection...")
    
    # Load data
    all_dfs = []
    for i in [1, 2, 3, 4, 5]:
        path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
        if path.exists():
            df = pd.read_csv(path)
            df['subject'] = f'elderly{i}'
            all_dfs.append(df)
    
    df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])
    
    # Test feature selection
    selected, summary = select_consistent_features(
        df_all,
        subject_col='subject',
        target_col='borg',
        min_subjects=4,
        top_n=30,
        verbose=True
    )
    
    print(f"\n\nFinal selected features: {len(selected)}")
    print("\nSummary:")
    print(summary[['feature', 'avg_r', 'min_abs_r', 'direction']].to_string())
