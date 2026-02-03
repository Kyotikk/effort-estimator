#!/usr/bin/env python3
"""
LOSO-Aware Feature Selection for Effort Estimation

This module provides feature selection that respects Leave-One-Subject-Out (LOSO)
evaluation to avoid data leakage.

Key improvements over original feature_selection_and_qc.py:
1. No random train/test split - uses subject-based splitting
2. Features selected based on TRAINING subjects only (no peeking at test subject)
3. Two strategies:
   - IMU-focused: Just use IMU features (proven to generalize best)
   - Consistent: Use features that correlate in same direction for all subjects
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_feature_columns(df):
    """Extract feature columns from dataframe (excluding metadata)"""
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            return True
        return False
    
    return [col for col in df.columns if not is_metadata(col)]


def select_imu_features(feature_cols):
    """Select only IMU features (acc_ prefix) - proven best for generalization"""
    return [c for c in feature_cols if c.startswith('acc_')]


def select_consistent_features_for_loso(X_train, y_train, subjects_train, feature_cols, min_corr=0.15):
    """
    Select features that correlate consistently (same direction) across ALL training subjects.
    
    This avoids data leakage because:
    1. Only looks at training subjects (test subject excluded)
    2. Requires feature to work for ALL training subjects
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets
        subjects_train: Subject ID for each training sample
        feature_cols: Feature column names
        min_corr: Minimum absolute correlation required
        
    Returns:
        selected_indices: Indices of selected features
        selected_cols: Names of selected features
    """
    unique_subjects = np.unique(subjects_train)
    n_features = X_train.shape[1]
    
    # Calculate per-subject correlations
    correlations = {}
    for subj in unique_subjects:
        mask = subjects_train == subj
        X_subj = X_train[mask]
        y_subj = y_train[mask]
        
        corrs = []
        for i in range(n_features):
            corr = np.corrcoef(X_subj[:, i], y_subj)[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0)
        correlations[subj] = np.array(corrs)
    
    # Find features with consistent direction across ALL subjects
    selected_indices = []
    selected_cols = []
    
    for i in range(n_features):
        subj_corrs = [correlations[s][i] for s in unique_subjects]
        
        # Check if all correlations have same sign and meet minimum threshold
        all_positive = all(c >= min_corr for c in subj_corrs)
        all_negative = all(c <= -min_corr for c in subj_corrs)
        
        if all_positive or all_negative:
            selected_indices.append(i)
            selected_cols.append(feature_cols[i])
    
    return selected_indices, selected_cols


def select_features_for_loso_fold(df, test_subject, strategy='imu', min_corr=0.15):
    """
    Select features for a single LOSO fold, using ONLY training data.
    
    Args:
        df: Full dataframe with all subjects
        test_subject: Subject to exclude (test subject)
        strategy: 'imu' (recommended) or 'consistent'
        min_corr: Minimum correlation for consistent strategy
        
    Returns:
        selected_cols: List of selected feature column names
    """
    feature_cols = get_feature_columns(df)
    
    if strategy == 'imu':
        # Just use IMU features - no fitting needed, proven best
        return select_imu_features(feature_cols)
    
    elif strategy == 'consistent':
        # Use features consistent across training subjects
        train_df = df[df['subject'] != test_subject].dropna(subset=['borg'])
        X_train = train_df[feature_cols].values
        y_train = train_df['borg'].values
        subjects_train = train_df['subject'].values
        
        _, selected_cols = select_consistent_features_for_loso(
            X_train, y_train, subjects_train, feature_cols, min_corr
        )
        return selected_cols
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_loso_feature_selection(fused_aligned_path, output_dir, strategy='imu'):
    """
    Main feature selection for LOSO evaluation.
    
    Instead of saving one feature set, we document the strategy so training
    code can apply it correctly for each fold.
    
    Args:
        fused_aligned_path: Path to fused_aligned_*.csv
        output_dir: Output directory
        strategy: 'imu' or 'consistent'
    """
    print("\n" + "="*80)
    print(f"LOSO-AWARE FEATURE SELECTION (strategy: {strategy})")
    print("="*80)
    
    # Load data
    df = pd.read_csv(fused_aligned_path)
    df_labeled = df.dropna(subset=['borg']).copy()
    subjects = df_labeled['subject'].unique()
    
    print(f"\n📊 Data summary:")
    print(f"   Subjects: {list(subjects)}")
    print(f"   Total samples: {len(df_labeled)}")
    
    feature_cols = get_feature_columns(df)
    print(f"   Total features: {len(feature_cols)}")
    
    # Show feature selection for each fold
    print(f"\n🎯 Feature selection per LOSO fold:")
    
    ensure_dir(output_dir)
    
    for test_subj in subjects:
        selected = select_features_for_loso_fold(df_labeled, test_subj, strategy)
        print(f"   Test {test_subj}: {len(selected)} features selected")
    
    # Save strategy info
    strategy_info = {
        'strategy': strategy,
        'description': {
            'imu': 'Use all IMU features (acc_* prefix). Best per-subject generalization.',
            'consistent': 'Use features with consistent correlation direction across ALL training subjects.'
        }[strategy],
        'n_subjects': len(subjects),
        'subjects': list(subjects)
    }
    
    # For IMU strategy, save the feature list (same for all folds)
    if strategy == 'imu':
        imu_features = select_imu_features(feature_cols)
        pd.Series(imu_features).to_csv(
            Path(output_dir) / 'features_selected_imu.csv',
            index=False, header=False
        )
        print(f"\n💾 Saved: features_selected_imu.csv ({len(imu_features)} features)")
    
    # Save strategy documentation
    with open(Path(output_dir) / 'feature_selection_strategy.txt', 'w') as f:
        f.write(f"Feature Selection Strategy: {strategy}\n")
        f.write("="*60 + "\n\n")
        f.write(strategy_info['description'] + "\n\n")
        f.write("IMPORTANT: For LOSO evaluation, features must be selected\n")
        f.write("on TRAINING subjects only to avoid data leakage.\n\n")
        f.write(f"Subjects: {strategy_info['subjects']}\n")
    
    print(f"💾 Saved: feature_selection_strategy.txt")
    
    print("\n" + "="*80)
    print("✅ LOSO-AWARE FEATURE SELECTION COMPLETE")
    print("="*80)
    
    return strategy_info


# Convenience function for training scripts
def get_loso_features(df, test_subject, strategy='imu'):
    """
    Get features for a LOSO fold. Call this from training code.
    
    Example:
        for test_subject in subjects:
            feature_cols = get_loso_features(df, test_subject, strategy='imu')
            train_df = df[df['subject'] != test_subject]
            test_df = df[df['subject'] == test_subject]
            X_train = train_df[feature_cols].values
            ...
    """
    return select_features_for_loso_fold(df, test_subject, strategy)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loso_feature_selection.py <fused_aligned_path> [output_dir] [strategy]")
        print("\nStrategies:")
        print("  imu        - Use all IMU features (recommended, r~0.55)")
        print("  consistent - Use features consistent across all subjects (r~0.30)")
        sys.exit(1)
    
    fused_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./feature_selection_output"
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'imu'
    
    run_loso_feature_selection(fused_path, out_dir, strategy)
