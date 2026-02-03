#!/usr/bin/env python3
"""
Data-Driven Feature Selection for LOSO Evaluation

This module automatically finds the BEST features for generalization,
regardless of modality (IMU, PPG, EDA, or any combination).

Key principle: Features are selected based on TRAINING subjects only,
avoiding data leakage to the test subject.

Selection criteria:
1. Feature must correlate with Borg in the SAME direction for all training subjects
2. Feature must have minimum correlation strength (configurable)
3. Optionally prune highly correlated features to reduce redundancy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


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


def find_best_features_for_fold(df_train, feature_cols, min_corr=0.10, 
                                 min_subject_ratio=0.75, top_n=None):
    """
    Find best features using ONLY training data.
    
    Strategy:
    1. Calculate per-subject correlations with Borg
    2. Keep features that correlate in SAME DIRECTION for most training subjects
    3. Rank by average correlation strength
    4. Optionally limit to top_n features
    
    Args:
        df_train: Training dataframe (test subject EXCLUDED)
        feature_cols: All possible feature columns
        min_corr: Minimum absolute correlation required per subject
        min_subject_ratio: Minimum fraction of subjects that must agree on direction
                          (0.75 = 75% = at least 3 of 4 subjects must agree)
        top_n: If set, return only top N features by avg correlation
        
    Returns:
        selected_cols: List of selected feature column names
        feature_info: Dict with selection details
    """
    subjects = df_train['subject'].unique()
    n_subjects = len(subjects)
    n_features = len(feature_cols)
    min_subjects = max(2, int(n_subjects * min_subject_ratio))  # At least 2 subjects
    
    # Calculate per-subject correlations
    subject_corrs = {}
    for subj in subjects:
        subj_df = df_train[df_train['subject'] == subj]
        X = subj_df[feature_cols].values
        y = subj_df['borg'].values
        
        corrs = []
        for i in range(n_features):
            if np.std(X[:, i]) > 1e-10:  # Check for constant feature
                r, _ = pearsonr(X[:, i], y)
                corrs.append(r if not np.isnan(r) else 0)
            else:
                corrs.append(0)
        subject_corrs[subj] = np.array(corrs)
    
    # Find consistent features (same direction for most subjects)
    selected_features = []
    
    for i, col in enumerate(feature_cols):
        corrs = [subject_corrs[s][i] for s in subjects]
        
        # Count subjects with meaningful correlation in each direction
        n_positive = sum(1 for c in corrs if c > min_corr)
        n_negative = sum(1 for c in corrs if c < -min_corr)
        
        # Feature is consistent if majority agree on direction
        is_consistent = (n_positive >= min_subjects) or (n_negative >= min_subjects)
        
        if is_consistent:
            # Use only subjects that agree on direction for avg calculation
            if n_positive >= n_negative:
                agreeing_corrs = [c for c in corrs if c > 0]
                direction = 'positive'
            else:
                agreeing_corrs = [abs(c) for c in corrs if c < 0]
                direction = 'negative'
            
            avg_corr = np.mean(agreeing_corrs) if agreeing_corrs else 0
            
            selected_features.append({
                'feature': col,
                'avg_corr': avg_corr,
                'n_agreeing': max(n_positive, n_negative),
                'n_subjects': n_subjects,
                'direction': direction,
                'per_subject': {s: subject_corrs[s][i] for s in subjects}
            })
    
    # Sort by average correlation (among agreeing subjects)
    selected_features.sort(key=lambda x: x['avg_corr'], reverse=True)
    
    # Limit to top_n if specified
    if top_n and len(selected_features) > top_n:
        selected_features = selected_features[:top_n]
    
    selected_cols = [f['feature'] for f in selected_features]
    
    # Analyze modality breakdown
    modality_counts = {
        'IMU': sum(1 for c in selected_cols if c.startswith('acc_')),
        'PPG': sum(1 for c in selected_cols if c.startswith('ppg_')),
        'EDA': sum(1 for c in selected_cols if c.startswith('eda_'))
    }
    
    feature_info = {
        'n_selected': len(selected_cols),
        'n_total': n_features,
        'modality_breakdown': modality_counts,
        'details': selected_features,
        'training_subjects': list(subjects)
    }
    
    return selected_cols, feature_info


def prune_redundant_features(df_train, selected_cols, corr_threshold=0.85):
    """
    Remove highly correlated features to reduce redundancy.
    Keeps the feature with higher target correlation.
    """
    if len(selected_cols) <= 1:
        return selected_cols
    
    X = df_train[selected_cols].values
    y = df_train['borg'].values
    
    # Feature-feature correlation matrix
    corr_matrix = np.corrcoef(X.T)
    np.fill_diagonal(corr_matrix, 0)
    
    # Target correlations
    target_corrs = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(len(selected_cols))])
    
    # Iteratively remove redundant features
    keep_mask = np.ones(len(selected_cols), dtype=bool)
    
    for i in range(len(selected_cols)):
        if not keep_mask[i]:
            continue
        for j in range(i+1, len(selected_cols)):
            if not keep_mask[j]:
                continue
            if abs(corr_matrix[i, j]) > corr_threshold:
                # Remove the one with lower target correlation
                if target_corrs[i] >= target_corrs[j]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    
    return [c for c, keep in zip(selected_cols, keep_mask) if keep]


def select_best_features_loso(df, test_subject, min_corr=0.10, 
                               min_subject_ratio=0.75, top_n=50, prune_redundant=True):
    """
    Main function: Select best features for a LOSO fold.
    
    Uses ONLY training data (test subject excluded) to avoid data leakage.
    
    Args:
        df: Full dataframe with all subjects
        test_subject: Subject to exclude (will be test subject)
        min_corr: Minimum per-subject correlation required
        min_subject_ratio: Fraction of training subjects that must agree on direction
        top_n: Maximum features to select
        prune_redundant: Remove highly correlated features
        
    Returns:
        selected_cols: List of selected feature column names
        feature_info: Selection details
    """
    feature_cols = get_feature_columns(df)
    df_train = df[df['subject'] != test_subject].dropna(subset=['borg'])
    
    # Find best features from training data
    selected_cols, feature_info = find_best_features_for_fold(
        df_train, feature_cols, 
        min_corr=min_corr,
        min_subject_ratio=min_subject_ratio,
        top_n=top_n
    )
    
    # Prune redundant features
    if prune_redundant and len(selected_cols) > 1:
        n_before = len(selected_cols)
        selected_cols = prune_redundant_features(df_train, selected_cols)
        feature_info['n_after_pruning'] = len(selected_cols)
        feature_info['n_pruned'] = n_before - len(selected_cols)
    
    # If no consistent features found, fall back to top correlated features
    if len(selected_cols) == 0:
        print(f"  ⚠️ No consistent features found for test={test_subject}, using top correlated")
        # Use features with highest average correlation across subjects
        selected_cols, feature_info = find_best_features_for_fold(
            df_train, feature_cols,
            min_corr=0.05,  # Lower threshold
            require_all_subjects=False,  # Allow majority
            top_n=top_n
        )
    
    return selected_cols, feature_info


def analyze_feature_selection(df, min_corr=0.10, verbose=True):
    """
    Analyze what features would be selected for each LOSO fold.
    Useful for understanding which modalities/features generalize best.
    """
    df = df.dropna(subset=['borg'])
    subjects = sorted(df['subject'].unique())
    feature_cols = get_feature_columns(df)
    
    if verbose:
        print(f"\n{'='*70}")
        print("DATA-DRIVEN FEATURE SELECTION ANALYSIS")
        print(f"{'='*70}")
        print(f"Total features available: {len(feature_cols)}")
        print(f"  IMU: {sum(1 for c in feature_cols if c.startswith('acc_'))}")
        print(f"  PPG: {sum(1 for c in feature_cols if c.startswith('ppg_'))}")
        print(f"  EDA: {sum(1 for c in feature_cols if c.startswith('eda_'))}")
    
    all_selected = []
    
    for test_subj in subjects:
        selected_cols, info = select_best_features_loso(
            df, test_subj, min_corr=min_corr
        )
        all_selected.append({
            'test_subject': test_subj,
            'n_features': len(selected_cols),
            'features': selected_cols,
            **info['modality_breakdown']
        })
        
        if verbose:
            print(f"\nTest subject: {test_subj}")
            print(f"  Selected: {len(selected_cols)} features")
            print(f"  Modalities: IMU={info['modality_breakdown']['IMU']}, "
                  f"PPG={info['modality_breakdown']['PPG']}, "
                  f"EDA={info['modality_breakdown']['EDA']}")
            if len(selected_cols) > 0 and 'details' in info:
                print(f"  Top 5 features:")
                for feat in info['details'][:5]:
                    print(f"    - {feat['feature']}: avg_r={feat['avg_corr']:.3f} ({feat['direction']})")
    
    # Find features selected in ALL folds (most robust)
    if len(all_selected) > 0:
        common_features = set(all_selected[0]['features'])
        for sel in all_selected[1:]:
            common_features &= set(sel['features'])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FEATURES SELECTED IN ALL FOLDS: {len(common_features)}")
            print(f"{'='*70}")
            for f in sorted(common_features):
                print(f"  - {f}")
    
    return all_selected, common_features


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python best_feature_selection.py <fused_aligned_csv> [min_corr]")
        print("\nAnalyzes which features generalize best across subjects.")
        sys.exit(1)
    
    fused_path = sys.argv[1]
    min_corr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    
    df = pd.read_csv(fused_path)
    analyze_feature_selection(df, min_corr=min_corr)
