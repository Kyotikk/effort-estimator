"""
Feature selection module - selects and prunes features.

Callable functions:
- select_features()  # Main feature selection function
"""

import pandas as pd
from pathlib import Path
from ml.feature_selection_and_qc import (
    select_and_prune_features,
    perform_pca_analysis,
    save_feature_selection_results
)


def select_features(df, target_col='borg', corr_threshold=0.90, top_n=100):
    """
    Select features using correlation-based pruning.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        corr_threshold: Correlation threshold for pruning
        top_n: Top N features to consider before pruning
        
    Returns:
        tuple: (selected_features_list, X_selected, y)
    """
    # Remove metadata columns
    skip_cols = {
        'window_id', 'start_idx', 'end_idx', 'valid',
        't_start', 't_center', 't_end', 'n_samples', 'win_sec',
        'modality', 'subject', target_col,
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        if col.endswith('_r') or any(col.endswith(f'_r.{i}') for i in range(1, 10)):
            return True
        return False
    
    feature_cols = [col for col in df.columns if not is_metadata(col)]
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"  Features before selection: {len(feature_cols)}")
    
    # Select top features by correlation + prune
    pruned_indices, pruned_cols = select_and_prune_features(
        X, y, feature_cols,
        corr_threshold=corr_threshold,
        top_n=top_n
    )
    
    X_selected = X[:, pruned_indices]
    
    print(f"  Features after selection: {len(pruned_cols)}")
    
    return pruned_cols, X_selected, y


def save_feature_selection_outputs(output_dir, df, selected_cols, window_length):
    """
    Save selected features to CSV and generate PCA analysis.
    
    Args:
        output_dir: Directory to save outputs
        df: Original DataFrame
        selected_cols: List of selected feature column names
        window_length: Window length for naming
        
    Returns:
        Path to selected features CSV
    """
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get labeled data
    df_labeled = df.dropna(subset=['borg']).copy()
    X = df_labeled[selected_cols].values
    y = df_labeled['borg'].values
    
    # PCA analysis
    explained_df, loadings_df, top_loadings_df, pcs = perform_pca_analysis(X, selected_cols)
    
    # Save selection results
    qc_dir = output_path / f"qc_{window_length:.1f}s"
    save_feature_selection_results(
        str(qc_dir),
        selected_cols,
        explained_df,
        loadings_df,
        top_loadings_df
    )
    
    # Save selected features to CSV
    features_file = output_path / f"features_selected_pruned.csv"
    pd.DataFrame({'feature': selected_cols}).to_csv(features_file, index=False)
    
    return features_file
