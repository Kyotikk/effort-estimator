#!/usr/bin/env python3
"""
Feature Selection with Correlation Pruning + Quality Checks (PCA, loadings, etc.)
Run after fusion/alignment to prepare features for model training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def select_and_prune_features(X_train, y_train, feature_cols, corr_threshold=0.90, top_n=100):
    """
    1. Select top N features by correlation with target
    2. Prune redundant features within each modality
    Returns: pruned feature indices, pruned feature names
    """
    print(f"\n🎯 Feature Selection (top {top_n} by correlation with target)...")
    
    # Get correlations with target
    correlations = np.array([
        np.corrcoef(X_train[:, i], y_train)[0, 1] 
        for i in range(X_train.shape[1])
    ])
    correlations = np.abs(np.nan_to_num(correlations, nan=0))
    top_indices = np.argsort(correlations)[-top_n:][::-1]
    selected_cols = [feature_cols[i] for i in top_indices]
    X_train_sel = X_train[:, top_indices]
    
    eda_count_before = sum(1 for c in selected_cols if c.startswith('eda_'))
    imu_count_before = sum(1 for c in selected_cols if c.startswith('acc_'))
    ppg_count_before = sum(1 for c in selected_cols if c.startswith('ppg_'))
    print(f"  ✓ Before pruning - EDA: {eda_count_before}, IMU: {imu_count_before}, PPG: {ppg_count_before}")
    
    # Prune redundant features within each modality
    print(f"\n🔪 Pruning redundant features (correlation threshold={corr_threshold})...")
    pruned_indices = []
    pruned_cols = []
    
    for modality_prefix in ['eda_', 'acc_', 'ppg_']:
        mod_mask = [i for i, c in enumerate(selected_cols) if c.startswith(modality_prefix)]
        if len(mod_mask) <= 1:
            pruned_indices.extend([top_indices[i] for i in mod_mask])
            pruned_cols.extend([selected_cols[i] for i in mod_mask])
            continue
        
        X_mod = X_train_sel[:, mod_mask]
        corr_matrix = np.abs(np.corrcoef(X_mod.T))
        np.fill_diagonal(corr_matrix, 0.0)
        
        keep = set(range(len(mod_mask)))
        while True:
            sub = corr_matrix[np.ix_(list(keep), list(keep))]
            max_val = sub.max() if sub.size > 0 else 0
            if max_val < corr_threshold:
                break
            keep_list = sorted(list(keep))
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            fi, fj = keep_list[i], keep_list[j]
            
            corr_fi = np.abs(np.corrcoef(X_mod[:, fi], y_train)[0, 1])
            corr_fj = np.abs(np.corrcoef(X_mod[:, fj], y_train)[0, 1])
            drop_idx = fi if corr_fi <= corr_fj else fj
            keep.discard(drop_idx)
        
        for idx in sorted(keep):
            pruned_indices.append(top_indices[mod_mask[idx]])
            pruned_cols.append(selected_cols[mod_mask[idx]])
    
    eda_count = sum(1 for c in pruned_cols if c.startswith('eda_'))
    imu_count = sum(1 for c in pruned_cols if c.startswith('acc_'))
    ppg_count = sum(1 for c in pruned_cols if c.startswith('ppg_'))
    print(f"  ✓ After pruning - EDA: {eda_count}, IMU: {imu_count}, PPG: {ppg_count}")
    print(f"  ✓ Total: {len(pruned_cols)} features (from {top_n})")
    
    return pruned_indices, pruned_cols


def perform_pca_analysis(X_pruned, feature_names):
    """Perform PCA on pruned features"""
    print(f"\n📊 PCA Analysis...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pruned)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    explained_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(evr))],
        'explained_variance_ratio': evr,
        'cumulative_explained_variance': cum
    })
    
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )
    
    # Top loadings summary
    pca_topk = 8
    pca_maxpcs = 8
    pcs = [c for c in loadings_df.columns if c.startswith('PC')][:pca_maxpcs]
    top_rows = []
    for pc in pcs:
        s = loadings_df[pc].abs().sort_values(ascending=False).head(pca_topk)
        for feat, val in s.items():
            top_rows.append({'PC': pc, 'feature': feat, 'abs_loading': float(val)})
    
    top_loadings_df = pd.DataFrame(top_rows)
    
    # Variance targets
    pca_variance_targets = (0.90, 0.95, 0.99)
    pcs_for_targets = {}
    for t in pca_variance_targets:
        k = int(np.searchsorted(cum, t) + 1)
        pcs_for_targets[f'pcs_for_{int(t*100)}pct'] = k
    
    print(f"  ✓ PCA variance targets: {pcs_for_targets}")
    
    return explained_df, loadings_df, top_loadings_df, pcs_for_targets


def save_feature_selection_results(output_dir, feature_names, explained_df, loadings_df, top_loadings_df):
    """Save all feature selection and QC outputs"""
    print(f"\n💾 Saving feature selection and QC outputs...")
    ensure_dir(output_dir)
    
    # Feature lists
    pd.Series(feature_names).to_csv(
        Path(output_dir) / "features_selected_pruned.csv", 
        index=False, header=False
    )
    
    # PCA outputs
    explained_df.to_csv(Path(output_dir) / "pca_variance_explained.csv", index=False)
    loadings_df.to_csv(Path(output_dir) / "pca_loadings.csv")
    top_loadings_df.to_csv(Path(output_dir) / "pca_top_loadings.csv", index=False)
    
    print(f"  ✓ Saved: features_selected_pruned.csv")
    print(f"  ✓ Saved: pca_variance_explained.csv")
    print(f"  ✓ Saved: pca_loadings.csv")
    print(f"  ✓ Saved: pca_top_loadings.csv")


def main(fused_aligned_path, output_dir, win_sec=10.0):
    """
    Main feature selection + pruning + QC pipeline
    
    Args:
        fused_aligned_path: Path to fused_aligned_*.csv file
        output_dir: Where to save outputs
        win_sec: Window length (for logging)
    """
    print("\n" + "="*100)
    print(f"FEATURE SELECTION + QUALITY CHECKS ({win_sec:.1f}s windows)")
    print("="*100)
    
    # Load data
    print(f"\n📂 Loading data from: {fused_aligned_path}")
    df = pd.read_csv(fused_aligned_path)
    
    # Determine target variable (HRV recovery rate if available, else Borg)
    target_col = None
    if 'hrv_recovery_rate' in df.columns:
        target_col = 'hrv_recovery_rate'
        print("  ✓ Using HRV Recovery Rate as target variable (primary)")
    elif 'borg' in df.columns:
        target_col = 'borg'
        print("  ✓ Using Borg RPE Scale as target variable (fallback)")
    else:
        raise ValueError("No target variable found (need 'hrv_recovery_rate' or 'borg')")
    
    df_labeled = df.dropna(subset=[target_col]).copy()
    print(f"  ✓ Loaded {len(df_labeled)} labeled samples")
    
    # Filter metadata
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
        "hrv_recovery_rate", "hrv_baseline", "hrv_effort", "hrv_recovery", "activity_borg"
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            return True
        return False
    
    feature_cols = [col for col in df_labeled.columns if not is_metadata(col)]
    X = df_labeled[feature_cols].values
    y = df_labeled[target_col].values
    print(f"  ✓ {len(feature_cols)} features (after metadata removal)")
    
    # Train-test split
    print(f"\n🔀 Train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Feature selection + pruning
    pruned_indices, pruned_cols = select_and_prune_features(
        X_train, y_train, feature_cols, 
        corr_threshold=0.90, 
        top_n=100
    )
    
    # PCA analysis
    X_train_pruned = X_train[:, pruned_indices]
    explained_df, loadings_df, top_loadings_df, pcs_for_targets = perform_pca_analysis(
        X_train_pruned, pruned_cols
    )
    
    # Save outputs
    save_feature_selection_results(
        output_dir, 
        pruned_cols, 
        explained_df, 
        loadings_df, 
        top_loadings_df
    )
    
    print("\n" + "="*100)
    print("✅ FEATURE SELECTION + QC COMPLETE")
    print("="*100)
    
    return pruned_indices, pruned_cols


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_selection_and_qc.py <fused_aligned_path> [output_dir] [win_sec]")
        sys.exit(1)
    
    fused_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./feature_selection_output"
    win_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    
    main(fused_path, out_dir, win_sec)
