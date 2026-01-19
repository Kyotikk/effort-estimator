#!/usr/bin/env python3
"""
Feature Selection with Correlation Pruning + Quality Checks (copied from pascal_update)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def select_and_prune_features(X_train, y_train, feature_cols, corr_threshold=0.90, top_n=100):
    """
    1. Select top N features by correlation with target
    2. Prune redundant features within each modality
    Returns: pruned feature indices, pruned feature names
    """
    print(f"\n🎯 Feature Selection (top {top_n} by correlation with target)...")

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
