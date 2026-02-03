#!/usr/bin/env python3
"""
Compare window sizes: 5s, 10s using existing fused_aligned files.
Generate 10s fused with proper tolerance, then train.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Paths
SUBJECT_DIRS = {
    "sim_elderly3": "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3",
    "sim_elderly4": "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4",
    "sim_elderly5": "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5",
}

OUTPUT_DIR = Path("/Users/pascalschlegel/data/interim/elderly_combined/window_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fuse_and_get_aligned(subject_dir, win_sec, tolerance_sec):
    """Fuse features and get aligned data with Borg labels."""
    from ml.fusion.fuse_windows import fuse_feature_tables
    
    base = Path(subject_dir)
    
    # Load all feature tables
    paths = {
        'eda': f'eda/eda_features_{win_sec:.1f}s.csv',
        'eda_advanced': f'eda/eda_advanced_features_{win_sec:.1f}s.csv',
        'imu_bioz': f'imu_bioz/imu_features_{win_sec:.1f}s.csv',
        'imu_wrist': f'imu_wrist/imu_features_{win_sec:.1f}s.csv',
        'ppg_green': f'ppg_green/ppg_green_features_{win_sec:.1f}s.csv',
        'ppg_green_hrv': f'ppg_green/ppg_green_hrv_features_{win_sec:.1f}s.csv',
        'ppg_infra': f'ppg_infra/ppg_infra_features_{win_sec:.1f}s.csv',
        'ppg_infra_hrv': f'ppg_infra/ppg_infra_hrv_features_{win_sec:.1f}s.csv',
        'ppg_red': f'ppg_red/ppg_red_features_{win_sec:.1f}s.csv',
        'ppg_red_hrv': f'ppg_red/ppg_red_hrv_features_{win_sec:.1f}s.csv',
    }
    
    tables = []
    for name, rel_path in paths.items():
        path = base / rel_path
        if path.exists():
            df = pd.read_csv(path)
            if 't_center' in df.columns and len(df) > 0:
                tables.append(df)
    
    if len(tables) < 2:
        return None
    
    # Fuse with proper tolerance
    fused = fuse_feature_tables(tables, join_col='t_center', tolerance_sec=tolerance_sec)
    
    # Now check if there's a fused_aligned file with Borg labels
    aligned_path = base / f"fused_aligned_{win_sec:.1f}s.csv"
    if aligned_path.exists():
        aligned = pd.read_csv(aligned_path)
        # The aligned file has Borg labels - merge them
        if 'borg' in aligned.columns:
            # Merge on t_center
            fused = fused.merge(
                aligned[['t_center', 'borg', 'label']].drop_duplicates(),
                on='t_center',
                how='left'
            )
    
    return fused


def select_features(df, target_col='borg', top_n=100, prune_threshold=0.90):
    """Select top features by correlation, then prune redundant."""
    meta_cols = ['t_center', 't_start', 't_end', 'start_idx', 'end_idx', 'window_id',
                 'borg', 'subject', 'label', 'modality', 'valid', 'n_samples', 'win_sec',
                 'valid_r', 'n_samples_r', 'win_sec_r']
    
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    # Remove constant columns
    valid_cols = []
    for c in feat_cols:
        if df[c].notna().sum() > 10:
            std = df[c].std()
            if pd.notna(std) and std > 1e-10:
                valid_cols.append(c)
    
    labeled = df[df[target_col].notna()].copy()
    if len(labeled) < 50:
        return valid_cols[:top_n]
    
    # Correlations
    correlations = {}
    for c in valid_cols:
        try:
            vals = labeled[c].fillna(0)
            r, _ = pearsonr(vals, labeled[target_col])
            if np.isfinite(r):
                correlations[c] = abs(r)
        except:
            pass
    
    sorted_feats = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, r in sorted_feats[:top_n]]
    
    # Prune
    if len(top_feats) < 2:
        return top_feats
    
    feat_data = labeled[top_feats].fillna(0)
    corr_matrix = feat_data.corr().abs()
    
    selected = []
    for feat in top_feats:
        dominated = False
        for sel in selected:
            if feat in corr_matrix.index and sel in corr_matrix.columns:
                if corr_matrix.loc[feat, sel] > prune_threshold:
                    dominated = True
                    break
        if not dominated:
            selected.append(feat)
    
    return selected


def train_and_evaluate(df, feature_cols, target_col='borg', group_col='label'):
    """Train models with GroupKFold CV."""
    labeled = df[df[target_col].notna()].copy()
    
    if len(labeled) < 50:
        return None
    
    X = labeled[feature_cols].fillna(0).values
    y = labeled[target_col].values
    
    # Groups
    if group_col in labeled.columns:
        groups = labeled[group_col].astype(str).factorize()[0]
    else:
        groups = np.arange(len(labeled))
    
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    
    if n_splits < 2:
        return None
    
    gkf = GroupKFold(n_splits=n_splits)
    
    results = {'xgboost': {'y_true': [], 'y_pred': []},
               'ridge': {'y_true': [], 'y_pred': []}}
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          verbosity=0)
        xgb.fit(X_train_sc, y_train)
        pred_xgb = xgb.predict(X_test_sc)
        results['xgboost']['y_true'].extend(y_test)
        results['xgboost']['y_pred'].extend(pred_xgb)
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        pred_ridge = ridge.predict(X_test_sc)
        results['ridge']['y_true'].extend(y_test)
        results['ridge']['y_pred'].extend(pred_ridge)
    
    metrics = {}
    for model in ['xgboost', 'ridge']:
        y_true = np.array(results[model]['y_true'])
        y_pred = np.array(results[model]['y_pred'])
        r, p = pearsonr(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        metrics[model] = {'r': r, 'p': p, 'rmse': rmse, 'mae': mae}
    
    return metrics


def run_comparison():
    """Compare 5s and 10s windows."""
    
    # First, load the existing 5s combined data
    print("="*70)
    print("Loading existing 5s aligned data...")
    print("="*70)
    
    df_5s = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
    n_labeled_5s = df_5s['borg'].notna().sum()
    print(f"5s: {len(df_5s)} total, {n_labeled_5s} labeled")
    
    # Select features for 5s
    selected_5s = select_features(df_5s, top_n=100, prune_threshold=0.90)
    print(f"5s selected features: {len(selected_5s)}")
    
    # Train 5s
    metrics_5s = train_and_evaluate(df_5s, selected_5s)
    
    print(f"\n5s Results:")
    print(f"  XGBoost: r={metrics_5s['xgboost']['r']:.3f}, RMSE={metrics_5s['xgboost']['rmse']:.3f}, MAE={metrics_5s['xgboost']['mae']:.3f}")
    print(f"  Ridge:   r={metrics_5s['ridge']['r']:.3f}, RMSE={metrics_5s['ridge']['rmse']:.3f}, MAE={metrics_5s['ridge']['mae']:.3f}")
    
    # Now do 10s with proper tolerance
    print("\n" + "="*70)
    print("Processing 10s windows with proper tolerance (5s)...")
    print("="*70)
    
    dfs_10s = []
    for subject, subject_dir in SUBJECT_DIRS.items():
        print(f"\n  {subject}...")
        
        # Check if 10s features exist
        test_file = Path(subject_dir) / "eda/eda_features_10.0s.csv"
        if not test_file.exists():
            print(f"    No 10s features")
            continue
        
        # Check for aligned file
        aligned_path = Path(subject_dir) / "fused_aligned_10.0s.csv"
        if aligned_path.exists():
            df = pd.read_csv(aligned_path)
            df['subject'] = subject
            n_labeled = df['borg'].notna().sum() if 'borg' in df.columns else 0
            print(f"    Loaded aligned: {len(df)} rows, {n_labeled} labeled")
            dfs_10s.append(df)
        else:
            # Try to fuse manually
            fused = fuse_and_get_aligned(subject_dir, 10.0, 5.0)
            if fused is not None and len(fused) > 0:
                fused['subject'] = subject
                n_labeled = fused['borg'].notna().sum() if 'borg' in fused.columns else 0
                print(f"    Fused: {len(fused)} rows, {n_labeled} labeled")
                dfs_10s.append(fused)
    
    if not dfs_10s:
        print("No 10s data available")
        return
    
    # Combine 10s - need to handle different columns
    common_cols = set(dfs_10s[0].columns)
    for df in dfs_10s[1:]:
        common_cols &= set(df.columns)
    common_cols = list(common_cols)
    
    df_10s = pd.concat([df[common_cols] for df in dfs_10s], ignore_index=True)
    n_labeled_10s = df_10s['borg'].notna().sum() if 'borg' in df_10s.columns else 0
    print(f"\n10s combined: {len(df_10s)} total, {n_labeled_10s} labeled")
    
    if n_labeled_10s < 50:
        print("Not enough labeled 10s data")
        return
    
    # Select features for 10s
    selected_10s = select_features(df_10s, top_n=100, prune_threshold=0.90)
    print(f"10s selected features: {len(selected_10s)}")
    
    # Train 10s
    metrics_10s = train_and_evaluate(df_10s, selected_10s)
    
    if metrics_10s:
        print(f"\n10s Results:")
        print(f"  XGBoost: r={metrics_10s['xgboost']['r']:.3f}, RMSE={metrics_10s['xgboost']['rmse']:.3f}, MAE={metrics_10s['xgboost']['mae']:.3f}")
        print(f"  Ridge:   r={metrics_10s['ridge']['r']:.3f}, RMSE={metrics_10s['ridge']['rmse']:.3f}, MAE={metrics_10s['ridge']['mae']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Window':<10} {'Samples':<10} {'Features':<10} {'XGB r':<10} {'Ridge r':<10} {'XGB MAE':<10}")
    print("-"*70)
    print(f"5s{'':<8} {n_labeled_5s:<10} {len(selected_5s):<10} {metrics_5s['xgboost']['r']:.3f}{'':<6} {metrics_5s['ridge']['r']:.3f}{'':<6} {metrics_5s['xgboost']['mae']:.3f}")
    if metrics_10s:
        print(f"10s{'':<7} {n_labeled_10s:<10} {len(selected_10s):<10} {metrics_10s['xgboost']['r']:.3f}{'':<6} {metrics_10s['ridge']['r']:.3f}{'':<6} {metrics_10s['xgboost']['mae']:.3f}")


if __name__ == "__main__":
    run_comparison()
