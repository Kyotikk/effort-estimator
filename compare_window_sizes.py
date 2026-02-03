#!/usr/bin/env python3
"""
Compare window sizes: 5s, 10s, 30s for effort estimation.
Uses existing features where available, generates new ones for 30s.
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
DATA_ROOTS = {
    "sim_elderly3": "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3",
    "sim_elderly4": "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4",
    "sim_elderly5": "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5",
}

ADL_PATHS = {
    "sim_elderly3": "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv",
    "sim_elderly4": "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/scai_app/ADLs_1.csv",
    "sim_elderly5": "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/scai_app/ADLs_1-5.csv",
}

OUTPUT_DIR = Path("/Users/pascalschlegel/data/interim/elderly_combined/window_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_adl_labels(adl_path):
    """Load ADL labels with Borg scores."""
    df = pd.read_csv(adl_path)
    
    # Find Borg column
    borg_col = None
    for col in df.columns:
        if 'borg' in col.lower():
            borg_col = col
            break
    
    if borg_col is None:
        return None
    
    # Get time columns
    start_col = 'start_time' if 'start_time' in df.columns else 'startTime'
    end_col = 'end_time' if 'end_time' in df.columns else 'endTime'
    
    labels = []
    for _, row in df.iterrows():
        try:
            start = pd.to_datetime(row[start_col]).timestamp()
            end = pd.to_datetime(row[end_col]).timestamp()
            borg = float(row[borg_col]) if pd.notna(row[borg_col]) else np.nan
            label = row.get('label', row.get('activity', 'unknown'))
            labels.append({'start': start, 'end': end, 'borg': borg, 'label': label})
        except:
            continue
    
    return labels


def align_features_with_borg(features_df, labels, subject):
    """Align feature windows with Borg labels."""
    if labels is None or len(labels) == 0:
        return features_df.assign(borg=np.nan, subject=subject, label='unknown')
    
    features_df = features_df.copy()
    features_df['borg'] = np.nan
    features_df['label'] = 'unknown'
    features_df['subject'] = subject
    
    for i, row in features_df.iterrows():
        t = row['t_center']
        for lab in labels:
            if lab['start'] <= t <= lab['end']:
                features_df.loc[i, 'borg'] = lab['borg']
                features_df.loc[i, 'label'] = lab['label']
                break
    
    return features_df


def fuse_features_for_window(subject_dir, win_sec, tolerance_sec=5.0):
    """Fuse all modality features for a given window size."""
    from ml.fusion.fuse_windows import fuse_feature_tables
    
    base = Path(subject_dir)
    
    # Define all feature file paths
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
    
    # Load available tables
    tables = []
    loaded = []
    for name, rel_path in paths.items():
        path = base / rel_path
        if path.exists():
            df = pd.read_csv(path)
            if 't_center' in df.columns and len(df) > 0:
                tables.append(df)
                loaded.append(name)
    
    if len(tables) < 2:
        print(f"    Not enough feature tables for {win_sec}s (found {len(tables)})")
        return None
    
    print(f"    Loaded {len(tables)} modalities: {loaded[:5]}...")
    
    # Fuse
    fused = fuse_feature_tables(tables, join_col='t_center', tolerance_sec=tolerance_sec)
    print(f"    Fused: {len(fused)} rows, {len(fused.columns)} features")
    
    return fused


def select_features(df, target_col='borg', top_n=100, prune_threshold=0.90):
    """Select top features by correlation, then prune redundant ones."""
    # Get feature columns (exclude meta)
    meta_cols = ['t_center', 't_start', 't_end', 'start_idx', 'end_idx', 'window_id',
                 'borg', 'subject', 'label', 'modality', 'valid', 'n_samples', 'win_sec',
                 'valid_r', 'n_samples_r', 'win_sec_r']
    feat_cols = [c for c in df.columns if c not in meta_cols and not c.endswith('_r')]
    
    # Remove constant and all-NaN columns
    valid_cols = []
    for c in feat_cols:
        if df[c].notna().sum() > 10 and df[c].std() > 1e-10:
            valid_cols.append(c)
    
    # Calculate correlations with target
    labeled = df[df[target_col].notna()].copy()
    if len(labeled) < 50:
        return valid_cols[:top_n]  # Not enough data
    
    correlations = {}
    for c in valid_cols:
        try:
            r, _ = pearsonr(labeled[c].fillna(0), labeled[target_col])
            if np.isfinite(r):
                correlations[c] = abs(r)
        except:
            pass
    
    # Sort by correlation
    sorted_feats = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, r in sorted_feats[:top_n]]
    
    # Prune redundant features
    if len(top_feats) < 2:
        return top_feats
    
    feat_data = labeled[top_feats].fillna(0)
    corr_matrix = feat_data.corr().abs()
    
    selected = []
    for feat in top_feats:
        # Check if highly correlated with already selected
        dominated = False
        for sel in selected:
            if corr_matrix.loc[feat, sel] > prune_threshold:
                dominated = True
                break
        if not dominated:
            selected.append(feat)
    
    return selected


def train_and_evaluate(df, feature_cols, target_col='borg', group_col='label'):
    """Train XGBoost and Ridge with GroupKFold CV."""
    # Filter to labeled data
    labeled = df[df[target_col].notna()].copy()
    
    if len(labeled) < 50:
        return None
    
    X = labeled[feature_cols].fillna(0).values
    y = labeled[target_col].values
    
    # Create groups for CV
    groups = labeled[group_col].astype('category').cat.codes.values
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
        
        # Standardize
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
    
    # Calculate metrics
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
    """Run comparison for 5s, 10s, 30s windows."""
    
    window_configs = [
        {'win_sec': 5.0, 'tolerance': 2.5},
        {'win_sec': 10.0, 'tolerance': 5.0},  # Fixed tolerance!
        {'win_sec': 30.0, 'tolerance': 15.0},
    ]
    
    all_results = []
    
    for config in window_configs:
        win_sec = config['win_sec']
        tolerance = config['tolerance']
        
        print(f"\n{'='*60}")
        print(f"WINDOW SIZE: {win_sec}s (tolerance: {tolerance}s)")
        print('='*60)
        
        combined_dfs = []
        
        for subject, subject_dir in DATA_ROOTS.items():
            print(f"\n  Processing {subject}...")
            
            # Check if features exist for this window size
            base = Path(subject_dir)
            test_file = base / f"eda/eda_features_{win_sec:.1f}s.csv"
            
            if not test_file.exists():
                print(f"    Features for {win_sec}s don't exist, skipping...")
                continue
            
            # Fuse features
            fused = fuse_features_for_window(subject_dir, win_sec, tolerance)
            
            if fused is None or len(fused) == 0:
                print(f"    No fused features for {subject}")
                continue
            
            # Load labels and align
            labels = load_adl_labels(ADL_PATHS[subject])
            aligned = align_features_with_borg(fused, labels, subject)
            
            n_labeled = aligned['borg'].notna().sum()
            print(f"    Aligned: {len(aligned)} rows, {n_labeled} with Borg labels")
            
            combined_dfs.append(aligned)
        
        if not combined_dfs:
            print(f"  No data for {win_sec}s windows")
            continue
        
        # Combine all subjects
        combined = pd.concat(combined_dfs, ignore_index=True)
        n_total = len(combined)
        n_labeled = combined['borg'].notna().sum()
        print(f"\n  Combined: {n_total} total, {n_labeled} labeled")
        
        # Feature selection
        print("  Selecting features...")
        selected_feats = select_features(combined, top_n=100, prune_threshold=0.90)
        print(f"  Selected {len(selected_feats)} features")
        
        if len(selected_feats) < 5:
            print(f"  Not enough features for {win_sec}s")
            continue
        
        # Train and evaluate
        print("  Training models...")
        metrics = train_and_evaluate(combined, selected_feats)
        
        if metrics is None:
            print(f"  Training failed for {win_sec}s")
            continue
        
        # Store results
        result = {
            'window_sec': win_sec,
            'n_samples': n_labeled,
            'n_features': len(selected_feats),
            'xgb_r': metrics['xgboost']['r'],
            'xgb_rmse': metrics['xgboost']['rmse'],
            'xgb_mae': metrics['xgboost']['mae'],
            'ridge_r': metrics['ridge']['r'],
            'ridge_rmse': metrics['ridge']['rmse'],
            'ridge_mae': metrics['ridge']['mae'],
        }
        all_results.append(result)
        
        print(f"\n  === Results for {win_sec}s ===")
        print(f"  XGBoost: r={metrics['xgboost']['r']:.3f}, RMSE={metrics['xgboost']['rmse']:.3f}, MAE={metrics['xgboost']['mae']:.3f}")
        print(f"  Ridge:   r={metrics['ridge']['r']:.3f}, RMSE={metrics['ridge']['rmse']:.3f}, MAE={metrics['ridge']['mae']:.3f}")
    
    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Window':<10} {'Samples':<10} {'Features':<10} {'XGB r':<10} {'Ridge r':<10} {'XGB MAE':<10} {'Ridge MAE':<10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['window_sec']:.0f}s{'':<7} {r['n_samples']:<10} {r['n_features']:<10} {r['xgb_r']:.3f}{'':<6} {r['ridge_r']:.3f}{'':<6} {r['xgb_mae']:.3f}{'':<6} {r['ridge_mae']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "window_comparison_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'window_comparison_results.csv'}")


if __name__ == "__main__":
    run_comparison()
