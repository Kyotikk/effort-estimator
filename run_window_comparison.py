#!/usr/bin/env python3
"""
Properly compare window sizes using the existing pipeline code.
Re-runs the full alignment and training for 10s windows.
"""

import sys
sys.path.insert(0, '/Users/pascalschlegel/effort-estimator')

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

from ml.fusion.fuse_windows import fuse_feature_tables
from ml.targets.run_target_alignment import align_features_with_adl

# Configuration
SUBJECT_CONFIGS = {
    "sim_elderly3": {
        "base": "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3",
        "adl": "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv",
    },
    "sim_elderly4": {
        "base": "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4",
        "adl": "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/scai_app/ADLs_1.csv",
    },
    "sim_elderly5": {
        "base": "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5",
        "adl": "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/scai_app/ADLs_1-5.csv",
    },
}

OUTPUT_DIR = Path("/Users/pascalschlegel/data/interim/elderly_combined")


def fuse_features(subject_dir, win_sec, tolerance_sec):
    """Fuse all modality features."""
    base = Path(subject_dir)
    
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
    
    fused = fuse_feature_tables(tables, join_col='t_center', tolerance_sec=tolerance_sec)
    return fused


def process_window_size(win_sec, tolerance_sec):
    """Process all subjects for a given window size."""
    print(f"\n{'='*70}")
    print(f"WINDOW SIZE: {win_sec}s (tolerance: {tolerance_sec}s)")
    print('='*70)
    
    all_aligned = []
    
    for subject, config in SUBJECT_CONFIGS.items():
        print(f"\n  Processing {subject}...")
        
        base = Path(config['base'])
        
        # Check if features exist
        test_file = base / f"eda/eda_features_{win_sec:.1f}s.csv"
        if not test_file.exists():
            print(f"    No {win_sec}s features found")
            continue
        
        # Fuse
        fused = fuse_features(config['base'], win_sec, tolerance_sec)
        if fused is None or len(fused) == 0:
            print(f"    Fusion failed")
            continue
        
        print(f"    Fused: {len(fused)} rows")
        
        # Save fused
        fused_path = base / f"fused_features_{win_sec:.1f}s.csv"
        fused.to_csv(fused_path, index=False)
        
        # Align with Borg labels using existing function
        aligned_path = base / f"fused_aligned_{win_sec:.1f}s.csv"
        
        try:
            align_features_with_adl(
                features_path=str(fused_path),
                adl_path=config['adl'],
                out_path=str(aligned_path),
            )
            
            aligned = pd.read_csv(aligned_path)
            aligned['subject'] = subject
            
            n_labeled = aligned['borg'].notna().sum() if 'borg' in aligned.columns else 0
            print(f"    Aligned: {len(aligned)} rows, {n_labeled} with Borg")
            
            all_aligned.append(aligned)
            
        except Exception as e:
            print(f"    Alignment failed: {e}")
            continue
    
    if not all_aligned:
        return None
    
    # Combine
    # Find common columns
    common_cols = set(all_aligned[0].columns)
    for df in all_aligned[1:]:
        common_cols &= set(df.columns)
    common_cols = list(common_cols)
    
    combined = pd.concat([df[common_cols] for df in all_aligned], ignore_index=True)
    
    return combined


def select_features_properly(df, top_n=100, prune_threshold=0.90):
    """Select features using correlation with Borg."""
    meta_cols = ['t_center', 't_start', 't_end', 'start_idx', 'end_idx', 'window_id',
                 'borg', 'subject', 'label', 'modality', 'valid', 'n_samples', 'win_sec',
                 'valid_r', 'n_samples_r', 'win_sec_r', 'activity_id']
    
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    # Filter valid columns
    valid_cols = []
    for c in feat_cols:
        try:
            if df[c].notna().sum() > 10 and df[c].std() > 1e-10:
                valid_cols.append(c)
        except:
            pass
    
    labeled = df[df['borg'].notna()].copy()
    if len(labeled) < 50:
        return valid_cols[:top_n]
    
    # Correlations
    correlations = {}
    for c in valid_cols:
        try:
            r, _ = pearsonr(labeled[c].fillna(0), labeled['borg'])
            if np.isfinite(r):
                correlations[c] = abs(r)
        except:
            pass
    
    sorted_feats = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, r in sorted_feats[:top_n]]
    
    # Prune redundant
    if len(top_feats) < 2:
        return top_feats
    
    feat_data = labeled[top_feats].fillna(0)
    corr_matrix = feat_data.corr().abs()
    
    selected = []
    for feat in top_feats:
        dominated = False
        for sel in selected:
            try:
                if corr_matrix.loc[feat, sel] > prune_threshold:
                    dominated = True
                    break
            except:
                pass
        if not dominated:
            selected.append(feat)
    
    return selected


def train_and_evaluate(df, feature_cols):
    """Train XGBoost and Ridge with proper GroupKFold."""
    labeled = df[df['borg'].notna()].copy()
    
    if len(labeled) < 50:
        return None
    
    X = labeled[feature_cols].fillna(0).values
    y = labeled['borg'].values
    
    # Create activity groups
    if 'label' in labeled.columns:
        # Create unique activity ID per subject
        labeled['activity_id'] = labeled['subject'].astype(str) + '_' + labeled['label'].astype(str)
        groups = labeled['activity_id'].factorize()[0]
    else:
        groups = labeled['subject'].factorize()[0]
    
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    
    if n_splits < 2:
        n_splits = 2
    
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
        results['xgboost']['y_true'].extend(y_test)
        results['xgboost']['y_pred'].extend(xgb.predict(X_test_sc))
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        results['ridge']['y_true'].extend(y_test)
        results['ridge']['y_pred'].extend(ridge.predict(X_test_sc))
    
    metrics = {}
    for model in ['xgboost', 'ridge']:
        y_true = np.array(results[model]['y_true'])
        y_pred = np.array(results[model]['y_pred'])
        r, p = pearsonr(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        metrics[model] = {'r': r, 'rmse': rmse, 'mae': mae}
    
    return metrics


def main():
    results = []
    
    # Process 5s (using existing data)
    print("\n" + "="*70)
    print("5s WINDOWS (existing aligned data)")
    print("="*70)
    
    df_5s = pd.read_csv(OUTPUT_DIR / "elderly_aligned_5.0s.csv")
    n_labeled_5s = df_5s['borg'].notna().sum()
    print(f"Loaded: {len(df_5s)} total, {n_labeled_5s} labeled")
    
    # Load previously selected features
    selected_5s_file = OUTPUT_DIR / "qc_5.0s/features_selected_pruned.csv"
    if selected_5s_file.exists():
        selected_5s = pd.read_csv(selected_5s_file, header=None)[0].tolist()
    else:
        selected_5s = select_features_properly(df_5s)
    
    print(f"Features: {len(selected_5s)}")
    
    metrics_5s = train_and_evaluate(df_5s, selected_5s)
    print(f"\nResults:")
    print(f"  XGBoost: r={metrics_5s['xgboost']['r']:.3f}, RMSE={metrics_5s['xgboost']['rmse']:.3f}, MAE={metrics_5s['xgboost']['mae']:.3f}")
    print(f"  Ridge:   r={metrics_5s['ridge']['r']:.3f}, RMSE={metrics_5s['ridge']['rmse']:.3f}, MAE={metrics_5s['ridge']['mae']:.3f}")
    
    results.append({
        'window': '5s', 'samples': n_labeled_5s, 'features': len(selected_5s),
        'xgb_r': metrics_5s['xgboost']['r'], 'xgb_mae': metrics_5s['xgboost']['mae'],
        'ridge_r': metrics_5s['ridge']['r'], 'ridge_mae': metrics_5s['ridge']['mae']
    })
    
    # Process 10s
    df_10s = process_window_size(10.0, 5.0)
    
    if df_10s is not None:
        n_labeled_10s = df_10s['borg'].notna().sum()
        print(f"\n10s combined: {len(df_10s)} total, {n_labeled_10s} labeled")
        
        if n_labeled_10s >= 50:
            selected_10s = select_features_properly(df_10s)
            print(f"Features: {len(selected_10s)}")
            
            metrics_10s = train_and_evaluate(df_10s, selected_10s)
            if metrics_10s:
                print(f"\nResults:")
                print(f"  XGBoost: r={metrics_10s['xgboost']['r']:.3f}, RMSE={metrics_10s['xgboost']['rmse']:.3f}, MAE={metrics_10s['xgboost']['mae']:.3f}")
                print(f"  Ridge:   r={metrics_10s['ridge']['r']:.3f}, RMSE={metrics_10s['ridge']['rmse']:.3f}, MAE={metrics_10s['ridge']['mae']:.3f}")
                
                results.append({
                    'window': '10s', 'samples': n_labeled_10s, 'features': len(selected_10s),
                    'xgb_r': metrics_10s['xgboost']['r'], 'xgb_mae': metrics_10s['xgboost']['mae'],
                    'ridge_r': metrics_10s['ridge']['r'], 'ridge_mae': metrics_10s['ridge']['mae']
                })
                
                # Save 10s combined
                df_10s.to_csv(OUTPUT_DIR / "elderly_aligned_10.0s.csv", index=False)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Window':<10} {'Samples':<10} {'Features':<10} {'XGB r':<10} {'Ridge r':<10} {'XGB MAE':<10} {'Ridge MAE':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['window']:<10} {r['samples']:<10} {r['features']:<10} {r['xgb_r']:.3f}{'':<6} {r['ridge_r']:.3f}{'':<6} {r['xgb_mae']:.3f}{'':<6} {r['ridge_mae']:.3f}")


if __name__ == "__main__":
    main()
