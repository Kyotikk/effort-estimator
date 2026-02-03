#!/usr/bin/env python3
"""
Improved Multi-Subject Training Pipeline for Effort Estimation

Key improvements over train_multisub_xgboost.py:
1. Uses RandomForest/GradientBoosting (better than Ridge for this data)
2. DATA-DRIVEN feature selection - automatically finds best features from ANY modality
3. Per-subject calibration with configurable amount
4. Reports HONEST per-subject r (not misleading pooled r)

Feature Selection Strategy:
- For each LOSO fold, selects features that correlate consistently 
  (same direction) across ALL training subjects
- This avoids data leakage and finds features that generalize
- Could be IMU, PPG, EDA, or any combination - data decides!

Usage:
    python train_improved.py data/feature_extraction/path/fused_aligned_5.0s.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import data-driven feature selection
from ml.best_feature_selection import select_best_features_loso, get_feature_columns


def train_loso_improved(df, cal_fraction=0.3, model_type='rf', min_corr=0.10, verbose=True):
    """
    Train with Leave-One-Subject-Out + per-subject calibration.
    
    FEATURE SELECTION: Data-driven, selects best features from ANY modality
    that correlate consistently across training subjects.
    
    Args:
        df: DataFrame with features and 'borg', 'subject' columns
        cal_fraction: Fraction of test subject data for calibration (0.3 = 30%)
        model_type: 'rf' (RandomForest), 'gb' (GradientBoosting), 'ridge', 'svr'
        min_corr: Minimum correlation threshold for feature selection
        verbose: Print progress
        
    Returns:
        results: Dict with predictions, metrics, etc.
    """
    df = df.dropna(subset=['borg']).copy()
    subjects = sorted(df['subject'].unique())
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOSO Training: {model_type.upper()} + DATA-DRIVEN features + {cal_fraction*100:.0f}% calibration")
        print(f"{'='*70}")
        print(f"Subjects: {subjects}")
    
    all_preds = []
    all_true = []
    all_subjects = []
    per_subject_metrics = {}
    all_feature_info = {}
    
    for test_subj in subjects:
        # DATA-DRIVEN FEATURE SELECTION (using ONLY training data)
        feature_cols, feature_info = select_best_features_loso(
            df, test_subj, min_corr=min_corr
        )
        all_feature_info[test_subj] = feature_info
        
        if verbose:
            mod = feature_info['modality_breakdown']
            print(f"\n  Test {test_subj}: {len(feature_cols)} features "
                  f"(IMU={mod['IMU']}, PPG={mod['PPG']}, EDA={mod['EDA']})")
        
        if len(feature_cols) == 0:
            print(f"  ⚠️ No features selected for {test_subj}, skipping")
            continue
        
        # Split data
        train_df = df[df['subject'] != test_subj]
        test_df = df[df['subject'] == test_subj]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['borg'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['borg'].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create model
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        elif model_type == 'gb':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_raw = model.predict(X_test_scaled)
        
        # Per-subject calibration
        n_cal = max(1, int(len(y_test) * cal_fraction))
        cal_idx = np.arange(n_cal)
        test_idx = np.arange(n_cal, len(y_test))
        
        # Simple linear calibration: shift predictions to match calibration mean
        cal_offset = y_test[cal_idx].mean() - y_pred_raw[cal_idx].mean()
        y_pred_cal = y_pred_raw + cal_offset
        
        # Evaluate on non-calibration samples
        if len(test_idx) > 5:
            y_pred_test = y_pred_cal[test_idx]
            y_true_test = y_test[test_idx]
            
            r, _ = pearsonr(y_pred_test, y_true_test)
            mae = np.mean(np.abs(y_pred_test - y_true_test))
            within_1 = np.mean(np.abs(y_pred_test - y_true_test) <= 1) * 100
            
            per_subject_metrics[test_subj] = {
                'r': r,
                'mae': mae,
                'within_1_borg': within_1,
                'n_features': len(feature_cols),
                'n_test': len(test_idx),
                'n_cal': n_cal
            }
            
            all_preds.extend(y_pred_test)
            all_true.extend(y_true_test)
            all_subjects.extend([test_subj] * len(test_idx))
    
    # Overall metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    pooled_r, _ = pearsonr(all_preds, all_true)
    per_subject_r = np.mean([m['r'] for m in per_subject_metrics.values()])
    overall_mae = np.mean(np.abs(all_preds - all_true))
    overall_within_1 = np.mean(np.abs(all_preds - all_true) <= 1) * 100
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   Pooled r = {pooled_r:.3f} (misleading!)")
        print(f"   Per-subject r = {per_subject_r:.3f} (HONEST)")
        print(f"   MAE = {overall_mae:.2f}")
        print(f"   ±1 Borg = {overall_within_1:.1f}%")
        
        print(f"\n   Per-subject breakdown:")
        for subj, m in per_subject_metrics.items():
            print(f"     {subj}: r={m['r']:.3f}, MAE={m['mae']:.2f}, ±1 Borg={m['within_1_borg']:.1f}%")
    
    return {
        'predictions': all_preds,
        'true_values': all_true,
        'subjects': all_subjects,
        'pooled_r': pooled_r,
        'per_subject_r': per_subject_r,
        'mae': overall_mae,
        'within_1_borg': overall_within_1,
        'per_subject_metrics': per_subject_metrics,
        'feature_info_per_fold': all_feature_info,
        'model_type': model_type,
        'cal_fraction': cal_fraction
    }


def save_results(results, output_dir):
    """Save training results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'subject': results['subjects'],
        'true_borg': results['true_values'],
        'pred_borg': results['predictions']
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save metrics
    metrics = {
        'pooled_r': results['pooled_r'],
        'per_subject_r': results['per_subject_r'],
        'mae': results['mae'],
        'within_1_borg': results['within_1_borg'],
        'model_type': results['model_type'],
        'cal_fraction': results['cal_fraction'],
        'n_features': len(results['feature_cols'])
    }
    pd.Series(metrics).to_csv(output_dir / 'metrics.csv')
    
    # Save per-subject metrics
    per_subj_df = pd.DataFrame(results['per_subject_metrics']).T
    per_subj_df.to_csv(output_dir / 'per_subject_metrics.csv')
    
    print(f"\n💾 Saved results to {output_dir}/")


def main(fused_path, output_dir=None, model_type='rf', cal_fraction=0.3):
    """Main training function"""
    # Load data
    print(f"\n📂 Loading: {fused_path}")
    df = pd.read_csv(fused_path)
    df_labeled = df.dropna(subset=['borg'])
    print(f"   Loaded {len(df_labeled)} labeled samples")
    
    # Train
    results = train_loso_improved(df_labeled, cal_fraction, model_type)
    
    # Save if output dir specified
    if output_dir:
        save_results(results, output_dir)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_improved.py <fused_aligned_path> [output_dir] [model_type] [cal_fraction]")
        print("\nModel types: rf (default), gb, ridge, svr")
        print("Cal fraction: 0.3 (default)")
        sys.exit(1)
    
    fused_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    model_type = sys.argv[3] if len(sys.argv) > 3 else 'rf'
    cal_fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    
    main(fused_path, output_dir, model_type, cal_fraction)
