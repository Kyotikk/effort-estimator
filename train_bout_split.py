#!/usr/bin/env python3
"""
XGBoost training with proper activity-bout-level splits to avoid temporal leakage.

The key fix: Split by ACTIVITY BOUT, not by random window.
With 70% overlap, adjacent windows share most of their data.
Random splits cause massive leakage (Train R²=0.999).
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


def identify_activity_bouts(df, gap_threshold=15.0):
    """
    Identify activity bouts based on time gaps and Borg changes.
    
    A new bout starts when:
    - Time gap > gap_threshold seconds, OR
    - Borg value changes
    """
    df = df.sort_values('t_center').reset_index(drop=True)
    
    bout_ids = [0]
    current_bout = 0
    
    for i in range(1, len(df)):
        time_gap = df.loc[i, 't_center'] - df.loc[i-1, 't_center']
        borg_changed = df.loc[i, 'borg'] != df.loc[i-1, 'borg']
        
        if time_gap > gap_threshold or borg_changed:
            current_bout += 1
        
        bout_ids.append(current_bout)
    
    df['bout_id'] = bout_ids
    return df


def load_and_prepare_data(filepath):
    """Load data and identify activity bouts."""
    df = pd.read_csv(filepath)
    df_labeled = df.dropna(subset=['borg']).copy()
    
    print(f"\n{'='*70}")
    print(f"DATA LOADING")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Labeled samples: {len(df_labeled)}")
    
    # Identify activity bouts
    df_labeled = identify_activity_bouts(df_labeled)
    n_bouts = df_labeled['bout_id'].nunique()
    print(f"Activity bouts identified: {n_bouts}")
    
    # Show bout summary
    print(f"\nBout summary:")
    for bout_id in sorted(df_labeled['bout_id'].unique())[:10]:
        bout_df = df_labeled[df_labeled['bout_id'] == bout_id]
        borg = bout_df['borg'].iloc[0]
        n_windows = len(bout_df)
        print(f"  Bout {bout_id}: {n_windows} windows, Borg={borg}")
    if n_bouts > 10:
        print(f"  ... and {n_bouts - 10} more bouts")
    
    return df_labeled


def get_features(df, pre_selected_features=None):
    """Extract feature columns."""
    if pre_selected_features is not None:
        feature_cols = [c for c in pre_selected_features if c in df.columns]
    else:
        skip_cols = {'t_center', 'borg', 'modality', 'subject', 'bout_id'}
        feature_cols = [c for c in df.columns if c not in skip_cols and not c.endswith('_r')]
    
    return feature_cols


def bout_level_split(df, test_fraction=0.2, random_state=42):
    """
    Split data by activity bout (not by window) to avoid temporal leakage.
    
    All windows from a bout go entirely to train OR test.
    """
    np.random.seed(random_state)
    
    bout_ids = df['bout_id'].unique()
    n_test_bouts = max(1, int(len(bout_ids) * test_fraction))
    
    # Randomly select test bouts
    test_bout_ids = np.random.choice(bout_ids, size=n_test_bouts, replace=False)
    train_bout_ids = [b for b in bout_ids if b not in test_bout_ids]
    
    train_mask = df['bout_id'].isin(train_bout_ids)
    test_mask = df['bout_id'].isin(test_bout_ids)
    
    return df[train_mask].copy(), df[test_mask].copy()


def train_with_bout_split(df, feature_cols):
    """Train XGBoost with proper bout-level splits."""
    
    print(f"\n{'='*70}")
    print(f"BOUT-LEVEL TRAIN/TEST SPLIT (NO TEMPORAL LEAKAGE)")
    print(f"{'='*70}")
    
    train_df, test_df = bout_level_split(df, test_fraction=0.2)
    
    print(f"Train bouts: {train_df['bout_id'].nunique()}")
    print(f"Test bouts: {test_df['bout_id'].nunique()}")
    print(f"Train windows: {len(train_df)}")
    print(f"Test windows: {len(test_df)}")
    
    # Check no overlap
    train_bouts = set(train_df['bout_id'].unique())
    test_bouts = set(test_df['bout_id'].unique())
    assert len(train_bouts & test_bouts) == 0, "Bout overlap detected!"
    print(f"✓ No bout overlap between train/test")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['borg'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['borg'].values
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (with regularization)
    print(f"\n{'='*70}")
    print(f"TRAINING XGBOOST (regularized)")
    print(f"{'='*70}")
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    print(f"✓ Model trained")
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n{'='*70}")
    print(f"RESULTS (BOUT-LEVEL SPLIT)")
    print(f"{'='*70}")
    
    print(f"\nTRAIN SET:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    gap = train_r2 - test_r2
    print(f"\nTrain-Test R² gap: {gap:.4f}")
    if gap > 0.15:
        print(f"⚠️  Still some overfitting (gap > 0.15)")
    else:
        print(f"✓ Reasonable generalization (gap ≤ 0.15)")
    
    return model, scaler, {
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'y_train': y_train, 'y_train_pred': y_train_pred,
    }


def leave_one_bout_out_cv(df, feature_cols):
    """Leave-one-bout-out cross-validation for robust estimates."""
    
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-BOUT-OUT CROSS-VALIDATION")
    print(f"{'='*70}")
    
    bout_ids = df['bout_id'].unique()
    results = []
    
    for test_bout in bout_ids:
        train_df = df[df['bout_id'] != test_bout]
        test_df = df[df['bout_id'] == test_bout]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['borg'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['borg'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train, verbose=False)
        
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            'bout_id': test_bout,
            'true_borg': y_test[0],
            'pred_borg': np.mean(y_pred),
            'mae': mae,
            'n_windows': len(y_test)
        })
    
    results_df = pd.DataFrame(results)
    
    # Aggregate predictions per bout (mean over windows)
    true_borgs = results_df['true_borg'].values
    pred_borgs = results_df['pred_borg'].values
    
    overall_mae = mean_absolute_error(true_borgs, pred_borgs)
    overall_rmse = np.sqrt(mean_squared_error(true_borgs, pred_borgs))
    overall_r2 = r2_score(true_borgs, pred_borgs)
    
    print(f"\nLOBO-CV Results (per-bout aggregated):")
    print(f"  MAE:  {overall_mae:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  R²:   {overall_r2:.4f}")
    print(f"  (Evaluated on {len(results_df)} bouts)")
    
    return results_df, {'mae': overall_mae, 'rmse': overall_rmse, 'r2': overall_r2}


def main(filepath=None):
    if filepath is None:
        filepath = '/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv'
    
    # Load pre-selected features
    qc_dir = Path('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/qc_10.0s')
    selected_features_file = qc_dir / 'features_selected_pruned.csv'
    
    if selected_features_file.exists():
        selected_df = pd.read_csv(selected_features_file)
        # Features are stored in first column (which might have a feature name as header)
        feature_cols = selected_df.iloc[:, 0].tolist()
        # Also include header if it's a valid feature name
        header = selected_df.columns[0]
        if header not in feature_cols and not header.startswith('Unnamed'):
            feature_cols = [header] + feature_cols
        print(f"✓ Loaded {len(feature_cols)} pre-selected features")
    else:
        feature_cols = None
        print("⚠️ No pre-selected features found, using all")
    
    # Load and prepare data
    df = load_and_prepare_data(filepath)
    
    if feature_cols is None:
        feature_cols = get_features(df)
    
    # Filter to valid features
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nUsing {len(feature_cols)} features")
    
    # Train with bout-level split
    model, scaler, metrics = train_with_bout_split(df, feature_cols)
    
    # Leave-one-bout-out CV
    lobo_results, lobo_metrics = leave_one_bout_out_cv(df, feature_cols)
    
    # Save outputs
    output_dir = Path('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models_bout_split')
    output_dir.mkdir(exist_ok=True)
    
    model.save_model(str(output_dir / 'xgboost_bout_split.json'))
    lobo_results.to_csv(output_dir / 'lobo_cv_results.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"OUTPUTS SAVED")
    print(f"{'='*70}")
    print(f"  Model: {output_dir / 'xgboost_bout_split.json'}")
    print(f"  LOBO results: {output_dir / 'lobo_cv_results.csv'}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Train vs Test (bout split)
    ax1 = axes[0]
    ax1.scatter(metrics['y_train'], metrics['y_train_pred'], alpha=0.5, label=f"Train (R²={metrics['train_r2']:.3f})")
    ax1.scatter(metrics['y_test'], metrics['y_test_pred'], alpha=0.7, label=f"Test (R²={metrics['test_r2']:.3f})")
    ax1.plot([0, 7], [0, 7], 'k--', label='Perfect')
    ax1.set_xlabel('True Borg')
    ax1.set_ylabel('Predicted Borg')
    ax1.set_title('Bout-Level Split (No Leakage)')
    ax1.legend()
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 7)
    
    # Plot 2: LOBO-CV results
    ax2 = axes[1]
    ax2.scatter(lobo_results['true_borg'], lobo_results['pred_borg'], alpha=0.7)
    ax2.plot([0, 7], [0, 7], 'k--', label='Perfect')
    ax2.set_xlabel('True Borg')
    ax2.set_ylabel('Predicted Borg (LOBO-CV)')
    ax2.set_title(f'Leave-One-Bout-Out CV (R²={lobo_metrics["r2"]:.3f})')
    ax2.legend()
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bout_split_results.png', dpi=150)
    print(f"  Plot: {output_dir / 'bout_split_results.png'}")
    
    plt.show()
    
    return model, metrics, lobo_metrics


if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
