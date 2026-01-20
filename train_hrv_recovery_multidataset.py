#!/usr/bin/env python3
"""
Multi-Dataset HRV Recovery Training

Combines data from multiple patients and conditions:
- parsingsim3: sim_healthy3, sim_elderly3, sim_severe3
- parsingsim4: sim_healthy4, sim_elderly4, sim_severe4
- parsingsim5: sim_healthy5, sim_elderly5, sim_severe5

Total: 9 datasets → ~288 activities → Much better generalization!
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# Dataset configurations
DATASETS = [
    {
        'patient': 'parsingsim3',
        'conditions': ['sim_healthy3', 'sim_elderly3', 'sim_severe3'],
    },
    {
        'patient': 'parsingsim4',
        'conditions': ['sim_healthy4', 'sim_elderly4', 'sim_severe4'],
    },
    {
        'patient': 'parsingsim5',
        'conditions': ['sim_healthy5', 'sim_elderly5', 'sim_severe5'],
    },
]

DATA_ROOT = "/Users/pascalschlegel/data/interim"


def load_dataset(patient, condition):
    """Load data from a single dataset."""
    base_path = Path(DATA_ROOT) / patient / condition / 'effort_estimation_output' / f'{patient}_{condition}'
    
    # Check if data exists
    fused_file = base_path / 'fused_aligned_10.0s.csv'
    
    if not fused_file.exists():
        print(f"  ⚠️  Missing: {fused_file}")
        return None, None
    
    # Load fused data
    df = pd.read_csv(fused_file)
    
    # Only use HRV recovery rate as target (no Borg fallback)
    if 'hrv_recovery_rate' not in df.columns:
        print(f"  ⚠️  No hrv_recovery_rate column in {patient}/{condition}")
        return None, None
    
    df_labeled = df.dropna(subset=['hrv_recovery_rate']).copy()
    
    if len(df_labeled) == 0:
        print(f"  ⚠️  No labeled samples with HRV recovery rate in {patient}/{condition}")
        return None, None
    
    df_labeled['target'] = df_labeled['hrv_recovery_rate']
    print(f"  ✓ Using HRV recovery rate: {len(df_labeled)} samples")
    
    if len(df_labeled) == 0:
        print(f"  ⚠️  No labeled samples in {patient}/{condition}")
        return None, None
    
    # Add dataset identifiers
    df_labeled['patient'] = patient
    df_labeled['condition'] = condition
    df_labeled['dataset_id'] = f"{patient}_{condition}"
    
    # Extract feature columns (exclude metadata and target columns)
    meta_cols = ['window_id', 'start_idx', 'end_idx', 't_start', 't_center', 't_end', 
                 'hrv_recovery_rate', 'target', 'patient', 'condition', 'dataset_id']
    features = [c for c in df_labeled.columns if c not in meta_cols]
    
    print(f"  ✓ {patient}/{condition}: {len(df_labeled)} samples, {len(features)} features")
    
    return df_labeled, features


def load_all_datasets():
    """Load and combine all datasets."""
    print(f"{'='*70}")
    print(f"LOADING MULTI-DATASET DATA")
    print(f"{'='*70}")
    
    all_dfs = []
    all_features = []
    
    for dataset in DATASETS:
        patient = dataset['patient']
        print(f"\n{patient}:")
        
        for condition in dataset['conditions']:
            df, features = load_dataset(patient, condition)
            if df is not None:
                all_dfs.append(df)
                all_features.append(set(features))
    
    if not all_dfs:
        raise ValueError("No datasets loaded successfully!")
    
    # Find common features across all datasets
    common_features = set.intersection(*all_features) if all_features else set()
    common_features = sorted(list(common_features))
    
    print(f"\n{'='*70}")
    print(f"COMBINING DATASETS")
    print(f"{'='*70}")
    print(f"  Loaded {len(all_dfs)} datasets")
    print(f"  Common features: {len(common_features)}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['dataset_id', 't_center']).reset_index(drop=True)
    
    print(f"  Total samples: {len(combined_df)}")
    
    # Show breakdown by dataset
    print(f"\n  Per-dataset breakdown:")
    for dataset_id in combined_df['dataset_id'].unique():
        n_samples = (combined_df['dataset_id'] == dataset_id).sum()
        print(f"    {dataset_id}: {n_samples} samples")
    
    return combined_df, common_features


def activity_based_split(df, test_size=0.2, random_state=42):
    """
    Split by activities across ALL datasets.
    
    Each dataset has its own activities. We detect activities within each dataset,
    then randomly assign entire activities to train or test.
    """
    print(f"\n{'='*70}")
    print(f"ACTIVITY-BASED SPLIT (MULTI-DATASET)")
    print(f"{'='*70}")
    
    # Assign activity IDs within each dataset
    activity_counter = 0
    df['activity_id'] = -1
    
    for dataset_id in df['dataset_id'].unique():
        dataset_mask = df['dataset_id'] == dataset_id
        dataset_df = df[dataset_mask].copy()
        dataset_df = dataset_df.sort_values('t_center').reset_index(drop=True)
        
        # Detect activities via target value changes (works for both HRV and Borg)
        target_changes = dataset_df['target'].diff().abs() > 0.0001
        activity_boundaries = target_changes | (dataset_df.index == 0)
        local_activity_ids = activity_boundaries.cumsum()
        
        # Assign global activity IDs
        global_activity_ids = local_activity_ids + activity_counter
        df.loc[dataset_mask, 'activity_id'] = global_activity_ids.values
        
        n_activities = local_activity_ids.max()
        activity_counter += n_activities
        
        print(f"  {dataset_id}: {n_activities} activities")
    
    total_activities = df['activity_id'].nunique()
    print(f"\n  Total activities: {total_activities}")
    
    # Random split of activities
    unique_activities = df['activity_id'].unique()
    n_test_activities = max(1, int(total_activities * test_size))
    
    np.random.seed(random_state)
    test_activities = np.random.choice(unique_activities, size=n_test_activities, replace=False)
    
    test_mask = df['activity_id'].isin(test_activities).values
    train_mask = ~test_mask
    
    print(f"\n  Split (test_size={test_size}):")
    print(f"    Train: {train_mask.sum()} windows from {total_activities - n_test_activities} activities")
    print(f"    Test: {test_mask.sum()} windows from {n_test_activities} activities")
    
    # Show train/test distribution by dataset
    print(f"\n  Train/Test distribution by dataset:")
    for dataset_id in df['dataset_id'].unique():
        dataset_mask = df['dataset_id'] == dataset_id
        n_train = (train_mask & dataset_mask).sum()
        n_test = (test_mask & dataset_mask).sum()
        print(f"    {dataset_id}: {n_train} train, {n_test} test")
    
    return train_mask, test_mask


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models."""
    results = {}
    
    print(f"\n{'='*70}")
    print(f"TRAINING MODELS")
    print(f"{'='*70}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest (regularized)
    print(f"\n1. Random Forest (regularized)")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)
    
    results['RandomForest'] = {
        'model': rf,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['RandomForest']['train_r2']:.4f}  |  Test R² = {results['RandomForest']['test_r2']:.4f}")
    
    # 2. Gradient Boosting (regularized)
    print(f"\n2. Gradient Boosting (regularized)")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    
    y_train_pred = gb.predict(X_train_scaled)
    y_test_pred = gb.predict(X_test_scaled)
    
    results['GradientBoosting'] = {
        'model': gb,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['GradientBoosting']['train_r2']:.4f}  |  Test R² = {results['GradientBoosting']['test_r2']:.4f}")
    
    return scaler, results


def print_results(results):
    """Print model comparison."""
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Gap':>10} {'Test MAE':>10}")
    print(f"{'-'*70}")
    
    for name, res in results.items():
        gap = res['train_r2'] - res['test_r2']
        print(f"{name:<20} {res['train_r2']:>10.4f} {res['test_r2']:>10.4f} {gap:>10.4f} {res['test_mae']:>10.4f}")
    
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_res = results[best_name]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*70}")
    print(f"Test R² = {best_res['test_r2']:.4f}")
    print(f"Test MAE = {best_res['test_mae']:.4f}")
    print(f"Train-Test Gap = {best_res['train_r2'] - best_res['test_r2']:.4f}")
    
    return best_name


def create_visualizations(results, y_test, feature_importances, output_dir):
    """Create performance visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_pred = results[best_name]['y_test_pred']
    
    # 1. Predicted vs True
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, best_pred, alpha=0.5, s=30)
    lim_min = min(y_test.min(), best_pred.min())
    lim_max = max(y_test.max(), best_pred.max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
    ax1.set_xlabel('True HRV Recovery Rate', fontsize=12)
    ax1.set_ylabel('Predicted', fontsize=12)
    ax1.set_title(f'{best_name}\nTest R² = {results[best_name]["test_r2"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Comparison
    ax2 = plt.subplot(2, 3, 2)
    model_names = list(results.keys())
    test_r2s = [results[m]['test_r2'] for m in model_names]
    ax2.bar(model_names, test_r2s, color='steelblue', alpha=0.7)
    ax2.set_ylabel('Test R²', fontsize=12)
    ax2.set_title('Model Performance', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Residuals
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_test - best_pred
    ax3.scatter(best_pred, residuals, alpha=0.5, s=30, color='green')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Residuals', fontsize=12)
    ax3.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance (if available)
    if feature_importances is not None:
        ax4 = plt.subplot(2, 3, 4)
        top_n = min(20, len(feature_importances))
        top_features = feature_importances.head(top_n)
        y_pos = np.arange(len(top_features))
        ax4.barh(y_pos, top_features['importance'].values, color='steelblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_features['feature'].values, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Importance', fontsize=12)
        ax4.set_title(f'Top {top_n} Features', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel('Residuals', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"MULTI-DATASET RESULTS\n\n"
    summary_text += f"Best Model: {best_name}\n\n"
    summary_text += f"Test R² = {results[best_name]['test_r2']:.4f}\n"
    summary_text += f"Test MAE = {results[best_name]['test_mae']:.4f}\n\n"
    summary_text += f"Training on multiple patients\n"
    summary_text += f"and conditions should improve\n"
    summary_text += f"generalization significantly\n"
    summary_text += f"compared to single-dataset\n"
    summary_text += f"training (R² = 0.13)."
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    plot_file = output_path / 'hrv_recovery_multidataset_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualizations saved to {plot_file}")


def save_model(best_name, scaler, results, feature_cols, output_dir):
    """Save best model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': results[best_name]['model'],
        'scaler': scaler,
        'features': feature_cols,
        'model_type': best_name,
        'test_r2': results[best_name]['test_r2'],
        'test_mae': results[best_name]['test_mae'],
        'training_type': 'multi_dataset_activity_split'
    }
    
    model_file = output_path / 'hrv_model_multidataset.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to {model_file}")


def main():
    """Main execution."""
    
    print(f"{'='*70}")
    print(f"MULTI-DATASET HRV RECOVERY TRAINING")
    print(f"{'='*70}")
    print(f"\nCombining data from 3 patients × 3 conditions = 9 datasets")
    print(f"Expected: ~288 activities (vs 32 from single dataset)")
    
    # Load all datasets
    combined_df, common_features = load_all_datasets()
    
    # Activity-based split
    train_mask, test_mask = activity_based_split(combined_df, test_size=0.2, random_state=42)
    
    # Prepare features and target
    X = combined_df[common_features].values
    y = combined_df['target'].values
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Train models
    scaler, results = train_models(X_train, y_train, X_test, y_test)
    
    # Print results
    best_name = print_results(results)
    
    # Feature importance
    feature_importances = None
    if hasattr(results[best_name]['model'], 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': common_features,
            'importance': results[best_name]['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Save outputs
    output_dir = Path.home() / 'data' / 'interim' / 'multidataset_hrv_results'
    create_visualizations(results, y_test, feature_importances, output_dir)
    save_model(best_name, scaler, results, common_features, output_dir)
    
    if feature_importances is not None:
        imp_file = output_dir / 'feature_importance_multidataset.csv'
        feature_importances.to_csv(imp_file, index=False)
        print(f"✓ Feature importance saved to {imp_file}")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nIMPROVEMENT over single dataset:")
    print(f"  Single dataset:     Test R² = 0.13")
    print(f"  Multi-dataset:      Test R² = {results[best_name]['test_r2']:.4f}")
    improvement = (results[best_name]['test_r2'] - 0.13) / 0.13 * 100
    if improvement > 0:
        print(f"  Improvement:        +{improvement:.1f}%")
    else:
        print(f"  Change:             {improvement:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
