#!/usr/bin/env python3
"""
HRV Recovery Rate Training with Activity-Based Split.

Eliminates data leakage by ensuring all windows from each activity
stay together in either train or test set (never split across both).

This is critical because:
1. Each activity generates ~17 windows (50s recovery / 3s step)
2. Windows have 70% overlap (7 seconds of shared data)
3. Random split causes 100% data leakage
4. Activity-based split ensures NO leakage between train/test
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data(output_dir):
    """Load fused data and selected features."""
    output_path = Path(output_dir)
    
    # Load fused data
    df = pd.read_csv(output_path / 'fused_aligned_10.0s.csv')
    df_labeled = df.dropna(subset=['hrv_recovery_rate']).copy()
    
    # Sort by time to ensure chronological order
    df_labeled = df_labeled.sort_values('t_center').reset_index(drop=True)
    
    # Load selected features
    features_file = output_path / 'feature_selection_qc' / 'qc_10.0s' / 'features_selected_pruned.csv'
    with open(features_file, 'r') as f:
        feature_cols = [line.strip() for line in f if line.strip() and line.strip() in df_labeled.columns]
    
    print(f"Loaded {len(df_labeled)} labeled samples")
    print(f"Using {len(feature_cols)} selected features")
    
    return df_labeled, feature_cols


def activity_based_split(df, test_size=0.2, random_state=42):
    """
    Split data by activities (not by windows) to eliminate data leakage.
    
    Each activity generates ~17 windows with 70% overlap. All windows from
    an activity must stay together in either train or test (never both).
    
    Args:
        df: DataFrame with 'hrv_recovery_rate' column (constant per activity)
        test_size: Fraction of activities for test set (not windows!)
        random_state: Random seed for reproducibility
        
    Returns:
        train_mask: Boolean mask for training samples
        test_mask: Boolean mask for test samples
    """
    # Sort by time
    df = df.sort_values('t_center').reset_index(drop=True)
    
    # Detect activity boundaries via changes in hrv_recovery_rate
    # Each activity has a constant HRV recovery rate across all its windows
    hrv_changes = df['hrv_recovery_rate'].diff().abs() > 0.0001
    activity_boundaries = hrv_changes | (df.index == 0)
    activity_ids = activity_boundaries.cumsum()
    df['activity_id'] = activity_ids
    
    print(f"\nDetected {activity_ids.max()} unique activities")
    print(f"  Windows per activity: {len(df) / activity_ids.max():.1f} avg")
    
    # Get unique activities
    unique_activities = df['activity_id'].unique()
    n_activities = len(unique_activities)
    n_test_activities = max(1, int(n_activities * test_size))
    
    # Randomly select test activities
    np.random.seed(random_state)
    test_activities = np.random.choice(unique_activities, size=n_test_activities, replace=False)
    
    # Create masks
    test_mask = df['activity_id'].isin(test_activities).values
    train_mask = ~test_mask
    
    print(f"\nActivity-based split (test_size={test_size}):")
    print(f"  Train activities: {(~test_mask).sum() // ((~test_mask).sum() / (n_activities - n_test_activities)):.0f}")
    print(f"  Test activities: {test_mask.sum() // (test_mask.sum() / n_test_activities):.0f}")
    print(f"  Train windows: {train_mask.sum()}")
    print(f"  Test windows: {test_mask.sum()}")
    
    # Verify no temporal overlap
    train_times = df.loc[train_mask, 't_center'].values
    test_times = df.loc[test_mask, 't_center'].values
    
    # Check if any test window overlaps with train windows (within 10s)
    overlap_count = 0
    for test_t in test_times:
        if np.any(np.abs(train_times - test_t) < 10):
            overlap_count += 1
    
    print(f"\n  Leakage check: {overlap_count}/{len(test_times)} test windows overlap with train")
    if overlap_count > 0:
        print(f"  ⚠️  Warning: {overlap_count/len(test_times)*100:.1f}% leakage detected")
    else:
        print(f"  ✓ No data leakage: All activities cleanly separated")
    
    return train_mask, test_mask


def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model and evaluate."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n{'='*70}")
    print(f"TRAINING RANDOM FOREST")
    print(f"{'='*70}")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print(f"✓ Model trained successfully")
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE (Activity-Based Split)")
    print(f"{'='*70}")
    print(f"Train R² = {train_r2:.4f}  |  Test R² = {test_r2:.4f}")
    print(f"Train MAE = {train_mae:.4f}  |  Test MAE = {test_mae:.4f}")
    print(f"Train RMSE = {train_rmse:.4f}  |  Test RMSE = {test_rmse:.4f}")
    
    return model, scaler, {
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
    }


def save_model(model, scaler, feature_cols, output_dir):
    """Save trained model and metadata."""
    output_path = Path(output_dir)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': feature_cols,
    }
    
    model_file = output_path / 'hrv_model_10.0s_activity_split.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to {model_file}")


def save_predictions(df_test, y_test, y_test_pred, output_dir):
    """Save test predictions with errors."""
    output_path = Path(output_dir)
    
    df_pred = df_test.copy()
    df_pred['y_true'] = y_test
    df_pred['y_pred'] = y_test_pred
    df_pred['error'] = y_test - y_test_pred
    df_pred['abs_error'] = np.abs(df_pred['error'])
    
    pred_file = output_path / 'predictions_10.0s_activity_split.csv'
    df_pred.to_csv(pred_file, index=False)
    
    print(f"✓ Predictions saved to {pred_file}")


def save_feature_importance(model, feature_cols, output_dir):
    """Save feature importance rankings."""
    output_path = Path(output_dir)
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    imp_file = output_path / 'feature_importance_10.0s_activity_split.csv'
    importance_df.to_csv(imp_file, index=False)
    
    print(f"✓ Feature importance saved to {imp_file}")
    
    # Show top 10
    print(f"\nTop 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")


def create_visualizations(metrics, feature_importances, feature_cols, output_dir):
    """Create comprehensive performance visualizations."""
    output_path = Path(output_dir)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Predicted vs True (Train)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(metrics['y_train'], metrics['y_train_pred'], alpha=0.5, s=30)
    lim_min = min(metrics['y_train'].min(), metrics['y_train_pred'].min())
    lim_max = max(metrics['y_train'].max(), metrics['y_train_pred'].max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
    ax1.set_xlabel('True HRV Recovery Rate', fontsize=12)
    ax1.set_ylabel('Predicted HRV Recovery Rate', fontsize=12)
    ax1.set_title(f'Train Set (Activity Split)\nR² = {metrics["train_r2"]:.4f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted vs True (Test)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(metrics['y_test'], metrics['y_test_pred'], alpha=0.5, s=30, color='orange')
    lim_min = min(metrics['y_test'].min(), metrics['y_test_pred'].min())
    lim_max = max(metrics['y_test'].max(), metrics['y_test_pred'].max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
    ax2.set_xlabel('True HRV Recovery Rate', fontsize=12)
    ax2.set_ylabel('Predicted HRV Recovery Rate', fontsize=12)
    ax2.set_title(f'Test Set (Activity Split)\nR² = {metrics["test_r2"]:.4f}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals (Test)
    ax3 = plt.subplot(2, 3, 3)
    residuals = metrics['y_test'] - metrics['y_test_pred']
    ax3.scatter(metrics['y_test_pred'], residuals, alpha=0.5, s=30, color='green')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted HRV Recovery Rate', fontsize=12)
    ax3.set_ylabel('Residuals', fontsize=12)
    ax3.set_title(f'Residual Plot (Test)\nMAE = {metrics["test_mae"]:.4f}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance (Top 20)
    ax4 = plt.subplot(2, 3, 4)
    top_n = 20
    top_features = feature_importances[:top_n]
    y_pos = np.arange(len(top_features))
    ax4.barh(y_pos, top_features['importance'].values, color='steelblue')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel('Residuals', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title(f'Error Distribution (Test)\nRMSE = {metrics["test_rmse"]:.4f}', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Comparison
    ax6 = plt.subplot(2, 3, 6)
    metrics_data = {
        'R²': [metrics['train_r2'], metrics['test_r2']],
        'MAE': [metrics['train_mae'], metrics['test_mae']],
        'RMSE': [metrics['train_rmse'], metrics['test_rmse']],
    }
    x = np.arange(len(metrics_data))
    width = 0.35
    train_vals = [metrics['train_r2'], metrics['train_mae'], metrics['train_rmse']]
    test_vals = [metrics['test_r2'], metrics['test_mae'], metrics['test_rmse']]
    ax6.bar(x - width/2, train_vals, width, label='Train', color='steelblue')
    ax6.bar(x + width/2, test_vals, width, label='Test', color='orange')
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_data.keys())
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_file = output_path / 'model_performance_activity_split.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {plot_file}")


def main():
    """Main execution."""
    output_dir = '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3'
    
    print(f"{'='*70}")
    print(f"HRV RECOVERY TRAINING: Activity-Based Split")
    print(f"{'='*70}")
    
    # Load data
    df_labeled, feature_cols = load_data(output_dir)
    
    # Activity-based split
    train_mask, test_mask = activity_based_split(df_labeled, test_size=0.2, random_state=42)
    
    # Prepare features and target
    X = df_labeled[feature_cols].values
    y = df_labeled['hrv_recovery_rate'].values
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    df_train = df_labeled[train_mask].reset_index(drop=True)
    df_test = df_labeled[test_mask].reset_index(drop=True)
    
    # Train model
    model, scaler, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    save_model(model, scaler, feature_cols, output_dir)
    save_predictions(df_test, y_test, metrics['y_test_pred'], output_dir)
    
    # Feature importance
    feature_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    save_feature_importance(model, feature_cols, output_dir)
    create_visualizations(metrics, feature_importances, feature_cols, output_dir)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Model: hrv_model_10.0s_activity_split.pkl")
    print(f"✓ Predictions: predictions_10.0s_activity_split.csv")
    print(f"✓ Feature importance: feature_importance_10.0s_activity_split.csv")
    print(f"✓ Visualizations: model_performance_activity_split.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
