#!/usr/bin/env python3
"""
Train effort estimation model using aligned physiological features and RMSSD labels.

This script trains a machine learning model to predict physiological effort from
wearable sensor features (PPG heart rate and IMU movement), using RMSSD as ground truth.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_and_prepare_data(filepath: str) -> tuple:
    """
    Load aligned dataset and prepare features and labels.
    
    Args:
        filepath: Path to aligned CSV file
        
    Returns:
        Tuple of (X_features, y_labels, feature_names, metadata_df)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading aligned dataset from {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Separate metadata, features, and labels
    metadata_cols = ['adl_id', 'adl_name', 'start_time', 'end_time', 'duration_sec', 'borg_rpe']
    label_col = 'rmssd'  # RMSSD in milliseconds
    
    # Exclude columns that are derived from label (would be data leakage)
    exclude_cols = metadata_cols + [label_col, 'ln_rmssd', 'n_beats', 'mean_rr', 'std_rr', 
                                     'session_id', 'rmssd', 'ln_rmssd']
    
    # Feature columns are everything except metadata and labels
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Drop columns with all missing values
    for col in feature_cols[:]:
        if df[col].isnull().all():
            logger.info(f"Dropping column {col} (all missing)")
            feature_cols.remove(col)
    
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Extract features and labels
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    metadata = df[metadata_cols].copy()
    
    # Check for missing values
    missing_features = X.isnull().sum()
    if missing_features.any():
        logger.warning("Missing values detected:")
        for col, count in missing_features[missing_features > 0].items():
            logger.warning(f"  {col}: {count}/{len(X)} samples")
    
    # Handle missing values: drop rows with any missing features or labels
    complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
    n_complete = complete_mask.sum()
    n_dropped = len(X) - n_complete
    
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} samples with missing values")
        X = X[complete_mask]
        y = y[complete_mask]
        metadata = metadata[complete_mask]
    
    logger.info(f"Final dataset: {len(X)} complete samples")
    logger.info(f"Label (RMSSD) range: {y.min():.2f} - {y.max():.2f} ms")
    logger.info(f"Label mean: {y.mean():.2f} ms, std: {y.std():.2f} ms")
    
    return X.values, y.values, feature_cols, metadata


def train_and_evaluate(X, y, feature_names, model_type='rf'):
    """
    Train model with cross-validation and return trained model and metrics.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: List of feature names
        model_type: 'rf' (Random Forest), 'gbm' (Gradient Boosting), or 'ridge'
        
    Returns:
        Tuple of (trained_model, scaler, cv_scores, metrics_dict)
    """
    logger = logging.getLogger(__name__)
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select model
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        logger.info("Using Random Forest Regressor")
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        logger.info("Using Gradient Boosting Regressor")
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
        logger.info("Using Ridge Regression")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation (5-fold or leave-one-out if very few samples)
    n_samples = len(X)
    if n_samples < 10:
        logger.warning(f"Only {n_samples} samples - using leave-one-out CV")
        cv = n_samples  # LOO-CV
    else:
        cv = min(5, n_samples)  # 5-fold or less
        logger.info(f"Using {cv}-fold cross-validation")
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    logger.info("Running cross-validation...")
    cv_mae = -cross_val_score(model, X_scaled, y, cv=kfold, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_r2 = cross_val_score(model, X_scaled, y, cv=kfold, 
                            scoring='r2', n_jobs=-1)
    
    logger.info(f"Cross-validation MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f} ms")
    logger.info(f"Cross-validation R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    
    # Train final model on all data
    logger.info("Training final model on full dataset...")
    model.fit(X_scaled, y)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    logger.info(f"Full dataset performance:")
    logger.info(f"  MAE: {mae:.3f} ms")
    logger.info(f"  RMSE: {rmse:.3f} ms")
    logger.info(f"  R²: {r2:.3f}")
    
    metrics = {
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'full_mae': mae,
        'full_rmse': rmse,
        'full_r2': r2,
        'n_samples': n_samples,
        'n_features': X.shape[1]
    }
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        logger.info("\nTop 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        metrics['feature_importance'] = feature_importance
    
    return model, scaler, y_pred, metrics


def plot_results(y_true, y_pred, metrics, output_dir):
    """Create visualization plots of model performance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual RMSSD (ms)', fontsize=12)
    ax.set_ylabel('Predicted RMSSD (ms)', fontsize=12)
    ax.set_title(f'Predicted vs Actual RMSSD\nR² = {metrics["full_r2"]:.3f}, MAE = {metrics["full_mae"]:.2f} ms', 
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=150)
    plt.close()
    
    # 2. Residuals
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted RMSSD (ms)', fontsize=12)
    ax.set_ylabel('Residuals (ms)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150)
    plt.close()
    
    # 3. Feature importance (if available)
    if 'feature_importance' in metrics:
        importance_df = metrics['feature_importance']
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(importance_df))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', ax=ax)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Train physiological effort estimation model'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to aligned dataset CSV'
    )
    parser.add_argument(
        '--model-type',
        choices=['rf', 'gbm', 'ridge'],
        default='rf',
        help='Model type: rf (Random Forest), gbm (Gradient Boosting), ridge (Ridge Regression)'
    )
    parser.add_argument(
        '--output-dir',
        default='models',
        help='Directory to save trained model and results'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("EFFORT ESTIMATION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    X, y, feature_names, metadata = load_and_prepare_data(args.data)
    
    if len(X) < 5:
        logger.error(f"Insufficient samples ({len(X)}) for training. Need at least 5.")
        sys.exit(1)
    
    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL")
    logger.info("=" * 60)
    
    model, scaler, y_pred, metrics = train_and_evaluate(
        X, y, feature_names, model_type=args.model_type
    )
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'effort_model_{args.model_type}.pkl'
    scaler_path = output_dir / f'scaler_{args.model_type}.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"\nSaved model to {model_path}")
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_path = output_dir / 'feature_names.txt'
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_names))
    logger.info(f"Saved feature names to {feature_path}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Number of samples: {metrics['n_samples']}\n")
        f.write(f"Number of features: {metrics['n_features']}\n\n")
        f.write("Cross-Validation Results:\n")
        f.write(f"  MAE: {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} ms\n")
        f.write(f"  R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}\n\n")
        f.write("Full Dataset Performance:\n")
        f.write(f"  MAE: {metrics['full_mae']:.3f} ms\n")
        f.write(f"  RMSE: {metrics['full_rmse']:.3f} ms\n")
        f.write(f"  R²: {metrics['full_r2']:.3f}\n")
        
        if 'feature_importance' in metrics:
            f.write("\n\nTop 20 Feature Importances:\n")
            f.write("-" * 50 + "\n")
            for idx, row in metrics['feature_importance'].head(20).iterrows():
                f.write(f"{row['feature']:30s} {row['importance']:.6f}\n")
    
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate plots
    if args.plots:
        logger.info("\nGenerating plots...")
        plot_results(y, y_pred, metrics, output_dir / 'plots')
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
