"""
Module 6: Training & Evaluation (Baseline)

Regression models to predict HRV recovery from physiological features.
Baseline: ElasticNet and XGBoost. Metrics: MAE, R², Pearson r.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def prepare_model_data(
    model_table: pd.DataFrame,
    target_col: str = "delta_rmssd",
    exclude_cols: Optional[list] = None,
    drop_qc_fail: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare X, y arrays for training.
    
    Args:
        model_table: Feature + label table
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        drop_qc_fail: Drop rows with qc_ok=False
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature column names
    """
    if model_table.empty:
        raise ValueError("Empty model table")
    
    # Filter QC
    if drop_qc_fail and 'qc_ok' in model_table.columns:
        df = model_table[model_table['qc_ok']].copy()
    else:
        df = model_table.copy()
    
    if df.empty:
        raise ValueError("No rows after QC filtering")
    
    # Check target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in data")
    
    y = df[target_col].values
    
    # Drop target and metadata
    if exclude_cols is None:
        exclude_cols = [
            'bout_id', 'task_name', 'session_id', 'qc_ok', 'note',
            target_col, 'rmssd_end', 'rmssd_recovery',  # also drop other target variants
            'delta_rmssd', 'recovery_slope',
        ]
    
    X_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[X_cols].fillna(X_cols[0] if len(X_cols) > 0 else 0).values
    
    # Handle NaN columns by filling with median
    for i, col in enumerate(X_cols):
        if np.isnan(X[:, i]).any():
            median_val = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = median_val
    
    logger.info(f"Prepared data: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"  Target {target_col} range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y, X_cols


def train_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train ElasticNet baseline model.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        results: {model, scaler, metrics, X_test, y_test}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'mae_train': np.mean(np.abs(model.predict(X_train_scaled) - y_train)),
        'mae_test': np.mean(np.abs(y_pred - y_test)),
        'r2_train': model.score(X_train_scaled, y_train),
        'r2_test': model.score(X_test_scaled, y_test),
    }
    
    # Pearson r
    if len(y_test) > 2:
        r, p = pearsonr(y_test, y_pred)
        metrics['r_pearson'] = r
        metrics['p_pearson'] = p
    
    logger.info(f"ElasticNet: MAE_test={metrics['mae_test']:.4f}, R²_test={metrics['r2_test']:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
    }


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> Dict:
    """
    Train XGBoost baseline model.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set fraction
        random_state: Random seed
        n_estimators: Number of boosting rounds
        
    Returns:
        results: {model, scaler, metrics, X_test, y_test}
    """
    if not HAS_XGBOOST:
        logger.warning("XGBoost not installed; skipping")
        return {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize (for feature importance interpretability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'mae_train': np.mean(np.abs(model.predict(X_train_scaled) - y_train)),
        'mae_test': np.mean(np.abs(y_pred - y_test)),
        'r2_train': model.score(X_train_scaled, y_train),
        'r2_test': model.score(X_test_scaled, y_test),
    }
    
    # Pearson r
    if len(y_test) > 2:
        r, p = pearsonr(y_test, y_pred)
        metrics['r_pearson'] = r
        metrics['p_pearson'] = p
    
    logger.info(f"XGBoost: MAE_test={metrics['mae_test']:.4f}, R²_test={metrics['r2_test']:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
    }


def evaluate_models(
    model_table: pd.DataFrame,
    target_col: str = "delta_rmssd",
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    End-to-end: prepare data, train models, report metrics.
    
    Args:
        model_table: Feature + label table
        target_col: Target column name
        output_dir: Save results here
        
    Returns:
        results: {elasticnet, xgboost, summary}
    """
    # Prepare
    X, y, feature_names = prepare_model_data(model_table, target_col=target_col)
    
    results = {
        'n_samples': len(X),
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    
    # Train ElasticNet
    results['elasticnet'] = train_elasticnet(X, y)
    
    # Train XGBoost
    results['xgboost'] = train_xgboost(X, y)
    
    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / f"training_summary_{target_col}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"HRV Recovery Estimation - Training Summary\n")
            f.write(f"Target: {target_col}\n")
            f.write(f"Samples: {results['n_samples']}\n")
            f.write(f"Features: {results['n_features']}\n\n")
            
            if results['elasticnet']:
                f.write("ElasticNet Metrics:\n")
                for k, v in results['elasticnet']['metrics'].items():
                    f.write(f"  {k}: {v:.4f}\n")
            
            if results['xgboost']:
                f.write("\nXGBoost Metrics:\n")
                for k, v in results['xgboost']['metrics'].items():
                    f.write(f"  {k}: {v:.4f}\n")
        
        logger.info(f"Saved summary to {summary_file}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    import sys
    if len(sys.argv) > 1:
        model_table_path = sys.argv[1]
        model_table = pd.read_csv(model_table_path)
        results = evaluate_models(model_table)
        print(f"Training complete: {results['n_samples']} samples, {results['n_features']} features")
