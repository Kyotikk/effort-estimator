"""
Training Module
===============
Trains and evaluates XGBoost model for effort estimation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    **xgb_params
) -> dict:
    """
    Train XGBoost model for effort estimation.
    
    Args:
        X: Feature matrix
        y: Target effort values
        test_size: Test split fraction
        random_state: Random seed
        **xgb_params: XGBoost parameters (objective, n_estimators, etc.)
    
    Returns:
        Dictionary with:
            - 'model': Trained XGBoost model
            - 'X_train', 'X_test': Train/test splits
            - 'y_train', 'y_test': Train/test targets
            - 'scaler': Fitted StandardScaler
    """
    # Remove rows with NaN targets
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Default XGBoost params
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": random_state,
    }
    default_params.update(xgb_params)
    
    # Train model
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train_scaled, y_train, verbose=0)
    
    logger.info("Model training complete")
    
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def evaluate_model(model_dict: dict) -> dict:
    """
    Evaluate trained model on test set.
    
    Args:
        model_dict: Dictionary returned from train_model()
    
    Returns:
        Dictionary with metrics: r2, rmse, mae, mape
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    
    model = model_dict["model"]
    X_test = model_dict["X_test"]
    y_test = model_dict["y_test"]
    scaler = model_dict["scaler"]
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        mape = np.nan
    
    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
    
    logger.info(f"Test R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return metrics
