#!/usr/bin/env python3
"""
Train separate XGBoost models for each condition.

Uses the combined multi-subject dataset but trains independent models:
- Model 1: elderly3 only
- Model 2: healthy3 only
- Model 3: severe3 only

Each model is evaluated on its own condition's data using train/test split.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined")
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["sim_elderly3", "sim_healthy3", "sim_severe3"]
WINDOW_LENGTH = 10.0

# Features to drop (metadata)
DROP_COLS = [
    "start_idx", "end_idx", "window_id", "t_start", "t_center", "t_end",
    "subject", "modality", "valid", "valid_r", "n_samples", "win_sec",
    "borg",  # TARGET - must not be in features!
]

# Get all columns that end with _r or have duplicate indicators and remove them
def get_drop_columns(df):
    """Get all duplicate/metadata columns to drop."""
    cols_to_drop = []
    
    for col in df.columns:
        # Drop metadata/timing columns
        if col in ["borg", "subject", "modality"]:
            cols_to_drop.append(col)
        # Drop any time-related columns
        elif col.startswith("t_") or "time" in col.lower():
            cols_to_drop.append(col)
        # Drop any window-related columns
        elif "window" in col.lower() or "idx" in col.lower():
            cols_to_drop.append(col)
        # Drop validity/sample columns
        elif "valid" in col.lower() or col in ["n_samples", "win_sec"]:
            cols_to_drop.append(col)
        # Drop duplicate columns with _r suffix or _r.X pattern
        elif col.endswith("_r") or "_r." in col:
            cols_to_drop.append(col)
    
    return list(set(cols_to_drop))  # Remove duplicates

def train_condition_model(condition, df):
    """Train XGBoost model for a specific condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition}")
    print(f"{'='*70}")
    
    # Filter to condition and remove unlabeled
    cond_data = df[df["subject"] == condition].copy()
    cond_data = cond_data.dropna(subset=["borg"])
    
    print(f"Total labeled samples: {len(cond_data)}")
    
    # Get columns to drop
    cols_to_drop = get_drop_columns(df)
    
    # Prepare features and target
    X = cond_data.drop(columns=[c for c in cols_to_drop if c in cond_data.columns])
    y = cond_data["borg"]
    
    # Remove any NaN in features
    valid_idx = ~X.isna().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"After removing NaN: {len(X)} samples")
    print(f"Features: {X.shape[1]}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nTRAIN SET:")
    print(f"  R²:   {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")
    print(f"  MAE:  {mae_train:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  R²:   {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  MAE:  {mae_test:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTOP 10 IMPORTANT FEATURES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    # Save model
    model_path = MODELS_DIR / f"xgboost_{condition}_{WINDOW_LENGTH:.1f}s.json"
    model.save_model(str(model_path))
    print(f"\n✓ Model saved: {model_path}")
    
    # Save scaler params
    scaler_path = MODELS_DIR / f"scaler_{condition}_{WINDOW_LENGTH:.1f}s.json"
    scaler_info = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_info, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature importance
    fi_path = MODELS_DIR / f"feature_importance_{condition}_{WINDOW_LENGTH:.1f}s.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"✓ Feature importance saved: {fi_path}")
    
    # Save metrics
    metrics = {
        "condition": condition,
        "window_length": WINDOW_LENGTH,
        "n_samples": len(cond_data),
        "n_labeled": len(X),
        "n_features": X.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
    }
    
    metrics_path = MODELS_DIR / f"metrics_{condition}_{WINDOW_LENGTH:.1f}s.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    # Load combined dataset
    data_path = DATA_DIR / f"multisub_aligned_{WINDOW_LENGTH:.1f}s.csv"
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Train model for each condition
    all_metrics = []
    for condition in CONDITIONS:
        metrics = train_condition_model(condition, df)
        all_metrics.append(metrics)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("CONDITION COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df[[
        "condition", "n_labeled", "n_features",
        "r2_train", "r2_test", "rmse_test", "mae_test"
    ]].to_string(index=False))
    
    # Overall comparison metrics
    summary_metrics_path = MODELS_DIR / f"summary_metrics_{WINDOW_LENGTH:.1f}s.json"
    with open(summary_metrics_path, "w") as f:
        json.dump({
            "window_length": WINDOW_LENGTH,
            "conditions": all_metrics,
        }, f, indent=2)
    
    print(f"\n✓ Summary metrics saved: {summary_metrics_path}")
    print(f"\n✓ Condition-specific training completed!")


if __name__ == "__main__":
    main()
