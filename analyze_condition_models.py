#!/usr/bin/env python3
"""
Analyze condition-specific model performance by effort level.

For each condition, breaks down MAE, RMSE, R² by Borg effort ranges.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined")
MODELS_DIR = DATA_DIR / "models"
WINDOW_LENGTH = 10.0

CONDITIONS = ["sim_elderly3", "sim_healthy3", "sim_severe3"]

# Define effort ranges for analysis
EFFORT_RANGES = [
    (0, 1, "Very Light (0-1)"),
    (1, 2, "Light (1-2)"),
    (2, 3, "Moderate (2-3)"),
    (3, 4, "Hard (3-4)"),
    (4, 5, "Very Hard (4-5)"),
    (5, 10, "Extreme (5+)"),
]


def analyze_condition_model(condition):
    """Analyze model performance by effort level for a condition."""
    print(f"\n{'='*80}")
    print(f"CONDITION: {condition}")
    print(f"{'='*80}")
    
    # Load combined dataset
    data_path = DATA_DIR / f"multisub_aligned_{WINDOW_LENGTH:.1f}s.csv"
    df = pd.read_csv(data_path)
    
    # Filter to condition and labeled data
    cond_data = df[df["subject"] == condition].dropna(subset=["borg"]).copy()
    
    # Get columns to drop
    cols_to_drop = ["borg", "subject", "modality"]  # Always drop these
    for col in df.columns:
        if col.startswith("t_") or "time" in col.lower():
            cols_to_drop.append(col)
        elif "window" in col.lower() or "idx" in col.lower():
            cols_to_drop.append(col)
        elif "valid" in col.lower() or col in ["n_samples", "win_sec"]:
            cols_to_drop.append(col)
        elif col.endswith("_r") or "_r." in col:
            cols_to_drop.append(col)
    
    # Prepare features and target
    X = cond_data.drop(columns=[c for c in cols_to_drop if c in cond_data.columns])
    y = cond_data["borg"].values
    
    # Remove non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    # Convert to numpy and remove any NaN
    X_np = X.values.astype(float)
    valid_idx = ~np.isnan(X_np).any(axis=1)
    X_np = X_np[valid_idx]
    y = y[valid_idx]
    
    print(f"Features shape: {X_np.shape}, Targets: {len(y)}")
    
    # Load scaler and model
    scaler_path = MODELS_DIR / f"scaler_{condition}_{WINDOW_LENGTH:.1f}s.json"
    model_path = MODELS_DIR / f"xgboost_{condition}_{WINDOW_LENGTH:.1f}s.json"
    
    with open(scaler_path, "r") as f:
        scaler_info = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_info["mean"])
    scaler.scale_ = np.array(scaler_info["scale"])
    
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
    # Make predictions
    X_scaled = (X_np - scaler.mean_) / scaler.scale_
    y_pred = model.predict(X_scaled)
    
    # Overall metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae_overall = mean_absolute_error(y, y_pred)
    rmse_overall = np.sqrt(mean_squared_error(y, y_pred))
    r2_overall = r2_score(y, y_pred)
    
    print(f"\nOVERALL PERFORMANCE ({len(y)} samples):")
    print(f"  MAE:  {mae_overall:.4f}")
    print(f"  RMSE: {rmse_overall:.4f}")
    print(f"  R²:   {r2_overall:.4f}")
    
    # Performance by effort level
    print(f"\nBY EFFORT LEVEL:")
    print(f"{'Level':<20} {'Count':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Samples %':>10}")
    print("-" * 60)
    
    all_results = {
        "condition": condition,
        "overall": {
            "mae": float(mae_overall),
            "rmse": float(rmse_overall),
            "r2": float(r2_overall),
            "n_samples": len(y),
        },
        "by_effort": []
    }
    
    for low, high, label in EFFORT_RANGES:
        mask = (y >= low) & (y < high)
        
        # Handle upper bound for "Extreme" range
        if low == 5:
            mask = (y >= low)
        
        if mask.sum() == 0:
            continue
        
        y_range = y[mask]
        y_pred_range = y_pred[mask]
        
        mae = mean_absolute_error(y_range, y_pred_range)
        rmse = np.sqrt(mean_squared_error(y_range, y_pred_range))
        r2 = r2_score(y_range, y_pred_range)
        pct = 100 * mask.sum() / len(y)
        
        print(f"{label:<20} {mask.sum():>6} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pct:>9.1f}%")
        
        all_results["by_effort"].append({
            "range": label,
            "low": float(low),
            "high": float(high),
            "n_samples": int(mask.sum()),
            "pct_of_total": float(pct),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        })
    
    # Average effort value
    print(f"\nEFFORT LABEL STATISTICS:")
    print(f"  Mean effort: {y.mean():.2f} Borg")
    print(f"  Std effort:  {y.std():.2f}")
    print(f"  Min/Max:     {y.min():.2f} / {y.max():.2f}")
    
    all_results["effort_stats"] = {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
    }
    
    # Save analysis
    analysis_path = MODELS_DIR / f"performance_analysis_{condition}_{WINDOW_LENGTH:.1f}s.json"
    with open(analysis_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Analysis saved: {analysis_path}")
    
    return all_results


def main():
    all_analyses = []
    
    for condition in CONDITIONS:
        analysis = analyze_condition_model(condition)
        all_analyses.append(analysis)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("CROSS-CONDITION SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Condition':<20} {'Overall R²':>12} {'Overall MAE':>12} {'Overall RMSE':>12} {'Samples':>10}")
    print("-" * 70)
    
    for analysis in all_analyses:
        r2 = analysis["overall"]["r2"]
        mae = analysis["overall"]["mae"]
        rmse = analysis["overall"]["rmse"]
        n = analysis["overall"]["n_samples"]
        cond = analysis["condition"]
        
        print(f"{cond:<20} {r2:>12.4f} {mae:>12.4f} {rmse:>12.4f} {n:>10}")
    
    # Save comprehensive summary
    summary_path = MODELS_DIR / f"performance_analysis_summary_{WINDOW_LENGTH:.1f}s.json"
    with open(summary_path, "w") as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"\n✓ Performance analysis completed!")


if __name__ == "__main__":
    main()
