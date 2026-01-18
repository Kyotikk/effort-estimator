#!/usr/bin/env python3
"""
XGBoost training script for Borg effort prediction.
Trains on fused multi-modal features (IMU + PPG + EDA).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pathlib import Path
import matplotlib.pyplot as plt
import json

# =========================================================================
# CONFIG
# =========================================================================
WINDOW_LENGTH = "10.0"
DATA_ROOT = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3")
ALIGNED_FILE = DATA_ROOT / f"fused_aligned_{WINDOW_LENGTH}s.csv"
OUTPUT_DIR = DATA_ROOT / "xgboost_models"

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

CV_FOLDS = 5


# =========================================================================
# LOAD & PREPARE DATA
# =========================================================================
print("=" * 70)
print(f"XGBoost Training for Borg Effort (Window: {WINDOW_LENGTH}s)")
print("=" * 70)

if not ALIGNED_FILE.exists():
    raise FileNotFoundError(f"Aligned file not found: {ALIGNED_FILE}")

print(f"\nLoading data from: {ALIGNED_FILE}")
df = pd.read_csv(ALIGNED_FILE)

print(f"Loaded shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Remove rows with missing Borg labels
df_labeled = df.dropna(subset=["borg"]).copy()
print(f"\nAfter dropping unlabeled rows: {df_labeled.shape[0]} windows")

# Identify feature columns (exclude metadata and labels)
# Include all window_id and timing columns from fusion (window_id_r, start_idx_r, t_start_r, etc.)
metadata_patterns = {"window_id", "start_idx", "end_idx", "valid", "t_start", "t_center", "t_end", "n_samples", "win_sec", "modality", "borg"}
metadata_cols = set()
for col in df_labeled.columns:
    # Check if column matches any metadata pattern (exact match or contains pattern with _r suffix)
    if col in metadata_patterns or any(pattern in col for pattern in ["window_id", "start_idx", "end_idx", "t_start_r", "t_end_r", "modality", "borg"]):
        metadata_cols.add(col)
    elif col in ["valid", "n_samples", "win_sec"]:
        metadata_cols.add(col)

feature_cols = [c for c in df_labeled.columns if c not in metadata_cols]

print(f"Feature columns: {len(feature_cols)}")
print(f"  - IMU features: {len([c for c in feature_cols if 'acc' in c])}")
print(f"  - PPG features: {len([c for c in feature_cols if 'ppg' in c])}")
print(f"  - EDA features: {len([c for c in feature_cols if 'eda' in c])}")

X = df_labeled[feature_cols].copy()
y = df_labeled["borg"].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target (Borg) statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Handle NaN values in features
X = X.fillna(X.mean())
print(f"\nHandled NaN values in features")

# =========================================================================
# FEATURE SELECTION (SelectKBest)
# =========================================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION - SelectKBest with f_regression")
print("=" * 70)

N_FEATURES_SELECT = 100
selector = SelectKBest(f_regression, k=min(N_FEATURES_SELECT, X.shape[1]))
X_selected = selector.fit_transform(X, y)

selected_feature_names = np.array(feature_cols)[selector.get_support()].tolist()
print(f"\nSelected {len(selected_feature_names)} features from {len(feature_cols)}")
print(f"\nTop 15 selected features:")

# Get feature scores and sort
scores = selector.scores_
feature_scores = list(zip(feature_cols, scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for fname, score in feature_scores[:15]:
    if fname in selected_feature_names:
        print(f"  ✓ {fname}: {score:.2f}")

# Create new dataframe with selected features for consistency
X = pd.DataFrame(X_selected, columns=selected_feature_names)
feature_cols = selected_feature_names

print(f"\nNew feature matrix shape: {X.shape}")

# =========================================================================
# TRAIN-TEST SPLIT
# =========================================================================
print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# =========================================================================
# TRAIN XGBoost
# =========================================================================
print("\n" + "=" * 70)
print("TRAINING XGBoost")
print("=" * 70)

model = xgb.XGBRegressor(**XGBOOST_PARAMS)
model.fit(X_train, y_train, verbose=False)

print("✓ Model trained successfully")

# =========================================================================
# EVALUATION
# =========================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Training performance
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\nTRAINING SET:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  R²:   {train_r2:.4f}")

# Test performance
y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTEST SET:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")

# =========================================================================
# CROSS-VALIDATION
# =========================================================================
print("\n" + "=" * 70)
print(f"CROSS-VALIDATION ({CV_FOLDS}-FOLD)")
print("=" * 70)

kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

# CV with different metrics
cv_r2 = cross_val_score(model, X, y, cv=kfold, scoring="r2")
cv_rmse = -cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(cv_rmse)
cv_mae = -cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_absolute_error")

print(f"\nR² scores: {cv_r2}")
print(f"  Mean: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

print(f"\nRMSE scores: {cv_rmse}")
print(f"  Mean: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

print(f"\nMAE scores: {cv_mae}")
print(f"  Mean: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

# =========================================================================
# FEATURE IMPORTANCE
# =========================================================================
print("\n" + "=" * 70)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 70)

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:8.4f}")

# =========================================================================
# SAVE MODEL & RESULTS
# =========================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save model
model_path = OUTPUT_DIR / f"xgboost_borg_{WINDOW_LENGTH}s.json"
model.get_booster().save_model(str(model_path))
print(f"\n✓ Model saved to: {model_path}")

# Save feature importance
importance_path = OUTPUT_DIR / f"feature_importance_{WINDOW_LENGTH}s.csv"
feature_importance.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved to: {importance_path}")

# Save metrics
metrics = {
    "window_length": WINDOW_LENGTH,
    "n_samples_total": len(df_labeled),
    "n_features": len(feature_cols),
    "train_set_size": len(X_train),
    "test_set_size": len(X_test),
    "train_rmse": float(train_rmse),
    "train_mae": float(train_mae),
    "train_r2": float(train_r2),
    "test_rmse": float(test_rmse),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2),
    "cv_r2_mean": float(cv_r2.mean()),
    "cv_r2_std": float(cv_r2.std()),
    "cv_rmse_mean": float(cv_rmse.mean()),
    "cv_rmse_std": float(cv_rmse.std()),
    "cv_mae_mean": float(cv_mae.mean()),
    "cv_mae_std": float(cv_mae.std()),
}

metrics_path = OUTPUT_DIR / f"metrics_{WINDOW_LENGTH}s.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved to: {metrics_path}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Window Length:       {WINDOW_LENGTH}s
Total Samples:       {len(df_labeled)}
Features Used:       {len(feature_cols)}

Test Set Performance:
  R²:   {test_r2:.4f}  (explains {100*test_r2:.1f}% of variance)
  RMSE: {test_rmse:.4f} Borg points
  MAE:  {test_mae:.4f} Borg points

Cross-Validation (5-fold):
  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}
  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}

Outputs saved to: {OUTPUT_DIR}
  - xgboost_borg_{WINDOW_LENGTH}s.json
  - feature_importance_{WINDOW_LENGTH}s.csv
  - metrics_{WINDOW_LENGTH}s.json
""")

print("=" * 70)
