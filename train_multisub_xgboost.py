#!/usr/bin/env python3
"""
Multi-subject XGBoost training for Borg effort estimation.

Trains a single model on combined data from multiple subjects.
Includes GroupKFold cross-validation to prevent subject leakage.
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


def load_combined_data(filepath, window_length=10.0):
    """Load combined multi-subject dataset."""
    df = pd.read_csv(filepath)
    
    # Drop rows without Borg labels
    df_labeled = df.dropna(subset=["borg"]).copy()
    
    print(f"\nDataset: {filepath}")
    print(f"  Total samples: {len(df)}")
    print(f"  Labeled samples: {len(df_labeled)}")
    print(f"  Subjects: {df_labeled['subject'].unique().tolist()}")
    
    # Show breakdown by subject
    print(f"\n  Per-subject breakdown:")
    for subject in df_labeled["subject"].unique():
        n_sub = len(df_labeled[df_labeled["subject"] == subject])
        print(f"    {subject}: {n_sub} samples")
    
    return df_labeled


def prepare_features(df):
    """Extract features and labels from dataset."""
    # Define columns to skip
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    # Features: all numeric columns except those to skip
    feature_cols = [
        col for col in df.columns
        if col not in skip_cols and not col.endswith("_r")
    ]
    
    X = df[feature_cols].values
    y = df["borg"].values
    groups = df["subject"].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, groups, feature_cols


def train_multisub_model(X, y, groups, feature_cols, window_length=10.0):
    """
    Train XGBoost on multi-subject data with GroupKFold CV.
    
    GroupKFold ensures each subject's data is entirely in either
    training or validation set (no subject leakage).
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-SUBJECT XGBOOST TRAINING")
    print(f"{'='*70}")
    
    # Feature selection (keep top k features by variance)
    k_features = 100
    feature_variance = np.var(X, axis=0)
    top_indices = np.argsort(feature_variance)[-k_features:]
    top_indices = np.sort(top_indices)
    
    X_selected = X[:, top_indices]
    selected_cols = [feature_cols[i] for i in top_indices]
    
    print(f"\nFeature selection: kept top {k_features} by variance")
    
    # Train-test split (stratified by subject, not random)
    unique_subjects = np.unique(groups)
    n_test_subjects = max(1, len(unique_subjects) // 4)  # ~25% test subjects
    
    test_subjects = unique_subjects[:n_test_subjects]
    train_mask = ~np.isin(groups, test_subjects)
    test_mask = np.isin(groups, test_subjects)
    
    X_train, X_test = X_selected[train_mask], X_selected[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]
    
    print(f"\nTrain-test split (by subject):")
    print(f"  Train subjects: {list(unique_subjects[n_test_subjects:])}")
    print(f"  Test subjects: {list(test_subjects)}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\n{'='*70}")
    print(f"TRAINING XGBOOST")
    print(f"{'='*70}")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    
    print(f"✓ Model trained successfully")
    
    # Evaluate on train and test sets
    print(f"\n{'='*70}")
    print(f"MODEL EVALUATION")
    print(f"{'='*70}")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTRAIN SET:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # GroupKFold cross-validation
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION (GroupKFold, n_splits=3)")
    print(f"{'='*70}\n")
    
    gkf = GroupKFold(n_splits=3)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_selected, y, groups), 1):
        X_train_cv = X_selected[train_idx]
        X_val_cv = X_selected[val_idx]
        y_train_cv = y[train_idx]
        y_val_cv = y[val_idx]
        
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        model_cv = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        
        model_cv.fit(X_train_cv_scaled, y_train_cv, verbose=False)
        
        y_val_pred = model_cv.predict(X_val_cv_scaled)
        fold_r2 = r2_score(y_val_cv, y_val_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val_cv, y_val_pred))
        fold_mae = mean_absolute_error(y_val_cv, y_val_pred)
        
        cv_r2_scores.append(fold_r2)
        cv_rmse_scores.append(fold_rmse)
        cv_mae_scores.append(fold_mae)
        
        val_subjects = np.unique(groups[val_idx])
        print(f"Fold {fold_idx} (test subjects: {list(val_subjects)}):")
        print(f"  R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}")
    
    print(f"\nCross-Validation Summary:")
    print(f"  R² (mean ± std): {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"  RMSE (mean ± std): {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
    print(f"  MAE (mean ± std): {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")
    
    # Feature importance
    print(f"\n{'='*70}")
    print(f"TOP 15 IMPORTANT FEATURES")
    print(f"{'='*70}\n")
    
    importances = model.feature_importances_
    top_indices_importance = np.argsort(importances)[-15:][::-1]
    
    feature_importance_df = pd.DataFrame({
        "feature": [selected_cols[i] for i in top_indices_importance],
        "importance": importances[top_indices_importance],
    })
    
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['feature']:50s} {row['importance']:8.4f}")
    
    # Save outputs
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"xgboost_multisub_{window_length:.1f}s.json"
    model.save_model(str(model_path))
    
    feature_importance_df.to_csv(
        output_dir / f"feature_importance_multisub_{window_length:.1f}s.csv",
        index=False,
    )
    
    metrics = {
        "window_length": window_length,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X_selected),
        "features_used": k_features,
        "train_r2": float(train_r2),
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "cv_r2_std": float(np.std(cv_r2_scores)),
        "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
        "cv_rmse_std": float(np.std(cv_rmse_scores)),
    }
    
    with open(output_dir / f"metrics_multisub_{window_length:.1f}s.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"OUTPUTS SAVED")
    print(f"{'='*70}")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {output_dir / f'feature_importance_multisub_{window_length:.1f}s.csv'}")
    print(f"  Metrics: {output_dir / f'metrics_multisub_{window_length:.1f}s.json'}")
    
    return model, scaler, selected_cols


if __name__ == "__main__":
    # Load combined dataset
    combined_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
    
    if not combined_file.exists():
        print(f"✗ File not found: {combined_file}")
        print(f"  Run: python run_multisub_pipeline.py")
        sys.exit(1)
    
    df = load_combined_data(combined_file, window_length=10.0)
    X, y, groups, feature_cols = prepare_features(df)
    
    model, scaler, selected_cols = train_multisub_model(X, y, groups, feature_cols, window_length=10.0)
    
    print(f"\n✓ Multi-subject training completed!")
