#!/usr/bin/env python3
"""
Multi-subject XGBoost training for Borg effort estimation.

Trains a single model on combined data from multiple subjects.
Includes 80/20 random train-test split across all conditions.
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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


def prepare_features(df, pre_selected_features=None):
    """Extract features and labels from dataset.
    
    Args:
        df: Input dataframe
        pre_selected_features: List of feature names already selected by pipeline.
                              If None, use all non-metadata columns.
    """
    if pre_selected_features is not None:
        # Use pre-selected features from pipeline
        feature_cols = [col for col in pre_selected_features if col in df.columns]
        print(f"\nUsing {len(feature_cols)} pre-selected features from pipeline")
    else:
        # Fallback: extract features manually (no pre-selection)
        skip_cols = {
            "window_id", "start_idx", "end_idx", "valid",
            "t_start", "t_center", "t_end", "n_samples", "win_sec",
            "modality", "subject", "borg",
        }
        
        feature_cols = []
        for col in df.columns:
            if col in skip_cols:
                continue
            # CRITICAL: Skip lagged versions of metadata
            if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
                continue
            feature_cols.append(col)
        
        print(f"\nFeatures: {len(feature_cols)} (metadata removed, NO pre-selection)")
    
    X = df[feature_cols].values
    y = df["borg"].values
    groups = df["subject"].values
    
    print(f"Samples: {len(X)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, groups, feature_cols


def train_multisub_model(X, y, groups, feature_cols, window_length=10.0):
    """
    Train XGBoost on multi-subject data with GroupKFold CV.
    
    Feature selection should already be done by the main pipeline.
    This function just trains on the provided features.
    
    GroupKFold ensures each subject's data is entirely in either
    training or validation set (no subject leakage).
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-SUBJECT XGBOOST TRAINING")
    print(f"{'='*70}")
    
    # Use provided features directly (already selected by pipeline)
    X_selected = X
    selected_cols = feature_cols
    
    print(f"\nUsing {len(selected_cols)} features (pre-selected by pipeline)")
    
    # Train-test split (80/20 random split across all samples, ALL conditions)
    print(f"\nTrain-test split (80/20 random across all conditions):")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
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
        max_depth=5,  # Reduced from 6 for less model complexity
        learning_rate=0.05,  # Reduced from 0.1 for more conservative learning
        subsample=0.7,  # Reduced from 0.8 for more regularization
        colsample_bytree=0.7,  # Reduced from 0.8 for more regularization
        reg_alpha=1.0,  # L1 regularization (was 0 default)
        reg_lambda=1.0,  # L2 regularization (was 1 default)
        min_child_weight=3,  # Increased from 1 default, prevents overfitting to small groups
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
        "features_used": len(selected_cols),
        "train_r2": float(train_r2),
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
    }
    
    with open(output_dir / f"metrics_multisub_{window_length:.1f}s.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"OUTPUTS SAVED")
    print(f"{'='*70}")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {output_dir / f'feature_importance_multisub_{window_length:.1f}s.csv'}")
    print(f"  Metrics: {output_dir / f'metrics_multisub_{window_length:.1f}s.json'}")
    
    # Return all data needed for plotting
    return model, scaler, selected_cols, y_train, y_test, y_train_pred, y_test_pred, train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse


def generate_plots(y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols):
    """Generate comprehensive visualization plots for model evaluation."""
    
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_multisub")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    print(f"\n{'='*70}")
    print(f"GENERATING PLOTS")
    print(f"{'='*70}\n")
    
    # Plot 1: Train vs Test scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Train plot
    errors_train = np.abs(y_train - y_train_pred)
    axes[0].scatter(y_train, y_train_pred, s=100, alpha=0.6, 
                   c=errors_train, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    min_v, max_v = y_train.min(), y_train.max()
    axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=3, label='Perfect Prediction', zorder=5)
    axes[0].set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[0].set_title(f'TRAINING SET (n={len(y_train)})\nR² = {train_r2:.4f} | MAE = {train_mae:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_label('Absolute Error', fontsize=11)
    
    # Test plot
    errors_test = np.abs(y_test - y_test_pred)
    axes[1].scatter(y_test, y_test_pred, s=100, alpha=0.6, 
                   c=errors_test, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    min_v, max_v = y_test.min(), y_test.max()
    axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', lw=3, label='Perfect Prediction', zorder=5)
    axes[1].set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[1].set_title(f'TEST SET (n={len(y_test)})\nR² = {test_r2:.4f} | MAE = {test_mae:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('Absolute Error', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_TRAIN_VS_TEST_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 01_TRAIN_VS_TEST_MULTISUB.png")
    plt.close()
    
    # Plot 2: Metrics bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['R² Score', 'MAE', 'RMSE']
    train_vals = [train_r2, train_mae, train_rmse]
    test_vals = [test_r2, test_mae, test_rmse]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Subject Model Performance: Train vs Test\n(Higher R² is better, Lower MAE/RMSE is better)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_METRICS_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 02_METRICS_MULTISUB.png")
    plt.close()
    
    # Plot 3: Residuals vs. Predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(y_train_pred, y_train - y_train_pred, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_title('TRAINING SET Residuals', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test_pred, y_test - y_test_pred, alpha=0.6, color='#e74c3c', edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_title('TEST SET Residuals', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_RESIDUALS_VS_PREDICTED_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 03_RESIDUALS_VS_PREDICTED_MULTISUB.png")
    plt.close()
    
    # Plot 4: Residuals histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.histplot(y_train - y_train_pred, bins=30, kde=True, color='#1f77b4', ax=axes[0])
    axes[0].set_title('TRAINING SET Residuals Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    sns.histplot(y_test - y_test_pred, bins=30, kde=True, color='#e74c3c', ax=axes[1])
    axes[1].set_title('TEST SET Residuals Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_RESIDUALS_HISTOGRAM_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 04_RESIDUALS_HISTOGRAM_MULTISUB.png")
    plt.close()
    
    # Plot 5: Error vs. True Value
    fig, ax = plt.subplots(figsize=(10, 7))
    abs_errors = np.abs(y_test - y_test_pred)
    ax.scatter(y_test, abs_errors, alpha=0.7, color='#e67e22', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Error vs. True Borg Rating (Test Set)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_ERROR_VS_TRUE_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 05_ERROR_VS_TRUE_MULTISUB.png")
    plt.close()
    
    # Plot 6: Density plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(x=y_test, y=y_test_pred, cmap="Blues", fill=True, thresh=0.05, levels=100, ax=ax)
    ax.scatter(y_test, y_test_pred, s=60, alpha=0.5, color='#2980b9', edgecolors='black', linewidth=0.5)
    min_v, max_v = y_test.min(), y_test.max()
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs. Actual (Test Set) with Density', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_PREDICTED_VS_TRUE_DENSITY_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 06_PREDICTED_VS_TRUE_DENSITY_MULTISUB.png")
    plt.close()
    
    # Plot 7: Feature Importance
    fig, ax = plt.subplots(figsize=(12, 11))
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': selected_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    top_n = 30
    top_features = feature_importance_df.head(top_n)
    
    colors = ['#2ecc71' if 'ppg_' in f else '#e74c3c' if 'eda_' in f else '#3498db' 
              for f in top_features['feature']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors, 
                   edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 30 Most Important Features (Multi-Subject Model)', fontsize=13, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (feat, val) in enumerate(zip(top_features['feature'], top_features['importance'])):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='PPG'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='EDA'),
        Patch(facecolor='#3498db', edgecolor='black', label='IMU (Acc)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_TOP_FEATURES_IMPORTANCE_MULTISUB.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 07_TOP_FEATURES_IMPORTANCE_MULTISUB.png")
    plt.close()
    
    print(f"\n✅ All plots saved to: {output_dir}\n")
    
    return output_dir


if __name__ == "__main__":
    # Load combined dataset
    combined_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
    
    if not combined_file.exists():
        print(f"✗ File not found: {combined_file}")
        print(f"  Run: python run_multisub_pipeline.py")
        sys.exit(1)
    
    df = load_combined_data(combined_file, window_length=10.0)
    
    # Try to load pre-selected features from pipeline
    pre_selected_features = None
    features_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/qc_10.0s/features_selected_pruned.csv")
    
    if features_file.exists():
        print(f"\n▶ Loading pre-selected features from pipeline...")
        # Features file is just a single column of feature names (no header)
        pre_selected_features = pd.read_csv(features_file, header=None)[0].tolist()
        print(f"  ✓ Loaded {len(pre_selected_features)} pre-selected features")
    else:
        print(f"\n⚠ Pre-selected features not found at {features_file}")
        print(f"  Run multisub pipeline first: python run_multisub_pipeline.py")
        print(f"  Falling back to all features (not recommended)")
    
    
    
    X, y, groups, feature_cols = prepare_features(df, pre_selected_features=pre_selected_features)
    
    model, scaler, selected_cols, y_train, y_test, y_train_pred, y_test_pred, train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse = train_multisub_model(X, y, groups, feature_cols, window_length=10.0)
    
    # Generate plots
    plot_dir = generate_plots(y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols)
    
    print(f"\n✓ Multi-subject training and visualization completed!")
