#!/usr/bin/env python3
"""
Scientifically Valid Multi-Subject XGBoost Training.

Key principles to avoid leakage and overfitting:
1. Leave-One-Subject-Out (LOSO) cross-validation
2. Feature selection done ONLY on training fold (not test)
3. Standardization fitted ONLY on training fold
4. No window overlap between train/test (guaranteed by subject split)
5. Proper nested CV for hyperparameter tuning (optional)

This ensures the model is evaluated on truly unseen data.
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def load_combined_data(filepath: str) -> pd.DataFrame:
    """Load combined multi-subject dataset."""
    df = pd.read_csv(filepath)
    df_labeled = df.dropna(subset=["borg"]).copy()
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Labeled samples: {len(df_labeled)}")
    print(f"Subjects: {df_labeled['subject'].unique().tolist()}")
    
    print(f"\nPer-subject breakdown:")
    for subject in sorted(df_labeled["subject"].unique()):
        n_sub = len(df_labeled[df_labeled["subject"] == subject])
        borg_range = df_labeled[df_labeled["subject"] == subject]["borg"]
        print(f"  {subject}: {n_sub} samples (Borg: {borg_range.min():.1f}-{borg_range.max():.1f})")
    
    return df_labeled


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names (exclude metadata)."""
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    feature_cols = []
    for col in df.columns:
        if col in skip_cols:
            continue
        # Skip lagged/duplicate metadata columns from fusion
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            continue
        feature_cols.append(col)
    
    return feature_cols


def select_features_on_train(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    feature_names: List[str],
    top_n: int = 50,
    corr_threshold: float = 0.85,
) -> Tuple[List[int], List[str]]:
    """
    Select features using ONLY training data.
    
    1. Rank features by absolute correlation with target
    2. Take top N features
    3. Remove redundant features (highly correlated with each other)
    
    Returns indices and names of selected features.
    """
    # Handle NaN/Inf in features
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate correlation with target for each feature
    correlations = []
    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.std(col) < 1e-10:  # Constant column
            correlations.append(0.0)
        else:
            corr = np.corrcoef(col, y_train)[0, 1]
            correlations.append(abs(corr) if np.isfinite(corr) else 0.0)
    
    correlations = np.array(correlations)
    
    # Get top N features by correlation
    top_indices = np.argsort(correlations)[-top_n:][::-1]
    
    # Remove redundant features (pairwise correlation > threshold)
    selected_indices = []
    for idx in top_indices:
        # Check if this feature is too correlated with already selected features
        is_redundant = False
        for sel_idx in selected_indices:
            col1 = X_clean[:, idx]
            col2 = X_clean[:, sel_idx]
            if np.std(col1) > 1e-10 and np.std(col2) > 1e-10:
                pair_corr = abs(np.corrcoef(col1, col2)[0, 1])
                if np.isfinite(pair_corr) and pair_corr > corr_threshold:
                    is_redundant = True
                    break
        
        if not is_redundant:
            selected_indices.append(idx)
    
    selected_names = [feature_names[i] for i in selected_indices]
    
    return selected_indices, selected_names


def leave_one_subject_out_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_features: int = 50,
    corr_threshold: float = 0.85,
) -> Dict:
    """
    Leave-One-Subject-Out Cross-Validation.
    
    For each fold:
    1. Train on all subjects except one
    2. Select features using ONLY training subjects
    3. Fit scaler using ONLY training subjects
    4. Evaluate on held-out subject
    
    This guarantees NO data leakage.
    """
    subjects = df["subject"].unique()
    n_subjects = len(subjects)
    
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION ({n_subjects} folds)")
    print(f"{'='*70}")
    
    all_results = []
    all_predictions = []
    feature_usage = {}  # Track which features are selected in each fold
    
    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n--- Fold {fold_idx + 1}/{n_subjects}: Test on {test_subject} ---")
        
        # Split data
        train_mask = df["subject"] != test_subject
        test_mask = df["subject"] == test_subject
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        X_train_full = df_train[feature_cols].values
        y_train = df_train["borg"].values
        X_test_full = df_test[feature_cols].values
        y_test = df_test["borg"].values
        
        print(f"  Train: {len(df_train)} samples from {df_train['subject'].unique().tolist()}")
        print(f"  Test:  {len(df_test)} samples from {test_subject}")
        
        # Feature selection on TRAINING DATA ONLY
        selected_indices, selected_names = select_features_on_train(
            X_train_full, y_train, feature_cols,
            top_n=n_features,
            corr_threshold=corr_threshold,
        )
        print(f"  Features selected: {len(selected_names)}")
        
        # Track feature usage
        for fname in selected_names:
            feature_usage[fname] = feature_usage.get(fname, 0) + 1
        
        # Extract selected features
        X_train = X_train_full[:, selected_indices]
        X_test = X_test_full[:, selected_indices]
        
        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize using TRAINING DATA ONLY
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with regularization to prevent overfitting
        model = xgb.XGBRegressor(
            n_estimators=200,  # Reduced from 500
            max_depth=4,       # Reduced from 5
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,     # L1 regularization
            reg_lambda=2.0,    # L2 regularization (increased)
            min_child_weight=5, # Increased from 3
            gamma=0.1,         # Minimum loss reduction for split
            random_state=42,
            n_jobs=-1,
        )
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Clip predictions to valid Borg range
        y_test_pred = np.clip(y_test_pred, 0, 10)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"  Test  R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        # Store results
        fold_result = {
            "fold": fold_idx + 1,
            "test_subject": test_subject,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "n_features": len(selected_names),
            "train_r2": train_r2,
            "train_mae": train_mae,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }
        all_results.append(fold_result)
        
        # Store predictions for later analysis
        for i, (true_val, pred_val) in enumerate(zip(y_test, y_test_pred)):
            all_predictions.append({
                "subject": test_subject,
                "y_true": true_val,
                "y_pred": pred_val,
                "fold": fold_idx + 1,
            })
    
    return {
        "fold_results": all_results,
        "predictions": all_predictions,
        "feature_usage": feature_usage,
    }


def compute_overall_metrics(results: Dict) -> Dict:
    """Compute overall metrics from LOSO CV results."""
    predictions = pd.DataFrame(results["predictions"])
    
    y_true = predictions["y_true"].values
    y_pred = predictions["y_pred"].values
    
    overall_r2 = r2_score(y_true, y_pred)
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Per-fold metrics
    fold_results = pd.DataFrame(results["fold_results"])
    
    return {
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "mean_test_r2": fold_results["test_r2"].mean(),
        "std_test_r2": fold_results["test_r2"].std(),
        "mean_test_mae": fold_results["test_mae"].mean(),
        "std_test_mae": fold_results["test_mae"].std(),
        "mean_train_r2": fold_results["train_r2"].mean(),
        "train_test_r2_gap": fold_results["train_r2"].mean() - fold_results["test_r2"].mean(),
    }


def train_final_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_features: int = 50,
    corr_threshold: float = 0.85,
) -> Tuple[xgb.XGBRegressor, StandardScaler, List[str]]:
    """
    Train final model on ALL data for deployment.
    
    This is used after LOSO CV has validated the approach.
    The CV metrics (not this model's metrics) are the true performance estimate.
    """
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL (on all data)")
    print(f"{'='*70}")
    
    X_full = df[feature_cols].values
    y_full = df["borg"].values
    
    # Feature selection on all data
    selected_indices, selected_names = select_features_on_train(
        X_full, y_full, feature_cols,
        top_n=n_features,
        corr_threshold=corr_threshold,
    )
    print(f"Final features selected: {len(selected_names)}")
    
    X_selected = X_full[:, selected_indices]
    X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_scaled, y_full, verbose=False)
    
    # Training metrics (NOT the true performance - use CV metrics!)
    y_pred = model.predict(X_scaled)
    train_r2 = r2_score(y_full, y_pred)
    train_mae = mean_absolute_error(y_full, y_pred)
    
    print(f"Training R² (NOT true performance): {train_r2:.4f}")
    print(f"Training MAE (NOT true performance): {train_mae:.4f}")
    print(f"\n⚠️  Use LOSO CV metrics for true performance estimate!")
    
    return model, scaler, selected_names


def generate_plots(results: Dict, output_dir: Path):
    """Generate diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = pd.DataFrame(results["predictions"])
    fold_results = pd.DataFrame(results["fold_results"])
    overall = compute_overall_metrics(results)
    
    # Plot 1: Predicted vs True (all folds combined)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    subjects = predictions["subject"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(subjects)))
    
    for subj, color in zip(subjects, colors):
        mask = predictions["subject"] == subj
        ax.scatter(
            predictions.loc[mask, "y_true"],
            predictions.loc[mask, "y_pred"],
            c=[color], label=subj, alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
    
    # Perfect prediction line
    ax.plot([0, 10], [0, 10], 'k--', lw=2, label='Perfect')
    
    ax.set_xlabel("True Borg Score", fontsize=12)
    ax.set_ylabel("Predicted Borg Score", fontsize=12)
    ax.set_title(
        f"LOSO Cross-Validation Results\n"
        f"Overall R² = {overall['overall_r2']:.4f}, MAE = {overall['overall_mae']:.4f}",
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower right')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_LOSO_CV_RESULTS.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 01_LOSO_CV_RESULTS.png")
    
    # Plot 2: Per-fold metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(fold_results))
    width = 0.35
    
    # R² per fold
    axes[0].bar(x - width/2, fold_results["train_r2"], width, label='Train', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, fold_results["test_r2"], width, label='Test', color='#e74c3c', alpha=0.8)
    axes[0].axhline(overall["mean_test_r2"], color='red', linestyle='--', label=f'Mean Test R²={overall["mean_test_r2"]:.3f}')
    axes[0].set_xlabel("Fold (Test Subject)")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("R² by Fold (Lower train-test gap = less overfitting)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fold_results["test_subject"], rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    # MAE per fold
    axes[1].bar(x - width/2, fold_results["train_mae"], width, label='Train', color='#2ecc71', alpha=0.8)
    axes[1].bar(x + width/2, fold_results["test_mae"], width, label='Test', color='#e74c3c', alpha=0.8)
    axes[1].axhline(overall["mean_test_mae"], color='red', linestyle='--', label=f'Mean Test MAE={overall["mean_test_mae"]:.3f}')
    axes[1].set_xlabel("Fold (Test Subject)")
    axes[1].set_ylabel("MAE (Borg points)")
    axes[1].set_title("MAE by Fold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fold_results["test_subject"], rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_LOSO_PER_FOLD_METRICS.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 02_LOSO_PER_FOLD_METRICS.png")
    
    # Plot 3: Residual analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    residuals = predictions["y_true"] - predictions["y_pred"]
    
    # Residuals vs predicted
    axes[0].scatter(predictions["y_pred"], residuals, alpha=0.5, edgecolors='black', linewidth=0.3)
    axes[0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel("Predicted Borg Score")
    axes[0].set_ylabel("Residual (True - Predicted)")
    axes[0].set_title("Residuals vs Predicted\n(Should be randomly scattered around 0)")
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].axvline(residuals.mean(), color='blue', linestyle='-', lw=2, label=f'Mean={residuals.mean():.3f}')
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution\nStd={residuals.std():.3f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_RESIDUAL_ANALYSIS.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 03_RESIDUAL_ANALYSIS.png")
    
    # Plot 4: Feature importance (most consistently selected)
    feature_usage = results["feature_usage"]
    top_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feat_names = [f[0] for f in top_features]
    feat_counts = [f[1] for f in top_features]
    n_folds = len(fold_results)
    
    colors = ['#2ecc71' if c == n_folds else '#3498db' if c >= n_folds - 1 else '#95a5a6' 
              for c in feat_counts]
    
    bars = ax.barh(range(len(feat_names)), feat_counts, color=colors, edgecolor='black')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names)
    ax.set_xlabel(f"Times Selected (out of {n_folds} folds)")
    ax.set_title("Most Consistently Selected Features\n(Green = selected in ALL folds)")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_FEATURE_CONSISTENCY.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 04_FEATURE_CONSISTENCY.png")
    
    return output_dir


def save_outputs(
    results: Dict,
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    selected_features: List[str],
    output_dir: Path,
):
    """Save all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "xgboost_loso_validated.json"
    model.save_model(str(model_path))
    print(f"  ✓ Model saved: {model_path}")
    
    # Save selected features
    features_path = output_dir / "selected_features.csv"
    pd.DataFrame({"feature": selected_features}).to_csv(features_path, index=False)
    print(f"  ✓ Features saved: {features_path}")
    
    # Save scaler parameters
    scaler_path = output_dir / "scaler_params.json"
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": selected_features,
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save metrics
    overall = compute_overall_metrics(results)
    metrics_path = output_dir / "loso_cv_metrics.json"
    
    metrics = {
        "validation_method": "Leave-One-Subject-Out Cross-Validation",
        "n_folds": len(results["fold_results"]),
        "total_samples": sum(r["n_test"] for r in results["fold_results"]),
        "n_features": len(selected_features),
        "overall_r2": overall["overall_r2"],
        "overall_mae": overall["overall_mae"],
        "overall_rmse": overall["overall_rmse"],
        "mean_test_r2": overall["mean_test_r2"],
        "std_test_r2": overall["std_test_r2"],
        "mean_test_mae": overall["mean_test_mae"],
        "std_test_mae": overall["std_test_mae"],
        "mean_train_r2": overall["mean_train_r2"],
        "train_test_r2_gap": overall["train_test_r2_gap"],
        "overfitting_risk": "LOW" if overall["train_test_r2_gap"] < 0.1 else "MEDIUM" if overall["train_test_r2_gap"] < 0.2 else "HIGH",
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    # Save fold-level results
    fold_results_path = output_dir / "loso_fold_results.csv"
    pd.DataFrame(results["fold_results"]).to_csv(fold_results_path, index=False)
    print(f"  ✓ Fold results saved: {fold_results_path}")
    
    # Save all predictions
    predictions_path = output_dir / "loso_all_predictions.csv"
    pd.DataFrame(results["predictions"]).to_csv(predictions_path, index=False)
    print(f"  ✓ Predictions saved: {predictions_path}")
    
    return metrics


def main():
    print(f"\n{'='*70}")
    print("SCIENTIFICALLY VALID EFFORT PREDICTION TRAINING")
    print("(Leave-One-Subject-Out Cross-Validation)")
    print(f"{'='*70}")
    
    # Load data
    combined_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
    
    if not combined_file.exists():
        print(f"✗ File not found: {combined_file}")
        print(f"  Run: python run_multisub_pipeline.py")
        sys.exit(1)
    
    df = load_combined_data(combined_file)
    feature_cols = get_feature_columns(df)
    
    print(f"\nFeatures available: {len(feature_cols)}")
    
    # Check we have enough subjects for LOSO
    subjects = df["subject"].unique()
    if len(subjects) < 2:
        print(f"✗ Need at least 2 subjects for LOSO CV, got {len(subjects)}")
        sys.exit(1)
    
    # Run LOSO cross-validation
    results = leave_one_subject_out_cv(
        df, feature_cols,
        n_features=50,
        corr_threshold=0.85,
    )
    
    # Compute and display overall metrics
    overall = compute_overall_metrics(results)
    
    print(f"\n{'='*70}")
    print("OVERALL LOSO CV RESULTS (TRUE PERFORMANCE ESTIMATE)")
    print(f"{'='*70}")
    print(f"\n  Overall R²:  {overall['overall_r2']:.4f}")
    print(f"  Overall MAE: {overall['overall_mae']:.4f} Borg points")
    print(f"  Overall RMSE: {overall['overall_rmse']:.4f}")
    print(f"\n  Mean Test R²: {overall['mean_test_r2']:.4f} ± {overall['std_test_r2']:.4f}")
    print(f"  Mean Test MAE: {overall['mean_test_mae']:.4f} ± {overall['std_test_mae']:.4f}")
    print(f"\n  Train-Test R² Gap: {overall['train_test_r2_gap']:.4f}")
    
    if overall['train_test_r2_gap'] < 0.1:
        print(f"  ✓ LOW overfitting risk (gap < 0.1)")
    elif overall['train_test_r2_gap'] < 0.2:
        print(f"  ⚠ MEDIUM overfitting risk (0.1 < gap < 0.2)")
    else:
        print(f"  ✗ HIGH overfitting risk (gap > 0.2)")
    
    # Train final model for deployment
    model, scaler, selected_features = train_final_model(
        df, feature_cols,
        n_features=50,
        corr_threshold=0.85,
    )
    
    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined")
    plot_dir = generate_plots(results, output_dir / "plots_loso_cv")
    
    # Save all outputs
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")
    
    model_dir = output_dir / "models_validated"
    metrics = save_outputs(results, model, scaler, selected_features, model_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  ✓ Validation: Leave-One-Subject-Out ({len(subjects)} folds)")
    print(f"  ✓ No data leakage: Feature selection per fold, train only")
    print(f"  ✓ No window overlap leakage: Subject-level split")
    print(f"\n  📊 TRUE PERFORMANCE (use these for papers!):")
    print(f"     R² = {overall['overall_r2']:.4f}")
    print(f"     MAE = {overall['overall_mae']:.4f} Borg points")
    print(f"     RMSE = {overall['overall_rmse']:.4f}")
    print(f"\n  📁 Outputs: {output_dir}")
    print(f"     - models_validated/xgboost_loso_validated.json")
    print(f"     - models_validated/loso_cv_metrics.json")
    print(f"     - plots_loso_cv/*.png")
    
    print(f"\n✅ Scientifically valid training complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
