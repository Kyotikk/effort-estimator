#!/usr/bin/env python3
"""
Single-Subject Training and Validation.

Tests if the model works WITHIN a single patient using time-based split.
This avoids cross-subject generalization issues and tests the core hypothesis:
"Can we predict effort from physiological signals for THIS person?"

Uses temporal split (first 80% train, last 20% test) to avoid window overlap leakage.
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
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def load_combined_data(filepath: str) -> pd.DataFrame:
    """Load combined multi-subject dataset."""
    df = pd.read_csv(filepath)
    df_labeled = df.dropna(subset=["borg"]).copy()
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
    """Select features using ONLY training data."""
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    correlations = []
    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.std(col) < 1e-10:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(col, y_train)[0, 1]
            correlations.append(abs(corr) if np.isfinite(corr) else 0.0)
    
    correlations = np.array(correlations)
    top_indices = np.argsort(correlations)[-top_n:][::-1]
    
    selected_indices = []
    for idx in top_indices:
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


def train_single_subject(
    df: pd.DataFrame,
    subject: str,
    feature_cols: List[str],
    train_ratio: float = 0.8,
    n_features: int = 50,
) -> Dict:
    """
    Train and evaluate on a single subject using temporal split.
    
    Uses first 80% of windows for training, last 20% for testing.
    This simulates: "Train on early data, predict later data."
    """
    df_subject = df[df["subject"] == subject].copy()
    
    # Sort by time to ensure temporal split
    if "t_center" in df_subject.columns:
        df_subject = df_subject.sort_values("t_center").reset_index(drop=True)
    
    n_samples = len(df_subject)
    n_train = int(n_samples * train_ratio)
    
    df_train = df_subject.iloc[:n_train]
    df_test = df_subject.iloc[n_train:]
    
    print(f"\n{'='*60}")
    print(f"SUBJECT: {subject}")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"Train: {len(df_train)} (first {train_ratio*100:.0f}%)")
    print(f"Test:  {len(df_test)} (last {(1-train_ratio)*100:.0f}%)")
    print(f"Borg range: {df_subject['borg'].min():.1f} - {df_subject['borg'].max():.1f}")
    print(f"Train Borg: {df_train['borg'].min():.1f} - {df_train['borg'].max():.1f}")
    print(f"Test Borg:  {df_test['borg'].min():.1f} - {df_test['borg'].max():.1f}")
    
    X_train_full = df_train[feature_cols].values
    y_train = df_train["borg"].values
    X_test_full = df_test[feature_cols].values
    y_test = df_test["borg"].values
    
    # Feature selection on training data only
    selected_indices, selected_names = select_features_on_train(
        X_train_full, y_train, feature_cols,
        top_n=n_features, corr_threshold=0.85,
    )
    print(f"Features selected: {len(selected_names)}")
    
    X_train = X_train_full[:, selected_indices]
    X_test = X_test_full[:, selected_indices]
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
    
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred = np.clip(y_test_pred, 0, 10)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"  Test  R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    gap = train_r2 - test_r2
    if gap < 0.1:
        print(f"  ✓ LOW overfitting (gap={gap:.3f})")
    elif gap < 0.2:
        print(f"  ⚠ MEDIUM overfitting (gap={gap:.3f})")
    else:
        print(f"  ✗ HIGH overfitting (gap={gap:.3f})")
    
    # Top features by importance
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-10:][::-1]
    print(f"\nTop 10 features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {selected_names[idx]}: {importance[idx]:.4f}")
    
    return {
        "subject": subject,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "gap": gap,
        "selected_features": selected_names,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "model": model,
        "scaler": scaler,
    }


def plot_results(results: List[Dict], output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_subjects = len(results)
    fig, axes = plt.subplots(1, n_subjects, figsize=(7*n_subjects, 6))
    if n_subjects == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        # Scatter plot
        ax.scatter(res["y_test"], res["y_test_pred"], alpha=0.6, 
                   edgecolors='black', linewidth=0.5, s=50, label='Test')
        
        # Perfect line
        min_val = min(res["y_test"].min(), res["y_test_pred"].min())
        max_val = max(res["y_test"].max(), res["y_test_pred"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect')
        
        ax.set_xlabel("True Borg Score", fontsize=12)
        ax.set_ylabel("Predicted Borg Score", fontsize=12)
        ax.set_title(
            f"{res['subject']}\n"
            f"Test R² = {res['test_r2']:.3f}, MAE = {res['test_mae']:.3f}",
            fontsize=12, fontweight='bold'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / "single_subject_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_dir / 'single_subject_results.png'}")
    
    # Time series plot
    fig, axes = plt.subplots(n_subjects, 1, figsize=(14, 5*n_subjects))
    if n_subjects == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        n_test = len(res["y_test"])
        x = np.arange(n_test)
        
        ax.plot(x, res["y_test"], 'b-', lw=2, label='True Borg', alpha=0.8)
        ax.plot(x, res["y_test_pred"], 'r-', lw=2, label='Predicted', alpha=0.8)
        ax.fill_between(x, res["y_test"], res["y_test_pred"], alpha=0.2, color='gray')
        
        ax.set_xlabel("Window Index (time →)", fontsize=12)
        ax.set_ylabel("Borg Score", fontsize=12)
        ax.set_title(f"{res['subject']} - Temporal Prediction", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "single_subject_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'single_subject_timeseries.png'}")


def main():
    print(f"\n{'='*70}")
    print("SINGLE-SUBJECT WITHIN-PATIENT VALIDATION")
    print("(Testing if model works WITHIN each patient)")
    print(f"{'='*70}")
    
    # Load data
    combined_file = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
    
    if not combined_file.exists():
        print(f"✗ File not found: {combined_file}")
        sys.exit(1)
    
    df = load_combined_data(combined_file)
    feature_cols = get_feature_columns(df)
    
    subjects = df["subject"].unique()
    print(f"\nSubjects available: {list(subjects)}")
    print(f"Features available: {len(feature_cols)}")
    
    # Train each subject separately
    all_results = []
    for subject in sorted(subjects):
        result = train_single_subject(df, subject, feature_cols)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: WITHIN-PATIENT PERFORMANCE")
    print(f"{'='*70}")
    print(f"\n{'Subject':<20} {'Test R²':>10} {'Test MAE':>10} {'Gap':>10} {'Status':>15}")
    print("-" * 70)
    
    for res in all_results:
        status = "✓ GOOD" if res["test_r2"] > 0.5 and res["gap"] < 0.2 else "⚠ CHECK" if res["test_r2"] > 0 else "✗ POOR"
        print(f"{res['subject']:<20} {res['test_r2']:>10.4f} {res['test_mae']:>10.4f} {res['gap']:>10.4f} {status:>15}")
    
    # Average
    avg_r2 = np.mean([r["test_r2"] for r in all_results])
    avg_mae = np.mean([r["test_mae"] for r in all_results])
    avg_gap = np.mean([r["gap"] for r in all_results])
    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_r2:>10.4f} {avg_mae:>10.4f} {avg_gap:>10.4f}")
    
    # Generate plots
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_single_subject")
    plot_results(all_results, output_dir)
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if avg_r2 > 0.5:
        print("\n✓ Model works WITHIN patients!")
        print("  The physiological features DO predict effort for individual patients.")
        print("  Cross-patient failure is due to subject-specific calibration needs.")
    elif avg_r2 > 0:
        print("\n⚠ Model partially works within patients.")
        print("  Some predictive power exists but performance is limited.")
    else:
        print("\n✗ Model doesn't work even within patients.")
        print("  The features may not be informative for effort prediction.")
    
    print(f"\n📁 Plots saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
