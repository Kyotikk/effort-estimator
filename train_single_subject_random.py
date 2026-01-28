#!/usr/bin/env python3
"""
Single-Subject Training with Random Split (Shuffled Cross-Validation).

This tests if the features work AT ALL for predicting effort.
Uses random 5-fold CV within each subject - windows can be from any time point.

NOTE: This may have some window overlap leakage (adjacent windows share data),
but it tests the fundamental question: "Do these features correlate with effort?"
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.dropna(subset=["borg"]).copy()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "subject_id", "borg",
    }
    return [col for col in df.columns if col not in skip_cols 
            and not col.endswith("_r") 
            and not any(col.endswith(f"_r.{i}") for i in range(1, 10))]


def select_features(X: np.ndarray, y: np.ndarray, names: List[str], 
                    top_n: int = 50, corr_thresh: float = 0.85):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    corrs = []
    for i in range(X.shape[1]):
        if np.std(X[:, i]) < 1e-10:
            corrs.append(0.0)
        else:
            c = np.corrcoef(X[:, i], y)[0, 1]
            corrs.append(abs(c) if np.isfinite(c) else 0.0)
    
    top_idx = np.argsort(corrs)[-top_n:][::-1]
    
    selected = []
    for idx in top_idx:
        redundant = False
        for sel in selected:
            if np.std(X[:, idx]) > 1e-10 and np.std(X[:, sel]) > 1e-10:
                pc = abs(np.corrcoef(X[:, idx], X[:, sel])[0, 1])
                if np.isfinite(pc) and pc > corr_thresh:
                    redundant = True
                    break
        if not redundant:
            selected.append(idx)
    
    return selected, [names[i] for i in selected]


def cv_single_subject(df: pd.DataFrame, subject: str, feature_cols: List[str], 
                      n_folds: int = 5, n_features: int = 50) -> Dict:
    """5-fold random CV within a single subject."""
    
    df_sub = df[df["subject"] == subject].copy().reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"SUBJECT: {subject}")
    print(f"{'='*60}")
    print(f"Samples: {len(df_sub)}")
    print(f"Borg range: {df_sub['borg'].min():.1f} - {df_sub['borg'].max():.1f}")
    print(f"Borg mean: {df_sub['borg'].mean():.2f} ± {df_sub['borg'].std():.2f}")
    
    X_full = df_sub[feature_cols].values
    y_full = df_sub["borg"].values
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_test = []
    all_y_pred = []
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        
        # Feature selection on train only
        sel_idx, sel_names = select_features(X_train, y_train, feature_cols, 
                                              top_n=n_features)
        
        X_train = X_train[:, sel_idx]
        X_test = X_test[:, sel_idx]
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = np.clip(model.predict(X_test), 0, 10)
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        
        fold_r2 = r2_score(y_test, y_pred)
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_metrics.append({"r2": fold_r2, "mae": fold_mae})
        
        print(f"  Fold {fold+1}: R²={fold_r2:.4f}, MAE={fold_mae:.4f}")
    
    # Overall metrics
    overall_r2 = r2_score(all_y_test, all_y_pred)
    overall_mae = mean_absolute_error(all_y_test, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
    
    print(f"\n  OVERALL: R²={overall_r2:.4f}, MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    
    if overall_r2 > 0.7:
        print(f"  ✓ EXCELLENT - Features predict effort well!")
    elif overall_r2 > 0.5:
        print(f"  ✓ GOOD - Features have predictive power")
    elif overall_r2 > 0.3:
        print(f"  ⚠ MODERATE - Some signal but noisy")
    elif overall_r2 > 0:
        print(f"  ⚠ WEAK - Limited predictive power")
    else:
        print(f"  ✗ POOR - No better than mean prediction")
    
    return {
        "subject": subject,
        "n_samples": len(df_sub),
        "borg_range": (df_sub['borg'].min(), df_sub['borg'].max()),
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "fold_metrics": fold_metrics,
        "y_test": np.array(all_y_test),
        "y_pred": np.array(all_y_pred),
    }


def plot_results(results: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        ax.scatter(res["y_test"], res["y_pred"], alpha=0.5, 
                   edgecolors='black', linewidth=0.3, s=40)
        
        lims = [min(res["y_test"].min(), res["y_pred"].min()) - 0.5,
                max(res["y_test"].max(), res["y_pred"].max()) + 0.5]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        
        ax.set_xlabel("True Borg", fontsize=12)
        ax.set_ylabel("Predicted Borg", fontsize=12)
        ax.set_title(f"{res['subject']}\nR²={res['overall_r2']:.3f}, MAE={res['overall_mae']:.3f}",
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "random_cv_results.png", dpi=300)
    plt.close()
    print(f"\n✓ Saved: {output_dir / 'random_cv_results.png'}")
    
    # Residual analysis
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        residuals = res["y_test"] - res["y_pred"]
        ax.scatter(res["y_pred"], residuals, alpha=0.5, edgecolors='black', linewidth=0.3)
        ax.axhline(0, color='red', linestyle='--', lw=2)
        ax.set_xlabel("Predicted Borg", fontsize=12)
        ax.set_ylabel("Residual (True - Pred)", fontsize=12)
        ax.set_title(f"{res['subject']} Residuals", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "random_cv_residuals.png", dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'random_cv_residuals.png'}")


def main():
    print(f"\n{'='*70}")
    print("WITHIN-SUBJECT RANDOM 5-FOLD CROSS-VALIDATION")
    print("(Testing if features correlate with effort AT ALL)")
    print(f"{'='*70}")
    
    filepath = "/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv"
    df = load_data(filepath)
    
    # Handle column naming (some files use 'subject', others use 'subject_id')
    if 'subject_id' in df.columns and 'subject' not in df.columns:
        df['subject'] = df['subject_id']
    
    feature_cols = get_feature_columns(df)
    
    subjects = sorted(df["subject"].unique())
    print(f"\nSubjects: {subjects}")
    print(f"Features: {len(feature_cols)}")
    
    results = []
    for subj in subjects:
        res = cv_single_subject(df, subj, feature_cols)
        results.append(res)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: WITHIN-PATIENT RANDOM CV")
    print(f"{'='*70}")
    print(f"\n{'Subject':<20} {'Borg Range':>15} {'R²':>10} {'MAE':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['subject']:<20} {r['borg_range'][0]:.1f}-{r['borg_range'][1]:.1f}:>15 {r['overall_r2']:>10.4f} {r['overall_mae']:>10.4f}")
    
    avg_r2 = np.mean([r["overall_r2"] for r in results])
    avg_mae = np.mean([r["overall_mae"] for r in results])
    print("-" * 60)
    print(f"{'AVERAGE':<20} {'':<15} {avg_r2:>10.4f} {avg_mae:>10.4f}")
    
    # Plot
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_random_cv")
    plot_results(results, output_dir)
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if avg_r2 > 0.5:
        print("\n✓ Features DO predict effort within patients!")
        print("  The temporal split failure was due to distribution shift.")
        print("  For production: need per-patient calibration or more diverse training data.")
    elif avg_r2 > 0.2:
        print("\n⚠ Features have SOME predictive power.")
        print("  Consider: more/better features, different model, data quality check.")
    else:
        print("\n✗ Features don't predict effort even with random split.")
        print("  Fundamental issue with feature-target relationship.")
    
    print(f"\n📁 Plots: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
