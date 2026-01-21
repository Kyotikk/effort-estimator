#!/usr/bin/env python3
"""Train HRV recovery model using ECG-derived RR labels.

This script expects fused aligned features produced by run_pipeline.py:
  data/interim/parsingsim3/sim_elderly3/effort_estimation_output/
    parsingsim3_sim_elderly3/fused_aligned_10.0s.csv

Outputs are saved to:
  ~/data/interim/hrv_recovery_results/
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


DATA_PATH = Path(
    "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv"
)
OUTPUT_DIR = Path.home() / "data" / "interim" / "hrv_recovery_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {
    "window_id",
    "start_idx",
    "end_idx",
    "t_start",
    "t_center",
    "t_end",
    "valid",
    "n_samples",
    "win_sec",
    "modality",
    "subject",
    "borg",
    "hrv_recovery_rate",
    "target",
    "patient",
    "condition",
    "dataset_id",
    "activity_id",
}

HRV_TERMS = ["rmssd", "pnn50", "sdnn", "hrv", "lf_hf", "lfhf", "nn50", "nn20"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fused aligned file: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["hrv_recovery_rate"]).copy()
    return df


def is_hrv_feature(col: str) -> bool:
    low = col.lower()
    return any(term in low for term in HRV_TERMS)


def is_meta(col: str) -> bool:
    if col in META_COLS:
        return True
    if col.endswith("_r"):
        return True
    if any(col.endswith(f"_r.{i}") for i in range(1, 10)):
        return True
    return False


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if not is_meta(c)]
    feature_cols = [c for c in feature_cols if not is_hrv_feature(c)]

    X = df[feature_cols].copy()
    y = df["hrv_recovery_rate"].values

    # Remove zero variance
    nunique = X.nunique()
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep]
    feature_cols = keep

    # Drop high-missing (>30%)
    missing_ratio = X.isna().mean()
    feature_cols = [c for c in feature_cols if missing_ratio[c] <= 0.3]
    X = X[feature_cols]

    # Impute median
    X = X.fillna(X.median())

    return X.values, y, feature_cols


def train_models(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        ),
    }

    results = {}
    best_name = None
    best_r2 = -1e9

    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)
        else:
            model.fit(X_train_s, y_train)
            y_pred_tr = model.predict(X_train_s)
            y_pred_te = model.predict(X_test_s)

        train_r2 = r2_score(y_train, y_pred_tr)
        test_r2 = r2_score(y_test, y_pred_te)
        test_mae = mean_absolute_error(y_test, y_pred_te)
        pr, _ = pearsonr(y_test, y_pred_te)

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "pearson_r": pr,
            "y_test": y_test,
            "y_pred": y_pred_te,
            "feature_names": feature_names,
            "scaler": scaler,
        }

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_name = name

    return {"results": results, "best": best_name}


def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].scatter(y_true, y_pred, s=12, alpha=0.7)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")

    residuals = y_pred - y_true
    sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Error")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_model(best_name: str, best_result: Dict, feature_names: List[str]) -> None:
    model_obj = {
        "model_type": best_name,
        "training_type": "hrv_recovery",
        "model": best_result["model"],
        "scaler": best_result["scaler"],
        "features": feature_names,
        "test_r2": best_result["test_r2"],
        "test_mae": best_result["test_mae"],
        "pearson_r": best_result["pearson_r"],
    }
    out_path = OUTPUT_DIR / "hrv_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"✓ Model saved to {out_path}")


def main():
    print("\n=== HRV RECOVERY TRAINING (ECG-derived RR) ===")
    df = load_data(DATA_PATH)
    X, y, feature_names = prepare_features(df)
    print(f"Samples: {len(y)}, Features: {len(feature_names)}")

    trained = train_models(X, y, feature_names)
    best_name = trained["best"]
    best = trained["results"][best_name]

    print("\nRESULTS:")
    for name, res in trained["results"].items():
        print(
            f"  {name:16s} Train R²={res['train_r2']:.4f} | "
            f"Test R²={res['test_r2']:.4f} | Pearson r={res['pearson_r']:.4f} | "
            f"Test MAE={res['test_mae']:.4f}"
        )

    print(f"\nBEST MODEL: {best_name} | Test R²={best['test_r2']:.4f} | Pearson r={best['pearson_r']:.4f}")

    # Plots
    diag_path = OUTPUT_DIR / "hrv_recovery_diagnostics.png"
    plot_diagnostics(best["y_test"], best["y_pred"], diag_path)
    print(f"✓ Diagnostics plot saved to {diag_path}")

    save_model(best_name, best, feature_names)


if __name__ == "__main__":
    main()
