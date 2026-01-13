#!/usr/bin/env python3
"""
feature_quality_check.py

Run (no args):
  /usr/bin/python3 /Users/pascalschlegel/effort-estimator/feature_quality_check.py

What it does:
- Loads your manual feature CSV (default: 5s_ol50 inside effort-estimator)
- Loads the feature list from config/selected_features.yaml (single source of truth)
- Validates all config features exist in the CSV
- Basic cleaning:
    * drops features with too many NaNs
    * drops low-variance features
    * imputes remaining NaNs with median
- Correlation pruning (absolute Pearson corr threshold)
- PCA on pruned features
- Saves artifacts into OUT_DIR and prints a concise summary

Outputs (in OUT_DIR):
- feature_quality_metrics.csv
- features_dropped_basic.csv
- features_kept_basic.csv                 (headerless)
- features_dropped_correlation.csv
- features_kept_correlation.csv           (headerless)  <-- recommended to freeze
- pca_variance.csv
- pca_loadings.csv
- pca_top_loadings.csv
- corr_matrix_abs.csv                     (optional but useful)
"""

import os
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# CONFIG (edit only here)
# -----------------------------
CFG_YAML = "/Users/pascalschlegel/effort-estimator/config/selected_features.yaml"

# Your real file that exists (effort-estimator repo output)
IN_CSV = "/Users/pascalschlegel/effort-estimator/data/feature_extraction/manual_out/manual_features_5s_ol50.csv"

# Write outputs into a dedicated run folder (avoid mixing with old runs)
OUT_DIR = "/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/quality_5s_ol50"

# Non-feature columns (metadata) that may appear in CSV
NON_FEATURE_COLS = {"window_id", "timestamp", "time", "datetime", "sample", "session_id"}

# Cleaning thresholds
MAX_NAN_RATIO_PER_FEATURE = 0.10   # drop feature if >10% NaNs
LOW_VARIANCE_EPS = 1e-8            # drop feature if variance <= this

# Correlation pruning
CORR_THRESHOLD = 0.95              # drop one of any pair with |corr| >= threshold

# PCA reporting
PCA_TOPK = 8
PCA_MAXPCS_PRINT = 8
PCA_VARIANCE_TARGETS = (0.90, 0.95, 0.99)

# Save absolute correlation matrix for inspection
SAVE_CORR_MATRIX = True


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_features_from_yaml(path: str) -> list:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    feats = cfg.get("features", [])
    if not feats or not isinstance(feats, list):
        raise ValueError(f"No valid 'features:' list found in YAML: {path}")
    feats = [str(x).strip() for x in feats if str(x).strip()]
    if not feats:
        raise ValueError(f"'features:' list is empty in YAML: {path}")
    return feats


def drop_high_nan_and_low_variance(X: pd.DataFrame):
    """
    Returns:
      X_clean (median-imputed),
      kept_features,
      dropped_df (reason per dropped feature),
      basic_metrics_df
    """
    dropped = []
    metrics = []

    nan_ratio = X.isna().mean()
    var = X.var(axis=0, ddof=0, skipna=True)

    for feat in X.columns:
        nr = float(nan_ratio[feat])
        vv = float(var[feat]) if np.isfinite(var[feat]) else np.nan
        metrics.append({"feature": feat, "nan_ratio": nr, "variance": vv})

        if nr > MAX_NAN_RATIO_PER_FEATURE:
            dropped.append({"feature": feat, "reason": f"nan_ratio>{MAX_NAN_RATIO_PER_FEATURE}", "value": nr})
        elif (not np.isfinite(vv)) or (vv <= LOW_VARIANCE_EPS):
            dropped.append({"feature": feat, "reason": f"variance<={LOW_VARIANCE_EPS}", "value": vv})

    dropped_df = (
        pd.DataFrame(dropped).sort_values(["reason", "value"], ascending=[True, False])
        if dropped else
        pd.DataFrame(columns=["feature", "reason", "value"])
    )

    drop_set = set(dropped_df["feature"].tolist()) if len(dropped_df) else set()
    kept = [c for c in X.columns if c not in drop_set]

    X_kept = X[kept].copy()

    # Median imputation for remaining NaNs
    med = X_kept.median(axis=0, skipna=True)
    X_imp = X_kept.fillna(med)

    basic_metrics_df = pd.DataFrame(metrics).sort_values("nan_ratio", ascending=False)

    return X_imp, kept, dropped_df, basic_metrics_df


def correlation_prune(X: pd.DataFrame, threshold: float):
    """
    Greedy pruning:
      - compute absolute correlation matrix
      - iteratively remove one feature from the most-correlated pair
      - drops the feature with higher mean absolute correlation (more globally redundant)
    Returns:
      X_pruned, kept_features, dropped_df, corr_abs_df
    """
    if X.shape[1] <= 1:
        corr_abs = X.corr().abs()
        return X.copy(), list(X.columns), pd.DataFrame(columns=["feature", "reason", "value"]), corr_abs

    corr_abs = X.corr().abs()
    np.fill_diagonal(corr_abs.values, 0.0)

    features = list(X.columns)
    dropped = []

    mean_abs_corr = corr_abs.mean(axis=0)
    keep = set(features)

    while True:
        sub = corr_abs.loc[list(keep), list(keep)]
        max_val = sub.values.max() if sub.size else 0.0
        if not np.isfinite(max_val) or max_val < threshold:
            break

        i, j = np.unravel_index(np.argmax(sub.values), sub.values.shape)
        fi = sub.index[i]
        fj = sub.columns[j]

        drop_feat = fi if mean_abs_corr[fi] >= mean_abs_corr[fj] else fj
        keep.remove(drop_feat)
        dropped.append({"feature": drop_feat, "reason": f"|corr|>={threshold}", "value": float(max_val)})

    kept = [c for c in features if c in keep]
    Xp = X[kept].copy()
    dropped_df = pd.DataFrame(dropped) if dropped else pd.DataFrame(columns=["feature", "reason", "value"])
    return Xp, kept, dropped_df, corr_abs


def pca_analyze(X: pd.DataFrame):
    """
    Standardize, fit PCA, compute explained variance and loadings.
    Returns:
      explained_df, loadings_df, pcs_for_targets dict
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA()
    pca.fit(Xs)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    explained_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum
    })

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )

    pcs_for_targets = {}
    for t in PCA_VARIANCE_TARGETS:
        k = int(np.searchsorted(cum, t) + 1)
        pcs_for_targets[f"pcs_for_{int(t*100)}pct"] = k

    return explained_df, loadings_df, pcs_for_targets


def summarize_pca_loadings(loadings_df: pd.DataFrame, top_k=8, max_pcs=8):
    pcs = [c for c in loadings_df.columns if c.startswith("PC")][:max_pcs]
    rows = []

    print("\nTop PCA loadings per component:")
    for pc in pcs:
        s = loadings_df[pc].abs().sort_values(ascending=False).head(top_k)
        print(f"\n{pc}")
        for feat, val in s.items():
            print(f"  {feat:40s} {val:.4f}")
            rows.append({"PC": pc, "feature": feat, "abs_loading": float(val)})

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input not found: {IN_CSV}")
    if not os.path.exists(CFG_YAML):
        raise FileNotFoundError(f"Config YAML not found: {CFG_YAML}")

    ensure_dir(OUT_DIR)

    cfg_feats = load_features_from_yaml(CFG_YAML)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded CSV: {IN_CSV}")
    print(f"Shape: {df.shape}")

    # Validate config features exist
    missing = [c for c in cfg_feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {IN_CSV}: {missing}")

    # Build feature matrix strictly from config
    X0 = df[cfg_feats].apply(pd.to_numeric, errors="coerce")

    # Basic drop (NaN, low variance) + median impute
    X1, kept_basic, dropped_basic_df, basic_metrics_df = drop_high_nan_and_low_variance(X0)
    print(f"After NaN/variance filter: ({X1.shape[0]}, {X1.shape[1]})")

    # Correlation prune
    X2, kept_corr, dropped_corr_df, corr_abs_df = correlation_prune(X1, CORR_THRESHOLD)
    print(f"After correlation pruning: ({X2.shape[0]}, {X2.shape[1]})")

    # PCA
    pca_var_df, pca_loadings_df, pcs_for_targets = pca_analyze(X2)

    print("\nPCA summary:")
    for k, v in pcs_for_targets.items():
        print(f"  {k}: {v}")

    # -----------------------------
    # Save artifacts
    # -----------------------------
    # Metrics
    metrics_out = os.path.join(OUT_DIR, "feature_quality_metrics.csv")
    basic_metrics_df.to_csv(metrics_out, index=False)

    # Dropped/kept lists (HEADERLESS for kept lists -> avoids 'feature' artifact)
    dropped_basic_path = os.path.join(OUT_DIR, "features_dropped_basic.csv")
    kept_basic_path    = os.path.join(OUT_DIR, "features_kept_basic.csv")
    dropped_corr_path  = os.path.join(OUT_DIR, "features_dropped_correlation.csv")
    kept_corr_path     = os.path.join(OUT_DIR, "features_kept_correlation.csv")

    dropped_basic_df.to_csv(dropped_basic_path, index=False)
    pd.Series(kept_basic).to_csv(kept_basic_path, index=False, header=False)

    dropped_corr_df.to_csv(dropped_corr_path, index=False)
    pd.Series(kept_corr).to_csv(kept_corr_path, index=False, header=False)

    # Correlation matrix (abs)
    if SAVE_CORR_MATRIX:
        corr_path = os.path.join(OUT_DIR, "corr_matrix_abs.csv")
        corr_abs_df.to_csv(corr_path, index=True)

    # PCA outputs
    pca_var_path = os.path.join(OUT_DIR, "pca_variance.csv")
    pca_loadings_path = os.path.join(OUT_DIR, "pca_loadings.csv")

    pca_var_df.to_csv(pca_var_path, index=False)
    pca_loadings_df.to_csv(pca_loadings_path)

    # Top loadings summary
    top_df = summarize_pca_loadings(pca_loadings_df, top_k=PCA_TOPK, max_pcs=PCA_MAXPCS_PRINT)
    top_path = os.path.join(OUT_DIR, "pca_top_loadings.csv")
    top_df.to_csv(top_path, index=False)

    print("\nSaved to:", OUT_DIR)
    print(f" - {os.path.basename(metrics_out)}")
    print(f" - {os.path.basename(dropped_basic_path)}")
    print(f" - {os.path.basename(kept_basic_path)}")
    print(f" - {os.path.basename(dropped_corr_path)}")
    print(f" - {os.path.basename(kept_corr_path)}")
    if SAVE_CORR_MATRIX:
        print(f" - {os.path.basename(corr_path)}")
    print(f" - {os.path.basename(pca_var_path)}")
    print(f" - {os.path.basename(pca_loadings_path)}")
    print(f" - {os.path.basename(top_path)}")

    print("\nDONE")


if __name__ == "__main__":
    main()
