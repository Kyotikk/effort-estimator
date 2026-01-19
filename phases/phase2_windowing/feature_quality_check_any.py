#!/usr/bin/env python3
"""
feature_quality_check_any.py

Unsupervised feature quality check for ANY window-feature CSV (IMU/PPG/ECG/etc.).

What it does:
1) Robust CSV read (handles the "title line" bug and encoding issues)
2) Auto-selects feature columns (does NOT require "__" names)
3) Basic cleaning:
   - drop features with too many NaNs
   - drop low-variance features
   - median-impute remaining NaNs
4) Correlation pruning (abs Pearson threshold)
5) PCA round 1 diagnostics + top loadings
6) Stage 2: PCA-energy ranking (unsupervised) and select top N
7) PCA round 2 diagnostics on selected top N
8) Saves artifacts to OUT_DIR

Run:
  source /Users/pascalschlegel/effort-estimator/.venv/bin/activate
  python /Users/pascalschlegel/effort-estimator/windowing/feature_quality_check_any.py \
    --in_csv "/path/to/features.csv" \
    --out_dir "/path/to/out_dir"

Example (your PPG):
  python windowing/feature_quality_check_any.py \
    --in_csv "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_ppg2_green_6/2025-12-04_ppg_features_30s.csv" \
    --out_dir "/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/quality_ppg_30s"
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# DEFAULT CONFIG (CLI can override IN/OUT)
# -----------------------------
DEFAULT_IN_CSV = ""
DEFAULT_OUT_DIR = ""

# Non-feature columns (metadata) that may appear in CSV
NON_FEATURE_COLS = {
    "window_id", "start_idx", "end_idx", "valid", "error",
    "timestamp", "time", "datetime", "sample", "session_id",
    "t_start", "t_center", "t_end", "n_samples", "win_sec", "modality",
    "ppg_ok", "_error", "vitalpy_ok", "_vitalpy_error"
}

# Cleaning thresholds
MAX_NAN_RATIO_PER_FEATURE = 0.10   # drop feature if >10% NaNs
LOW_VARIANCE_EPS = 1e-8            # drop feature if variance <= this

# Correlation pruning
CORR_THRESHOLD = 0.95              # drop one of any pair with |corr| >= threshold

# PCA reporting
PCA_TOPK = 8
PCA_MAXPCS_PRINT = 8
PCA_VARIANCE_TARGETS = (0.90, 0.95, 0.99)
PRINT_PCS_EXPLAINED = 15          # print PC1..PC15 explained variance ratios

# Save absolute correlation matrix for inspection
SAVE_CORR_MATRIX = True

# Feature column selection strategy:
# - if there are columns containing "__", you may optionally prefer them (TIFEX style)
# - else: use all numeric columns excluding NON_FEATURE_COLS
PREFER_DOUBLE_UNDERSCORE = False  # IMPORTANT for PPG features (no "__")

# Stage 2 selection: choose top-N by PCA-energy ranking (unsupervised)
STAGE2_TOP_N = 30
STAGE2_VARIANCE_TARGET = 0.90     # use PCs up to this cumulative variance for energy score


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def detect_skiprows_for_title_line(path: str) -> int:
    """
    Broken file starts with a single title line like: 'features_10.0s'
    (no commas), then the real CSV header line.
    If so, skip the first line.
    """
    with open(path, "rb") as f:
        first = f.readline()

    first_s = first.decode("utf-8", errors="replace").strip()
    if first_s.startswith("features_") and ("," not in first_s) and (";" not in first_s) and ("\t" not in first_s):
        return 1
    return 0


def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Robust reader:
    - skips the title line if present
    - uses python engine (tolerant to oddities)
    - handles encoding junk
    """
    skip = detect_skiprows_for_title_line(path)

    try:
        df = pd.read_csv(
            path,
            skiprows=skip,
            engine="python",
            encoding="utf-8",
            encoding_errors="replace",
        )
        return df
    except TypeError:
        # pandas <2.0 has no encoding_errors
        df = pd.read_csv(
            path,
            skiprows=skip,
            engine="python",
            encoding="utf-8",
        )
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            skiprows=skip,
            engine="python",
            encoding="latin1",
        )
        return df


def select_feature_columns(df: pd.DataFrame) -> list:
    # Drop accidental index columns created by exports
    cols0 = [c for c in df.columns if not str(c).startswith("Unnamed:")]

    # Remove explicit metadata cols
    cols = [c for c in cols0 if c not in NON_FEATURE_COLS]

    # Prefer tifex-style features if enabled and present
    if PREFER_DOUBLE_UNDERSCORE:
        dd = [c for c in cols if "__" in str(c)]
        if dd:
            return dd

    # Otherwise: any column that has at least one numeric value (after coercion)
    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)

    return numeric_cols


def drop_high_nan_and_low_variance(X: pd.DataFrame):
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
        if (not np.isfinite(max_val)) or (max_val < threshold):
            break

        i, j = np.unravel_index(np.argmax(sub.values), sub.values.shape)
        fi = sub.index[i]
        fj = sub.columns[j]

        # drop the more "redundant" one (higher mean abs corr)
        drop_feat = fi if mean_abs_corr[fi] >= mean_abs_corr[fj] else fj
        keep.remove(drop_feat)
        dropped.append({"feature": drop_feat, "reason": f"|corr|>={threshold}", "value": float(max_val)})

    kept = [c for c in features if c in keep]
    Xp = X[kept].copy()
    dropped_df = pd.DataFrame(dropped) if dropped else pd.DataFrame(columns=["feature", "reason", "value"])
    return Xp, kept, dropped_df, corr_abs


def pca_analyze(X: pd.DataFrame):
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


def print_pca_explained(explained_df: pd.DataFrame, n: int, title: str):
    print(f"\nExplained variance per PC ({title}):")
    m = min(n, len(explained_df))
    for i in range(m):
        pc = explained_df.loc[i, "PC"]
        ev = float(explained_df.loc[i, "explained_variance_ratio"])
        cum = float(explained_df.loc[i, "cumulative_explained_variance"])
        print(f"  {pc:>4s}: {ev:.4f}   (cum {cum:.4f})")


def summarize_pca_loadings(loadings_df: pd.DataFrame, top_k=8, max_pcs=8, title: str = ""):
    pcs = [c for c in loadings_df.columns if c.startswith("PC")][:max_pcs]
    rows = []

    print(f"\nTop PCA loadings per component ({title}):" if title else "\nTop PCA loadings per component:")

    for pc in pcs:
        s = loadings_df[pc].abs().sort_values(ascending=False).head(top_k)
        print(f"\n{pc}")
        for feat, val in s.items():
            print(f"  {feat:40s} {val:.4f}")
            rows.append({"PC": pc, "feature": feat, "abs_loading": float(val)})

    return pd.DataFrame(rows)


def rank_features_by_pca_energy(
    loadings_df: pd.DataFrame,
    explained_var: np.ndarray,
    variance_target: float = 0.90,
):
    """
    score(f) = sum_{k<=K} |loading(f,k)| * explained_var(k)
    where K is PCs needed to reach variance_target.
    """
    cum = np.cumsum(explained_var)
    K = int(np.searchsorted(cum, variance_target) + 1)

    weights = explained_var[:K]
    L = loadings_df.iloc[:, :K].abs()

    scores = (L * weights).sum(axis=1)

    ranking = (
        pd.DataFrame({
            "feature": scores.index,
            "pca_energy_score": scores.values
        })
        .sort_values("pca_energy_score", ascending=False)
        .reset_index(drop=True)
    )
    return ranking, K


# -----------------------------
# Main
# -----------------------------
def run(in_csv: str, out_dir: str):
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input not found: {in_csv}")

    ensure_dir(out_dir)

    df = read_csv_robust(in_csv)
    print(f"Loaded CSV: {in_csv}")
    print(f"Shape: {df.shape}")

    feats = select_feature_columns(df)
    if not feats:
        print("Columns seen:", list(df.columns)[:50], "...")
        raise ValueError(
            "No feature columns found.\n"
            "If this is a TIFEX file and you want only '__' features, set PREFER_DOUBLE_UNDERSCORE=True.\n"
            "If this is PPG (stats/PSD) keep it False and ensure NON_FEATURE_COLS is correct."
        )

    print(f"Selected feature columns: {len(feats)}")

    X0 = df[feats].apply(pd.to_numeric, errors="coerce")

    # Basic drop (NaN, low variance) + median impute
    X1, kept_basic, dropped_basic_df, basic_metrics_df = drop_high_nan_and_low_variance(X0)
    print(f"After NaN/variance filter: ({X1.shape[0]}, {X1.shape[1]})")

    # Correlation prune
    X2, kept_corr, dropped_corr_df, corr_abs_df = correlation_prune(X1, CORR_THRESHOLD)
    print(f"After correlation pruning: ({X2.shape[0]}, {X2.shape[1]})")

    if X2.shape[1] < 2:
        raise ValueError(f"Too few features after pruning ({X2.shape[1]}). Lower pruning thresholds or check input.")

    # -----------------------------
    # PCA round 1
    # -----------------------------
    pca_var1_df, pca_load1_df, pcs_for_targets1 = pca_analyze(X2)

    print("\nPCA summary (round 1):")
    for k, v in pcs_for_targets1.items():
        print(f"  {k}: {v}")

    print_pca_explained(pca_var1_df, PRINT_PCS_EXPLAINED, title="round 1")
    top1_df = summarize_pca_loadings(
        pca_load1_df,
        top_k=PCA_TOPK,
        max_pcs=PCA_MAXPCS_PRINT,
        title="round 1"
    )

    # -----------------------------
    # Stage 2: PCA-energy ranking
    # -----------------------------
    ranking_df, K_used = rank_features_by_pca_energy(
        pca_load1_df,
        pca_var1_df["explained_variance_ratio"].values,
        variance_target=STAGE2_VARIANCE_TARGET,
    )

    print(f"\nStage 2: PCs used to reach {int(STAGE2_VARIANCE_TARGET*100)}%: {K_used}")
    print(f"Stage 2: ranking pool size: {len(ranking_df)}")

    selected_features = ranking_df.head(STAGE2_TOP_N)["feature"].tolist()
    print(f"Stage 2: selected features: {len(selected_features)} (topN={STAGE2_TOP_N})")

    # -----------------------------
    # PCA round 2 (diagnostics only)
    # -----------------------------
    Xr = X2[selected_features].copy()
    pca_var2_df, pca_load2_df, pcs_for_targets2 = pca_analyze(Xr)

    print("\nPCA summary (round 2 / reduced):")
    for k, v in pcs_for_targets2.items():
        print(f"  {k}: {v}")

    print_pca_explained(pca_var2_df, PRINT_PCS_EXPLAINED, title="round 2")
    top2_df = summarize_pca_loadings(
        pca_load2_df,
        top_k=PCA_TOPK,
        max_pcs=PCA_MAXPCS_PRINT,
        title="round 2"
    )

    # -----------------------------
    # Save artifacts
    # -----------------------------
    metrics_out = os.path.join(out_dir, "feature_quality_metrics.csv")
    dropped_basic_path = os.path.join(out_dir, "features_dropped_basic.csv")
    kept_basic_path    = os.path.join(out_dir, "features_kept_basic.csv")
    dropped_corr_path  = os.path.join(out_dir, "features_dropped_correlation.csv")
    kept_corr_path     = os.path.join(out_dir, "features_kept_correlation.csv")

    basic_metrics_df.to_csv(metrics_out, index=False)
    dropped_basic_df.to_csv(dropped_basic_path, index=False)
    pd.Series(kept_basic).to_csv(kept_basic_path, index=False, header=False)

    dropped_corr_df.to_csv(dropped_corr_path, index=False)
    pd.Series(kept_corr).to_csv(kept_corr_path, index=False, header=False)

    if SAVE_CORR_MATRIX:
        corr_path = os.path.join(out_dir, "corr_matrix_abs.csv")
        corr_abs_df.to_csv(corr_path, index=True)

    # PCA round 1
    pca_var1_path = os.path.join(out_dir, "pca_variance_round1.csv")
    pca_load1_path = os.path.join(out_dir, "pca_loadings_round1.csv")
    pca_top1_path = os.path.join(out_dir, "pca_top_loadings_round1.csv")

    pca_var1_df.to_csv(pca_var1_path, index=False)
    pca_load1_df.to_csv(pca_load1_path)
    top1_df.to_csv(pca_top1_path, index=False)

    # Stage 2 ranking + selection
    ranking_path = os.path.join(out_dir, "pca_energy_ranking_round1.csv")
    selected_path = os.path.join(out_dir, f"features_selected_pca_energy_top{STAGE2_TOP_N}.csv")
    ranking_df.to_csv(ranking_path, index=False)
    pd.Series(selected_features).to_csv(selected_path, index=False, header=False)

    # PCA round 2
    pca_var2_path = os.path.join(out_dir, "pca_variance_round2.csv")
    pca_load2_path = os.path.join(out_dir, "pca_loadings_round2.csv")
    pca_top2_path = os.path.join(out_dir, "pca_top_loadings_round2.csv")

    pca_var2_df.to_csv(pca_var2_path, index=False)
    pca_load2_df.to_csv(pca_load2_path)
    top2_df.to_csv(pca_top2_path, index=False)

    print("\nSaved to:", out_dir)
    print(" - feature_quality_metrics.csv")
    print(" - features_* (basic + correlation)")
    if SAVE_CORR_MATRIX:
        print(" - corr_matrix_abs.csv")
    print(" - pca_variance_round1.csv / pca_loadings_round1.csv / pca_top_loadings_round1.csv")
    print(f" - pca_energy_ranking_round1.csv / features_selected_pca_energy_top{STAGE2_TOP_N}.csv")
    print(" - pca_variance_round2.csv / pca_loadings_round2.csv / pca_top_loadings_round2.csv")
    print("DONE")


def main():
    ap = argparse.ArgumentParser(description="Unsupervised feature quality check for any window-feature CSV.")
    ap.add_argument("--in_csv", default=DEFAULT_IN_CSV, required=(DEFAULT_IN_CSV == ""), help="Input features CSV")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, required=(DEFAULT_OUT_DIR == ""), help="Output directory")
    args = ap.parse_args()

    run(args.in_csv, args.out_dir)


if __name__ == "__main__":
    main()
