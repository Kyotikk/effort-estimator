"""
Phase 2: Windowing & Quality Check
===================================
1. Creates sliding windows over preprocessed signals
2. Performs QC on window-based features (correlation pruning, variance check, PCA)
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# WINDOW CREATION
# ============================================================================

def create_windows(
    df: pd.DataFrame,
    fs: float,
    win_sec: float,
    overlap: float,
) -> pd.DataFrame:
    """
    Create sliding windows over uniformly sampled time series.

    If df contains 't_unix', windows are defined in unix seconds.
    Else they are defined in 't_sec'.

    Args:
        df: DataFrame with signal data and 't_unix' or 't_sec' column
        fs: Sampling frequency (Hz)
        win_sec: Window length (seconds)
        overlap: Overlap fraction in [0, 1)

    Returns:
        DataFrame with columns: start_idx, end_idx, t_start, t_center, t_end, n_samples, win_sec
    """
    time_col = "t_unix" if "t_unix" in df.columns else "t_sec"

    if time_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{time_col}'. Found {list(df.columns)}")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    # Verify uniform sampling
    dt = df[time_col].diff().dropna()
    median_dt = dt.median()
    if not np.allclose(dt, median_dt, rtol=1e-3, atol=1e-6):
        raise ValueError(f"Non-uniform sampling detected in {time_col}. Resample before windowing.")

    win_len = int(round(fs * win_sec))
    if win_len < 2:
        raise ValueError("Window length too short (<2 seconds).")

    hop = int(round(win_len * (1.0 - overlap)))
    hop = max(1, hop)

    n = len(df)
    rows = []

    for start in range(0, n - win_len + 1, hop):
        end = start + win_len

        rows.append({
            "start_idx": int(start),
            "end_idx": int(end),  # end-exclusive
            "t_start": float(df[time_col].iloc[start]),
            "t_center": float(df[time_col].iloc[start + win_len // 2]),
            "t_end": float(df[time_col].iloc[end - 1]),
            "n_samples": int(win_len),
            "win_sec": float(win_sec),
        })

    windows = pd.DataFrame(rows)
    logger.info(
        "Created %d windows (time_col=%s, win_sec=%.3f, overlap=%.2f)",
        len(windows), time_col, win_sec, overlap
    )

    return windows


# ============================================================================
# QUALITY CHECK: Feature Validation
# ============================================================================

NON_FEATURE_COLS = {
    "window_id", "start_idx", "end_idx", "valid", "error",
    "timestamp", "time", "datetime", "sample", "session_id",
    "t_start", "t_center", "t_end", "n_samples", "win_sec", "modality",
    "ppg_ok", "_error", "vitalpy_ok", "_vitalpy_error"
}

MAX_NAN_RATIO_PER_FEATURE = 0.10
LOW_VARIANCE_EPS = 1e-8
CORR_THRESHOLD = 0.95
STAGE2_TOP_N = 30
STAGE2_VARIANCE_TARGET = 0.90


def _select_feature_columns(df: pd.DataFrame) -> list:
    """Auto-select numeric feature columns, excluding metadata."""
    cols0 = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    cols = [c for c in cols0 if c not in NON_FEATURE_COLS]
    
    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)
    
    return numeric_cols


def _drop_high_nan_and_low_variance(X: pd.DataFrame):
    """Remove features with too many NaNs or low variance."""
    dropped = []
    
    nan_ratio = X.isna().mean()
    var = X.var(axis=0, ddof=0, skipna=True)
    
    for feat in X.columns:
        nr = float(nan_ratio[feat])
        vv = float(var[feat]) if np.isfinite(var[feat]) else np.nan
        
        if nr > MAX_NAN_RATIO_PER_FEATURE:
            dropped.append({"feature": feat, "reason": f"nan_ratio>{MAX_NAN_RATIO_PER_FEATURE}", "value": nr})
        elif (not np.isfinite(vv)) or (vv <= LOW_VARIANCE_EPS):
            dropped.append({"feature": feat, "reason": f"variance<={LOW_VARIANCE_EPS}", "value": vv})
    
    drop_set = {d["feature"] for d in dropped}
    kept = [c for c in X.columns if c not in drop_set]
    
    X_kept = X[kept].copy()
    med = X_kept.median(axis=0, skipna=True)
    X_imp = X_kept.fillna(med)
    
    return X_imp, kept, pd.DataFrame(dropped) if dropped else pd.DataFrame()


def _correlation_prune(X: pd.DataFrame, threshold: float):
    """Remove highly correlated features."""
    if X.shape[1] <= 1:
        return X.copy(), list(X.columns), pd.DataFrame()
    
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
        
        drop_feat = fi if mean_abs_corr[fi] >= mean_abs_corr[fj] else fj
        keep.remove(drop_feat)
        dropped.append({"feature": drop_feat, "reason": f"|corr|>={threshold}", "value": float(max_val)})
    
    kept = [c for c in features if c in keep]
    return X[kept].copy(), kept, pd.DataFrame(dropped) if dropped else pd.DataFrame()


def _pca_analyze(X: pd.DataFrame, n_components: int = None):
    """Perform PCA analysis and return diagnostics."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    
    n_comps = min(n_components or X.shape[1], X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comps)
    pca.fit(Xs)
    
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    explained_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum
    })
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(evr))],
        index=X.columns
    )
    
    return pca, explained_df, loadings


def quality_check_windows(
    features_csv: str,
    out_dir: str = None,
) -> dict:
    """
    Perform QC on window-based features:
    1. Load features
    2. Drop high-NaN and low-variance features
    3. Prune correlated features
    4. Perform PCA analysis
    5. Save diagnostics
    
    Args:
        features_csv: Path to CSV with windowed features
        out_dir: Optional output directory for diagnostic CSVs
    
    Returns:
        Dictionary with:
            - 'X_clean': cleaned feature matrix (DataFrame)
            - 'kept_features': list of kept feature names
            - 'diagnostics': dict with QC results
    """
    # Load
    df = pd.read_csv(features_csv)
    feature_cols = _select_feature_columns(df)
    X = df[feature_cols].copy()
    
    logger.info(f"Loaded {X.shape[0]} windows x {X.shape[1]} features")
    
    # Step 1: High NaN & low variance
    X, kept1, dropped1 = _drop_high_nan_and_low_variance(X)
    logger.info(f"After NaN/variance filter: {X.shape[1]} features kept")
    
    # Step 2: Correlation prune
    X, kept2, dropped2 = _correlation_prune(X, CORR_THRESHOLD)
    logger.info(f"After correlation prune: {X.shape[1]} features kept")
    
    # Step 3: PCA analysis
    pca, explained_df, loadings = _pca_analyze(X)
    
    # Save diagnostics if out_dir provided
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
        dropped1.to_csv(f"{out_dir}/features_dropped_basic.csv", index=False)
        dropped2.to_csv(f"{out_dir}/features_dropped_correlation.csv", index=False)
        
        pd.DataFrame({"feature": kept2}).to_csv(f"{out_dir}/features_kept_basic.csv", index=False)
        
        explained_df.to_csv(f"{out_dir}/pca_explained_variance.csv", index=False)
        loadings.to_csv(f"{out_dir}/pca_loadings.csv")
        
        logger.info(f"Saved diagnostics to {out_dir}")
    
    return {
        "X_clean": X,
        "kept_features": kept2,
        "pca": pca,
        "explained_variance": explained_df,
        "loadings": loadings,
    }
