"""
IMU Feature Extraction
======================
Extracts top 20+ features from accelerometer windows.
Uses fast manual feature computation (replaces TIFEX).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Top features from previous analysis - kept for backward compatibility
TOP_FEATURES = [
    "acc_x_dyn__harmonic_mean_of_abs",
    "acc_x_dyn__quantile_0.4",
    "acc_z_dyn__approximate_entropy_0.1",
    "acc_z_dyn__quantile_0.4",
    "acc_x_dyn__sample_entropy",
    "acc_y_dyn__harmonic_mean_of_abs",
    "acc_y_dyn__sample_entropy",
    "acc_z_dyn__sum_of_absolute_changes",
    "acc_y_dyn__avg_amplitude_change",
    "acc_z_dyn__quantile_0.6",
    "acc_z_dyn__variance_of_absolute_differences",
    "acc_x_dyn__quantile_0.6",
    "acc_z_dyn__sample_entropy",
    "acc_y_dyn__variance_of_absolute_differences",
    "acc_x_dyn__max",
    "acc_y_dyn__quantile_0.4",
    "acc_y_dyn__tsallis_entropy",
    "acc_y_dyn__katz_fractal_dimension",
    "acc_x_dyn__cardinality",
    "acc_x_dyn__variance_of_absolute_differences",
]


# ============================================================================
# FEATURE PRIMITIVES
# ============================================================================

def _as_float_array(x):
    """Ensure input is float numpy array."""
    return np.asarray(x, dtype=float)


def harmonic_mean_of_abs(x: np.ndarray) -> float:
    """Harmonic mean of absolute values."""
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    abs_x = np.abs(x)
    denom = np.mean(1.0 / np.clip(abs_x, 1e-12, None))
    return 1.0 / denom if denom > 0 else np.nan


def quantile(x: np.ndarray, q: float) -> float:
    """q-th quantile."""
    x = _as_float_array(x)
    return float(np.quantile(x, q)) if x.size > 0 else np.nan


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2 * np.std(np.random.randn(100))) -> float:
    """Sample entropy (SampEn)."""
    x = _as_float_array(x)
    if x.size < m + 1:
        return np.nan
    
    if r == 0:
        r = 0.2 * np.std(x) if np.std(x) > 0 else 0.1
    
    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = [x[j:j + m] for j in range(len(x) - m + 1)]
        C = sum(1 for i in range(len(patterns)) 
                for j in range(i + 1, len(patterns))
                if _maxdist(patterns[i], patterns[j], m) <= r)
        return C
    
    return -np.log(_phi(m + 1) / max(_phi(m), 1)) if _phi(m) > 0 else np.nan


def sum_of_absolute_changes(x: np.ndarray) -> float:
    """Sum of absolute differences."""
    x = _as_float_array(x)
    return float(np.sum(np.abs(np.diff(x)))) if x.size > 1 else np.nan


def avg_amplitude_change(x: np.ndarray) -> float:
    """Average absolute change."""
    x = _as_float_array(x)
    return float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else np.nan


def variance_of_absolute_differences(x: np.ndarray) -> float:
    """Variance of absolute differences."""
    x = _as_float_array(x)
    if x.size < 2:
        return np.nan
    d = np.abs(np.diff(x))
    return float(np.var(d, ddof=0))


def max_(x: np.ndarray) -> float:
    """Maximum value."""
    x = _as_float_array(x)
    return float(np.max(x)) if x.size > 0 else np.nan


def tsallis_entropy(x: np.ndarray, q: float = 2.0) -> float:
    """Tsallis entropy."""
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    
    # Compute histogram
    hist, _ = np.histogram(x, bins=max(10, int(np.sqrt(x.size))))
    p = hist / np.sum(hist)
    p = p[p > 0]
    
    if q == 1:
        return float(-np.sum(p * np.log(p)))
    else:
        return float((1 - np.sum(p ** q)) / (q - 1))


def katz_fractal_dimension(x: np.ndarray) -> float:
    """Katz fractal dimension."""
    x = _as_float_array(x)
    n = x.size
    if n < 2:
        return np.nan
    
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    
    if L < 1e-12 or d < 1e-12:
        return np.nan
    
    return float(np.log10(n) / (np.log10(d / L) + np.log10(n)))


def cardinality(x: np.ndarray, round_decimals: int = 3) -> float:
    """Cardinality (unique rounded values)."""
    x = _as_float_array(x)
    if x.size == 0:
        return np.nan
    xr = np.round(x, round_decimals)
    return float(np.unique(xr).size)


def approximate_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """Approximate entropy (ApEn)."""
    x = _as_float_array(x)
    if x.size < m + 1:
        return np.nan
    
    if r is None:
        r = 0.2 * np.std(x) if np.std(x) > 0 else 0.1
    
    def _maxdist(xi, xj):
        return max(abs(ua - va) for ua, va in zip(xi, xj))
    
    def _phi(m):
        patterns = [x[j:j + m] for j in range(len(x) - m + 1)]
        C = sum(1 for i in range(len(patterns))
                for j in range(len(patterns))
                if _maxdist(patterns[i], patterns[j]) <= r)
        return np.log(C / len(patterns))
    
    return abs(_phi(m) - _phi(m + 1))


# ============================================================================
# EXTRACTION
# ============================================================================

def _extract_window_features(signal: np.ndarray, sig_name: str) -> dict:
    """Extract features from single acceleration window."""
    if signal.size == 0:
        return {}
    
    features = {}
    
    # Basic statistics
    features[f"{sig_name}__max"] = max_(signal)
    features[f"{sig_name}__harmonic_mean_of_abs"] = harmonic_mean_of_abs(signal)
    features[f"{sig_name}__quantile_0.3"] = quantile(signal, 0.3)
    features[f"{sig_name}__quantile_0.4"] = quantile(signal, 0.4)
    features[f"{sig_name}__quantile_0.6"] = quantile(signal, 0.6)
    features[f"{sig_name}__quantile_0.9"] = quantile(signal, 0.9)
    
    # Complexity
    features[f"{sig_name}__sample_entropy"] = sample_entropy(signal)
    features[f"{sig_name}__approximate_entropy_0.1"] = approximate_entropy(signal, r=0.1)
    features[f"{sig_name}__tsallis_entropy"] = tsallis_entropy(signal)
    
    # Change metrics
    features[f"{sig_name}__sum_of_absolute_changes"] = sum_of_absolute_changes(signal)
    features[f"{sig_name}__avg_amplitude_change"] = avg_amplitude_change(signal)
    features[f"{sig_name}__variance_of_absolute_differences"] = variance_of_absolute_differences(signal)
    
    # Distribution
    features[f"{sig_name}__cardinality"] = cardinality(signal)
    features[f"{sig_name}__katz_fractal_dimension"] = katz_fractal_dimension(signal)
    
    return features


def extract_imu_features(
    imu_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract IMU features for each window.
    
    Args:
        imu_df: Preprocessed IMU DataFrame with columns: t_unix, acc_x_dyn, acc_y_dyn, acc_z_dyn
        windows_df: Windows DataFrame with columns: start_idx, end_idx, ...
    
    Returns:
        DataFrame with one row per window + extracted features
    """
    rows = []
    
    for _, win in windows_df.iterrows():
        start = int(win["start_idx"])
        end = int(win["end_idx"])
        
        ax = imu_df.iloc[start:end]["acc_x_dyn"].values
        ay = imu_df.iloc[start:end]["acc_y_dyn"].values
        az = imu_df.iloc[start:end]["acc_z_dyn"].values
        
        features = {}
        features.update(_extract_window_features(ax, "acc_x_dyn"))
        features.update(_extract_window_features(ay, "acc_y_dyn"))
        features.update(_extract_window_features(az, "acc_z_dyn"))
        
        # Add window metadata
        for col in windows_df.columns:
            features[col] = win[col]
        
        rows.append(features)
    
    result = pd.DataFrame(rows)
    logger.info(f"Extracted IMU features: {result.shape[0]} windows x {result.shape[1]} columns")
    
    return result
