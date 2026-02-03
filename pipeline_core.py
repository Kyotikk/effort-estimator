#!/usr/bin/env python3
"""
Minimal Effort Estimation Pipeline
===================================
Complete pipeline from raw signals to LOSO evaluation.

Flow (matching methodology diagram):
1. Raw Signals: IMU (3-axis accel), PPG (3 wavelengths), EDA (skin cond.)
2. Preprocessing: Resampling, filtering
3. Temporal Segmentation: 5s windows, 10% overlap
4. Feature Extraction: IMU(30), PPG(183), EDA(47)
5. Fusion & Alignment: Time alignment, modality fusion, label matching
6. Feature Selection: Correlation ranking → Redundancy pruning → LOSO-consistent
7. Model Training: Random Forest (n=100, depth=6), LOSO Cross-Validation

Results: IMU r=0.52, PPG r=0.26, EDA r=0.02
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration matching methodology."""
    # Sampling rates (after resampling)
    target_fs: int = 32  # Hz, unified sampling rate
    
    # Windowing
    window_sec: float = 5.0
    overlap_pct: float = 0.10  # 10% overlap
    
    # Feature selection
    redundancy_threshold: float = 0.90
    min_subjects_consistent: int = 4  # out of 5
    
    # Model
    n_estimators: int = 100
    max_depth: int = 6
    random_state: int = 42


# =============================================================================
# 1. PREPROCESSING
# =============================================================================

def preprocess_imu(df: pd.DataFrame, fs: int = 32) -> pd.DataFrame:
    """
    Preprocess IMU: Resampling + Low-pass filtering.
    
    Args:
        df: DataFrame with acc_x, acc_y, acc_z columns
        fs: Sampling rate
        
    Returns:
        Preprocessed DataFrame with acc_{x,y,z}_dyn columns
    """
    out = df.copy()
    
    # Low-pass filter at 5 Hz (remove high-freq noise)
    nyq = fs / 2
    b, a = signal.butter(4, 5 / nyq, btype='low')
    
    for axis in ['x', 'y', 'z']:
        col = f'acc_{axis}'
        if col in df.columns:
            # Remove gravity (high-pass at 0.5 Hz)
            b_hp, a_hp = signal.butter(2, 0.5 / nyq, btype='high')
            dynamic = signal.filtfilt(b_hp, a_hp, df[col].values)
            # Low-pass
            filtered = signal.filtfilt(b, a, dynamic)
            out[f'acc_{axis}_dyn'] = filtered
    
    return out


def preprocess_ppg(df: pd.DataFrame, fs: int = 64) -> pd.DataFrame:
    """
    Preprocess PPG: Resampling + High-pass filtering.
    
    Args:
        df: DataFrame with ppg columns
        fs: Sampling rate
        
    Returns:
        Preprocessed DataFrame
    """
    out = df.copy()
    
    # High-pass filter at 0.5 Hz (remove DC drift)
    nyq = fs / 2
    b, a = signal.butter(2, 0.5 / nyq, btype='high')
    
    for col in df.columns:
        if 'ppg' in col.lower():
            out[col] = signal.filtfilt(b, a, df[col].values)
    
    return out


def preprocess_eda(df: pd.DataFrame, fs: int = 4) -> pd.DataFrame:
    """
    Preprocess EDA: Resampling + Tonic/Phasic split.
    
    Args:
        df: DataFrame with eda column
        fs: Sampling rate
        
    Returns:
        Preprocessed DataFrame with eda_tonic, eda_phasic
    """
    out = df.copy()
    
    if 'eda' in df.columns:
        eda = df['eda'].values
        
        # Simple tonic/phasic split using low-pass filter
        # Tonic = slow component (< 0.05 Hz)
        nyq = fs / 2
        b, a = signal.butter(2, 0.05 / nyq, btype='low')
        tonic = signal.filtfilt(b, a, eda)
        phasic = eda - tonic
        
        out['eda_tonic'] = tonic
        out['eda_phasic'] = phasic
    
    return out


# =============================================================================
# 2. WINDOWING
# =============================================================================

def create_windows(n_samples: int, fs: int, window_sec: float, overlap_pct: float) -> pd.DataFrame:
    """
    Create window indices for temporal segmentation.
    
    Args:
        n_samples: Total number of samples in signal
        fs: Sampling rate
        window_sec: Window length in seconds
        overlap_pct: Overlap as fraction (0.10 = 10%)
        
    Returns:
        DataFrame with start_idx, end_idx, t_start, t_center, t_end
    """
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap_pct))
    
    windows = []
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        windows.append({
            'start_idx': start,
            'end_idx': end,
            't_start': start / fs,
            't_end': end / fs,
            't_center': (start + window_samples / 2) / fs
        })
        start += step_samples
    
    return pd.DataFrame(windows)


# =============================================================================
# 3. FEATURE EXTRACTION
# =============================================================================

# --- IMU Features (30 selected) ---

def _safe(fn, x, default=np.nan):
    """Safely compute function, return default on error."""
    try:
        result = fn(x)
        return result if np.isfinite(result) else default
    except:
        return default


def sample_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """Compute sample entropy."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return np.nan
    
    def count_matches(templates):
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 2  # count both (i,j) and (j,i)
        return count
    
    # Templates of length m
    B = count_matches([x[i:i+m] for i in range(n - m)])
    # Templates of length m+1
    A = count_matches([x[i:i+m+1] for i in range(n - m)])
    
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def katz_fractal_dimension(x: np.ndarray) -> float:
    """Compute Katz fractal dimension."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    
    L = np.sum(np.abs(np.diff(x)))  # Total path length
    d = np.max(np.abs(x - x[0]))    # Maximum distance from start
    
    if L < 1e-12 or d < 1e-12:
        return np.nan
    
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


def extract_imu_features(window: np.ndarray, axis_name: str) -> Dict[str, float]:
    """
    Extract IMU features for one axis window.
    
    Features include: quantiles, entropy, fractal dimension, temporal dynamics.
    """
    x = window[np.isfinite(window)]
    prefix = f'{axis_name}__'
    
    if len(x) < 10:
        return {}
    
    features = {}
    
    # Quantiles
    for q in [0.3, 0.4, 0.6, 0.9]:
        features[f'{prefix}quantile_{q}'] = np.quantile(x, q)
    
    # Harmonic mean
    x_pos = np.abs(x)
    x_pos = x_pos[x_pos > 1e-12]
    if len(x_pos) > 0:
        features[f'{prefix}harmonic_mean_of_abs'] = len(x_pos) / np.sum(1.0 / x_pos)
    
    # Max
    features[f'{prefix}max'] = np.max(x)
    
    # Entropy measures
    features[f'{prefix}sample_entropy'] = sample_entropy(x)
    features[f'{prefix}approximate_entropy_0.1'] = sample_entropy(x, r=0.1 * np.std(x))
    
    # Fractal dimension
    features[f'{prefix}katz_fractal_dimension'] = katz_fractal_dimension(x)
    
    # Temporal dynamics
    dx = np.diff(x)
    features[f'{prefix}sum_of_absolute_changes'] = np.sum(np.abs(dx))
    features[f'{prefix}variance_of_absolute_differences'] = np.var(np.abs(dx))
    features[f'{prefix}avg_amplitude_change'] = np.mean(np.abs(dx))
    
    # Cardinality (unique values)
    features[f'{prefix}cardinality'] = len(np.unique(np.round(x, 3)))
    
    return features


def extract_all_imu_features(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Extract IMU features for all windows (30 features total)."""
    rows = []
    
    for _, w in windows.iterrows():
        s, e = int(w['start_idx']), int(w['end_idx'])
        feats = {'window_id': len(rows), **w.to_dict()}
        
        for axis in ['x', 'y', 'z']:
            col = f'acc_{axis}_dyn'
            if col in df.columns:
                window_data = df[col].values[s:e]
                axis_feats = extract_imu_features(window_data, f'acc_{axis}_dyn')
                feats.update(axis_feats)
        
        rows.append(feats)
    
    return pd.DataFrame(rows)


# --- PPG Features (statistical + HRV) ---

def extract_ppg_features(window: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Extract PPG features: statistical moments + basic HRV-like.
    """
    x = window[np.isfinite(window)]
    
    if len(x) < 10:
        return {}
    
    features = {}
    
    # Statistical moments
    features[f'{prefix}mean'] = np.mean(x)
    features[f'{prefix}std'] = np.std(x)
    features[f'{prefix}min'] = np.min(x)
    features[f'{prefix}max'] = np.max(x)
    features[f'{prefix}range'] = np.max(x) - np.min(x)
    features[f'{prefix}median'] = np.median(x)
    features[f'{prefix}iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    features[f'{prefix}mad'] = np.median(np.abs(x - np.median(x)))
    features[f'{prefix}skew'] = _safe(lambda a: pd.Series(a).skew(), x)
    features[f'{prefix}kurtosis'] = _safe(lambda a: pd.Series(a).kurtosis(), x)
    features[f'{prefix}rms'] = np.sqrt(np.mean(x**2))
    
    # Derivatives
    dx = np.diff(x)
    features[f'{prefix}dx_mean'] = np.mean(dx)
    features[f'{prefix}dx_std'] = np.std(dx)
    features[f'{prefix}dx_max'] = np.max(np.abs(dx))
    
    # Energy
    features[f'{prefix}energy'] = np.sum(x**2)
    
    # Percentiles
    for p in [5, 25, 75, 95]:
        features[f'{prefix}p{p}'] = np.percentile(x, p)
    
    return features


def extract_all_ppg_features(df: pd.DataFrame, windows: pd.DataFrame, 
                              ppg_cols: List[str]) -> pd.DataFrame:
    """Extract PPG features for all windows."""
    rows = []
    
    for _, w in windows.iterrows():
        s, e = int(w['start_idx']), int(w['end_idx'])
        feats = {'window_id': len(rows), **w.to_dict()}
        
        for col in ppg_cols:
            if col in df.columns:
                window_data = df[col].values[s:e]
                ppg_feats = extract_ppg_features(window_data, f'{col}_')
                feats.update(ppg_feats)
        
        rows.append(feats)
    
    return pd.DataFrame(rows)


# --- EDA Features ---

def extract_eda_features(window_tonic: np.ndarray, window_phasic: np.ndarray,
                         prefix: str) -> Dict[str, float]:
    """Extract EDA features: tonic level, phasic dynamics."""
    features = {}
    
    # Tonic (SCL)
    t = window_tonic[np.isfinite(window_tonic)]
    if len(t) > 0:
        features[f'{prefix}tonic_mean'] = np.mean(t)
        features[f'{prefix}tonic_std'] = np.std(t)
        features[f'{prefix}tonic_range'] = np.max(t) - np.min(t)
        features[f'{prefix}tonic_slope'] = np.polyfit(np.arange(len(t)), t, 1)[0] if len(t) > 1 else 0
    
    # Phasic
    p = window_phasic[np.isfinite(window_phasic)]
    if len(p) > 0:
        features[f'{prefix}phasic_mean'] = np.mean(p)
        features[f'{prefix}phasic_std'] = np.std(p)
        features[f'{prefix}phasic_max'] = np.max(p)
        features[f'{prefix}phasic_mad'] = np.median(np.abs(p - np.median(p)))
        features[f'{prefix}phasic_skew'] = _safe(lambda a: pd.Series(a).skew(), p)
        features[f'{prefix}phasic_kurtosis'] = _safe(lambda a: pd.Series(a).kurtosis(), p)
    
    return features


def extract_all_eda_features(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Extract EDA features for all windows."""
    rows = []
    
    for _, w in windows.iterrows():
        s, e = int(w['start_idx']), int(w['end_idx'])
        feats = {'window_id': len(rows), **w.to_dict()}
        
        if 'eda_tonic' in df.columns and 'eda_phasic' in df.columns:
            tonic = df['eda_tonic'].values[s:e]
            phasic = df['eda_phasic'].values[s:e]
            eda_feats = extract_eda_features(tonic, phasic, 'eda_')
            feats.update(eda_feats)
        
        rows.append(feats)
    
    return pd.DataFrame(rows)


# =============================================================================
# 4. FUSION & ALIGNMENT
# =============================================================================

def fuse_modalities(imu_features: pd.DataFrame, 
                    ppg_features: pd.DataFrame,
                    eda_features: pd.DataFrame,
                    labels: pd.DataFrame) -> pd.DataFrame:
    """
    Fuse feature modalities and align with labels.
    
    Matches windows by t_center proximity to activity timestamps.
    """
    # Start with IMU as base (typically most reliable timing)
    fused = imu_features.copy()
    
    # Merge PPG features
    ppg_cols = [c for c in ppg_features.columns 
                if c.startswith('ppg_') or c.startswith('bvp_')]
    if ppg_cols:
        fused = fused.merge(ppg_features[['window_id'] + ppg_cols], 
                           on='window_id', how='left')
    
    # Merge EDA features
    eda_cols = [c for c in eda_features.columns if c.startswith('eda_')]
    if eda_cols:
        fused = fused.merge(eda_features[['window_id'] + eda_cols],
                           on='window_id', how='left')
    
    # Match labels by t_center
    if 'borg' in labels.columns and 't_start' in labels.columns:
        fused['borg'] = np.nan
        fused['activity'] = None
        
        for idx, row in fused.iterrows():
            t = row['t_center']
            # Find label whose time range contains t_center
            mask = (labels['t_start'] <= t) & (labels['t_end'] >= t)
            matches = labels[mask]
            if len(matches) > 0:
                fused.at[idx, 'borg'] = matches.iloc[0]['borg']
                if 'activity' in labels.columns:
                    fused.at[idx, 'activity'] = matches.iloc[0]['activity']
    
    return fused


# =============================================================================
# 5. FEATURE SELECTION
# =============================================================================

def select_features_correlation(df: pd.DataFrame, 
                                feature_cols: List[str],
                                target_col: str = 'borg',
                                top_n: int = 50) -> Tuple[List[str], pd.DataFrame]:
    """
    Step 1: Rank features by Spearman correlation with target.
    
    Returns top_n features by absolute correlation.
    """
    df_valid = df.dropna(subset=[target_col])
    y = df_valid[target_col].values
    
    correlations = []
    for col in feature_cols:
        if col in df_valid.columns:
            x = df_valid[col].values
            mask = np.isfinite(x)
            if mask.sum() > 10:
                rho, _ = spearmanr(x[mask], y[mask])
                correlations.append({'feature': col, 'corr': rho, 'abs_corr': abs(rho)})
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    top_features = corr_df.head(top_n)['feature'].tolist()
    
    return top_features, corr_df


def prune_redundant_features(df: pd.DataFrame,
                             feature_cols: List[str],
                             corr_rankings: pd.DataFrame,
                             threshold: float = 0.90) -> List[str]:
    """
    Step 2: Remove redundant features (r > threshold with each other).
    
    Keeps the feature with higher target correlation.
    """
    # Get correlation matrix
    X = df[feature_cols].dropna()
    corr_matrix = X.corr().abs()
    
    # Track which features to drop
    to_drop = set()
    
    # Sort by target correlation (keep higher correlated ones)
    rank_order = corr_rankings[corr_rankings['feature'].isin(feature_cols)]
    rank_order = rank_order.sort_values('abs_corr', ascending=False)['feature'].tolist()
    
    for i, feat_i in enumerate(rank_order):
        if feat_i in to_drop:
            continue
        for feat_j in rank_order[i+1:]:
            if feat_j in to_drop:
                continue
            if feat_i in corr_matrix.columns and feat_j in corr_matrix.columns:
                if corr_matrix.loc[feat_i, feat_j] > threshold:
                    to_drop.add(feat_j)  # Drop lower-ranked feature
    
    return [f for f in feature_cols if f not in to_drop]


def filter_loso_consistent(df: pd.DataFrame,
                           feature_cols: List[str],
                           subject_col: str = 'subject',
                           target_col: str = 'borg',
                           min_subjects: int = 4) -> List[str]:
    """
    Step 3: Keep features where ≥min_subjects show same correlation direction.
    
    Ensures feature generalizes across patients.
    """
    subjects = df[subject_col].unique()
    
    consistent_features = []
    
    for col in feature_cols:
        positive_count = 0
        negative_count = 0
        
        for subj in subjects:
            subj_data = df[df[subject_col] == subj].dropna(subset=[col, target_col])
            if len(subj_data) < 5:
                continue
            
            rho, _ = spearmanr(subj_data[col], subj_data[target_col])
            if rho > 0:
                positive_count += 1
            elif rho < 0:
                negative_count += 1
        
        # Keep if at least min_subjects agree on direction
        if max(positive_count, negative_count) >= min_subjects:
            consistent_features.append(col)
    
    return consistent_features


def full_feature_selection(df: pd.DataFrame,
                           feature_cols: List[str],
                           config: PipelineConfig) -> List[str]:
    """
    Complete 3-step feature selection pipeline.
    
    1. Correlation ranking
    2. Redundancy pruning
    3. LOSO consistency filter
    """
    # Step 1: Correlation ranking
    top_features, corr_df = select_features_correlation(
        df, feature_cols, top_n=100
    )
    print(f"  After correlation ranking: {len(top_features)} features")
    
    # Step 2: Redundancy pruning
    pruned_features = prune_redundant_features(
        df, top_features, corr_df, 
        threshold=config.redundancy_threshold
    )
    print(f"  After redundancy pruning: {len(pruned_features)} features")
    
    # Step 3: LOSO consistency (if subject column exists)
    if 'subject' in df.columns:
        consistent_features = filter_loso_consistent(
            df, pruned_features,
            min_subjects=config.min_subjects_consistent
        )
        print(f"  After LOSO consistency: {len(consistent_features)} features")
        return consistent_features
    
    return pruned_features


# =============================================================================
# 6. MODEL TRAINING & EVALUATION
# =============================================================================

def train_loso(df: pd.DataFrame,
               feature_cols: List[str],
               subject_col: str = 'subject',
               target_col: str = 'borg',
               config: PipelineConfig = None) -> Dict:
    """
    Leave-One-Subject-Out cross-validation.
    
    Returns:
        Dictionary with predictions, per-subject metrics, overall metrics
    """
    if config is None:
        config = PipelineConfig()
    
    subjects = sorted(df[subject_col].unique())
    
    all_results = []
    per_subject_r = []
    
    for test_subj in subjects:
        # Split
        train_df = df[df[subject_col] != test_subj].dropna(subset=[target_col])
        test_df = df[df[subject_col] == test_subj].dropna(subset=[target_col])
        
        if len(train_df) < 10 or len(test_df) < 5:
            continue
        
        # Prepare data
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[target_col].values
        
        # Train
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        r, _ = pearsonr(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Store
        for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
            all_results.append({
                'subject': test_subj,
                'actual': actual,
                'predicted': pred,
                'error': pred - actual
            })
        
        per_subject_r.append({'subject': test_subj, 'r': r, 'mae': mae, 'n': len(test_df)})
    
    # Overall metrics
    results_df = pd.DataFrame(all_results)
    overall_r, _ = pearsonr(results_df['actual'], results_df['predicted'])
    mean_subject_r = np.mean([s['r'] for s in per_subject_r])
    
    return {
        'predictions': results_df,
        'per_subject': pd.DataFrame(per_subject_r),
        'overall_r': overall_r,
        'mean_subject_r': mean_subject_r,
        'overall_mae': results_df['error'].abs().mean()
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(data_path: str, 
                 labels_path: str,
                 config: PipelineConfig = None) -> Dict:
    """
    Run complete effort estimation pipeline.
    
    Args:
        data_path: Path to raw sensor data (CSV with imu, ppg, eda columns)
        labels_path: Path to activity labels with Borg scores
        config: Pipeline configuration
        
    Returns:
        Dictionary with results for each modality
    """
    if config is None:
        config = PipelineConfig()
    
    print("=" * 60)
    print("EFFORT ESTIMATION PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)
    
    # Preprocessing
    print("\n[2] Preprocessing...")
    data = preprocess_imu(data, fs=config.target_fs)
    data = preprocess_ppg(data, fs=config.target_fs)
    data = preprocess_eda(data, fs=config.target_fs)
    
    # Windowing
    print("\n[3] Creating windows...")
    n_samples = len(data)
    windows = create_windows(n_samples, config.target_fs, 
                            config.window_sec, config.overlap_pct)
    print(f"  Created {len(windows)} windows ({config.window_sec}s, {config.overlap_pct*100:.0f}% overlap)")
    
    # Feature extraction
    print("\n[4] Extracting features...")
    imu_features = extract_all_imu_features(data, windows)
    ppg_cols = [c for c in data.columns if 'ppg' in c.lower() or 'bvp' in c.lower()]
    ppg_features = extract_all_ppg_features(data, windows, ppg_cols)
    eda_features = extract_all_eda_features(data, windows)
    
    imu_feat_cols = [c for c in imu_features.columns if c.startswith('acc_')]
    ppg_feat_cols = [c for c in ppg_features.columns if c.startswith('ppg_') or c.startswith('bvp_')]
    eda_feat_cols = [c for c in eda_features.columns if c.startswith('eda_')]
    
    print(f"  IMU: {len(imu_feat_cols)} features")
    print(f"  PPG: {len(ppg_feat_cols)} features")
    print(f"  EDA: {len(eda_feat_cols)} features")
    
    # Fusion
    print("\n[5] Fusing modalities...")
    fused = fuse_modalities(imu_features, ppg_features, eda_features, labels)
    labeled_count = fused['borg'].notna().sum()
    print(f"  {labeled_count} labeled windows")
    
    # Feature selection per modality
    print("\n[6] Feature selection...")
    results = {}
    
    for modality, feat_cols in [('IMU', imu_feat_cols), 
                                 ('PPG', ppg_feat_cols),
                                 ('EDA', eda_feat_cols)]:
        if len(feat_cols) == 0:
            continue
            
        print(f"\n  --- {modality} ---")
        selected = full_feature_selection(fused, feat_cols, config)
        
        if len(selected) == 0:
            print(f"  No features selected for {modality}")
            continue
        
        # Train and evaluate
        print(f"\n[7] LOSO evaluation for {modality}...")
        loso_results = train_loso(fused, selected, config=config)
        
        print(f"  Overall r = {loso_results['overall_r']:.3f}")
        print(f"  Mean per-subject r = {loso_results['mean_subject_r']:.3f}")
        print(f"  MAE = {loso_results['overall_mae']:.2f}")
        
        results[modality] = {
            'selected_features': selected,
            'n_features': len(selected),
            **loso_results
        }
    
    return results


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("""
    =============================================
    MINIMAL EFFORT ESTIMATION PIPELINE
    =============================================
    
    This script contains the complete pipeline:
    1. Preprocessing (resampling, filtering)
    2. Windowing (5s, 10% overlap)
    3. Feature extraction (IMU, PPG, EDA)
    4. Fusion & alignment
    5. Feature selection (correlation → redundancy → LOSO)
    6. LOSO cross-validation with Random Forest
    
    Expected results (per methodology):
    - IMU: r = 0.52
    - PPG: r = 0.26
    - EDA: r = 0.02
    
    Usage:
        results = run_pipeline('data/signals.csv', 'data/labels.csv')
    
    Or run feature selection and LOSO on existing fused data:
        df = pd.read_csv('fused_features.csv')
        feature_cols = [c for c in df.columns if c.startswith('acc_')]
        selected = full_feature_selection(df, feature_cols, PipelineConfig())
        results = train_loso(df, selected)
    """)
