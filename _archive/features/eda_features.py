"""
Electrodermal Activity (EDA) feature extraction for sympathetic arousal.

Computes EDA-based features reflecting sympathetic nervous system activity:
- Tonic level (skin conductance level, SCL)
- Phasic responses (skin conductance responses, SCR)
- SCR frequency, amplitude, and rise time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, filtfilt, find_peaks
import logging

logger = logging.getLogger(__name__)


class EDAFeatureExtractor:
    """
    Extract electrodermal activity features for sympathetic arousal measurement.
    """
    
    def __init__(self, sampling_rate: float = 4.0):
        """
        Args:
            sampling_rate: EDA sampling frequency (Hz), typically low (1-8 Hz)
        """
        self.sampling_rate = sampling_rate
    
    def lowpass_filter(self, eda_signal: np.ndarray, cutoff: float = 1.0) -> np.ndarray:
        """
        Low-pass filter EDA signal to extract tonic (SCL) component.
        
        Args:
            eda_signal: Raw EDA signal (microsiemens, μS)
            cutoff: Cutoff frequency (Hz)
            
        Returns:
            Tonic SCL component
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        
        if normal_cutoff >= 1.0:
            logger.warning(f"Cutoff {cutoff} Hz too high for sampling rate {self.sampling_rate} Hz")
            return eda_signal
        
        b, a = butter(3, normal_cutoff, btype='low')
        scl = filtfilt(b, a, eda_signal)
        
        return scl
    
    def decompose_eda(self, eda_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose EDA into tonic (SCL) and phasic (SCR) components.
        
        Simple method: SCL = low-pass filtered, SCR = signal - SCL
        
        Args:
            eda_signal: Raw EDA signal
            
        Returns:
            scl: Skin conductance level (tonic)
            scr: Skin conductance response (phasic)
        """
        # Extract tonic component
        scl = self.lowpass_filter(eda_signal, cutoff=0.05)  # Very slow variations
        
        # Phasic = original - tonic
        scr = eda_signal - scl
        
        return scl, scr
    
    def detect_scr_peaks(self, scr: np.ndarray, 
                        min_amplitude: float = 0.01,
                        min_distance_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect skin conductance response (SCR) peaks.
        
        Args:
            scr: Phasic SCR component
            min_amplitude: Minimum SCR amplitude (μS)
            min_distance_sec: Minimum time between SCRs (seconds)
            
        Returns:
            peak_indices: Indices of detected SCR peaks
            peak_amplitudes: Amplitudes of detected SCRs
        """
        min_distance = int(self.sampling_rate * min_distance_sec)
        
        peaks, properties = find_peaks(scr, 
                                      height=min_amplitude,
                                      distance=min_distance)
        
        peak_amplitudes = properties['peak_heights'] if len(peaks) > 0 else np.array([])
        
        logger.debug(f"Detected {len(peaks)} SCR peaks")
        return peaks, peak_amplitudes
    
    def compute_scr_features(self, scr: np.ndarray, 
                            peak_indices: np.ndarray,
                            peak_amplitudes: np.ndarray,
                            duration_sec: float) -> Dict[str, float]:
        """
        Compute features from detected SCRs.
        
        Args:
            scr: Phasic SCR signal
            peak_indices: Indices of SCR peaks
            peak_amplitudes: Amplitudes of SCR peaks
            duration_sec: Duration of measurement window
            
        Returns:
            Dictionary of SCR features
        """
        features = {}
        
        if len(peak_indices) == 0:
            features['eda_scr_count'] = 0
            features['eda_scr_rate'] = 0.0  # SCRs per minute
            features['eda_scr_mean_amplitude'] = 0.0
            features['eda_scr_max_amplitude'] = 0.0
            features['eda_scr_sum_amplitude'] = 0.0
        else:
            features['eda_scr_count'] = len(peak_indices)
            features['eda_scr_rate'] = (len(peak_indices) / duration_sec) * 60.0  # per minute
            features['eda_scr_mean_amplitude'] = np.mean(peak_amplitudes)
            features['eda_scr_max_amplitude'] = np.max(peak_amplitudes)
            features['eda_scr_sum_amplitude'] = np.sum(peak_amplitudes)
        
        return features
    
    def extract_features(self, eda_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all EDA features from raw signal.
        
        Args:
            eda_signal: Raw EDA signal (μS)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        duration_sec = len(eda_signal) / self.sampling_rate
        
        # === TONIC FEATURES (SCL) ===
        
        scl, scr = self.decompose_eda(eda_signal)
        
        features['eda_mean'] = np.mean(eda_signal)
        features['eda_std'] = np.std(eda_signal)
        features['eda_min'] = np.min(eda_signal)
        features['eda_max'] = np.max(eda_signal)
        features['eda_range'] = features['eda_max'] - features['eda_min']
        
        # Tonic level statistics
        features['eda_scl_mean'] = np.mean(scl)
        features['eda_scl_std'] = np.std(scl)
        
        # Slope of tonic level (increasing = rising arousal)
        time_points = np.arange(len(scl)) / self.sampling_rate
        if len(scl) >= 3:
            scl_slope = np.polyfit(time_points, scl, 1)[0]
            features['eda_slope'] = scl_slope  # μS per second
        else:
            features['eda_slope'] = 0.0
        
        # === PHASIC FEATURES (SCR) ===
        
        peak_indices, peak_amplitudes = self.detect_scr_peaks(scr)
        scr_features = self.compute_scr_features(scr, peak_indices, peak_amplitudes, duration_sec)
        features.update(scr_features)
        
        logger.debug(f"EDA features: mean {features['eda_mean']:.3f} μS, "
                    f"SCR count {features['eda_scr_count']}, "
                    f"rate {features['eda_scr_rate']:.2f} /min")
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                       value_col: str = 'value') -> Dict[str, float]:
        """
        Extract features from a DataFrame with EDA data.
        
        Args:
            df: DataFrame with EDA values
            value_col: Name of column containing EDA values
            
        Returns:
            Dictionary of features
        """
        eda_signal = df[value_col].values
        return self.extract_features(eda_signal)


def load_corsano_eda(filepath: str) -> pd.DataFrame:
    """
    Load EDA/GSR data from Corsano bioz_emography sensor CSV/gzipped CSV.
    
    Note: corsano_bioz_emography may contain multiple signals.
    Adapt column selection based on actual data format.
    
    Args:
        filepath: Path to EDA CSV (e.g., corsano_bioz_emography/2025-12-04.csv.gz)
        
    Returns:
        DataFrame with EDA data
    """
    logger.info(f"Loading EDA from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples with columns: {df.columns.tolist()}")
    return df


def load_respiration_proxy(filepath: str) -> pd.DataFrame:
    """
    If corsano_bioz_emography contains respiration signals, load them.
    
    This is a placeholder - adapt based on actual column structure.
    
    Args:
        filepath: Path to bioz CSV
        
    Returns:
        DataFrame with respiration proxy data
    """
    logger.info(f"Loading respiration proxy from {filepath}")
    df = pd.read_csv(filepath)
    
    # Identify respiration-related columns (adapt as needed)
    resp_cols = [col for col in df.columns if 'resp' in col.lower() or 'breath' in col.lower()]
    
    if resp_cols:
        logger.info(f"Found respiration columns: {resp_cols}")
    else:
        logger.warning("No respiration columns identified")
    
    return df
