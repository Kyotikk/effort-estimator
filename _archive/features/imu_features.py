"""
IMU/Accelerometer feature extraction for physical activity quantification.

Computes movement-based features reflecting external mechanical load:
- Acceleration magnitude statistics
- Movement duration and intensity
- Step counting and cadence
- Gyroscope features (if available)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class IMUFeatureExtractor:
    """
    Extract movement features from accelerometer/gyroscope data.
    """
    
    def __init__(self, sampling_rate: float = 50.0, gravity: float = 9.81):
        """
        Args:
            sampling_rate: IMU sampling frequency (Hz)
            gravity: Gravitational constant for unit conversion (m/s²)
        """
        self.sampling_rate = sampling_rate
        self.gravity = gravity
    
    def compute_magnitude(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute 3D magnitude from x, y, z components.
        
        Args:
            x, y, z: Acceleration or gyroscope components
            
        Returns:
            Magnitude vector
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    def highpass_filter(self, signal: np.ndarray, cutoff: float = 0.3) -> np.ndarray:
        """
        High-pass filter to remove gravity component from acceleration.
        
        Args:
            signal: Raw signal
            cutoff: Cutoff frequency (Hz)
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        
        b, a = butter(4, normal_cutoff, btype='high')
        filtered = filtfilt(b, a, signal)
        
        return filtered
    
    def detect_steps(self, acc_magnitude: np.ndarray, 
                    threshold_factor: float = 1.5) -> Tuple[int, np.ndarray]:
        """
        Simple step detection from acceleration magnitude.
        
        Args:
            acc_magnitude: Acceleration magnitude (after gravity removal)
            threshold_factor: Threshold multiplier for peak detection
            
        Returns:
            step_count: Number of detected steps
            step_indices: Indices of detected steps
        """
        from scipy.signal import find_peaks
        
        # Compute dynamic threshold
        threshold = threshold_factor * np.std(acc_magnitude)
        
        # Minimum time between steps (max ~200 steps/min = 3.33 steps/sec)
        min_distance = int(self.sampling_rate * 0.3)  # 300ms
        
        peaks, _ = find_peaks(acc_magnitude, 
                            height=threshold,
                            distance=min_distance)
        
        return len(peaks), peaks
    
    def compute_cadence(self, step_indices: np.ndarray, duration_sec: float) -> float:
        """
        Compute walking/running cadence (steps per minute).
        
        Args:
            step_indices: Indices of detected steps
            duration_sec: Duration of measurement window (seconds)
            
        Returns:
            Cadence in steps/minute
        """
        if duration_sec <= 0 or len(step_indices) == 0:
            return 0.0
        
        cadence = (len(step_indices) / duration_sec) * 60.0
        return cadence
    
    def extract_features(self, acc_x: np.ndarray, 
                        acc_y: np.ndarray, 
                        acc_z: np.ndarray,
                        gyro_x: Optional[np.ndarray] = None,
                        gyro_y: Optional[np.ndarray] = None,
                        gyro_z: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract movement features from IMU data.
        
        Args:
            acc_x, acc_y, acc_z: Acceleration components (g or m/s²)
            gyro_x, gyro_y, gyro_z: Optional gyroscope components (rad/s or deg/s)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # === ACCELEROMETER FEATURES ===
        
        # Raw magnitude
        acc_magnitude_raw = self.compute_magnitude(acc_x, acc_y, acc_z)
        features['acc_raw_mean'] = np.mean(acc_magnitude_raw)
        
        # Remove gravity (subtract ~1g or use high-pass filter)
        # Method 1: Subtract mean (static gravity)
        acc_magnitude_nograv = acc_magnitude_raw - np.mean(acc_magnitude_raw)
        
        # Method 2: High-pass filter (better for dynamic movements)
        acc_x_filt = self.highpass_filter(acc_x)
        acc_y_filt = self.highpass_filter(acc_y)
        acc_z_filt = self.highpass_filter(acc_z)
        acc_magnitude = self.compute_magnitude(acc_x_filt, acc_y_filt, acc_z_filt)
        
        # Magnitude statistics
        features['acc_mag_mean'] = np.mean(np.abs(acc_magnitude))
        features['acc_mag_std'] = np.std(acc_magnitude)
        features['acc_mag_max'] = np.max(np.abs(acc_magnitude))
        features['acc_mag_min'] = np.min(np.abs(acc_magnitude))
        features['acc_mag_range'] = features['acc_mag_max'] - features['acc_mag_min']
        
        # Movement intensity (integral of acceleration)
        duration_sec = len(acc_magnitude) / self.sampling_rate
        features['acc_mag_integral'] = np.sum(np.abs(acc_magnitude)) / self.sampling_rate
        
        # Per-axis statistics (can reveal posture/orientation)
        for axis, data in [('x', acc_x_filt), ('y', acc_y_filt), ('z', acc_z_filt)]:
            features[f'acc_{axis}_mean'] = np.mean(np.abs(data))
            features[f'acc_{axis}_std'] = np.std(data)
        
        # Step detection and cadence
        step_count, step_indices = self.detect_steps(acc_magnitude)
        features['steps_sum'] = step_count
        features['cadence_mean'] = self.compute_cadence(step_indices, duration_sec)
        
        # Movement duration (percentage of time with significant movement)
        movement_threshold = 0.5 * features['acc_mag_std']
        movement_mask = np.abs(acc_magnitude) > movement_threshold
        features['movement_duration'] = np.sum(movement_mask) / len(acc_magnitude)
        
        logger.debug(f"IMU features: acc_mag {features['acc_mag_mean']:.3f} ± {features['acc_mag_std']:.3f}, "
                    f"steps {features['steps_sum']}, cadence {features['cadence_mean']:.1f} steps/min")
        
        # === GYROSCOPE FEATURES (if available) ===
        
        if gyro_x is not None and gyro_y is not None and gyro_z is not None:
            gyro_magnitude = self.compute_magnitude(gyro_x, gyro_y, gyro_z)
            
            features['gyro_mag_mean'] = np.mean(np.abs(gyro_magnitude))
            features['gyro_mag_std'] = np.std(gyro_magnitude)
            features['gyro_mag_max'] = np.max(np.abs(gyro_magnitude))
            
            # Per-axis gyroscope
            for axis, data in [('x', gyro_x), ('y', gyro_y), ('z', gyro_z)]:
                features[f'gyro_{axis}_mean'] = np.mean(np.abs(data))
                features[f'gyro_{axis}_std'] = np.std(data)
            
            logger.debug(f"Gyro features: gyro_mag {features['gyro_mag_mean']:.3f}")
        else:
            # Fill with NaN if gyro not available
            for key in ['gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max',
                       'gyro_x_mean', 'gyro_x_std', 'gyro_y_mean', 
                       'gyro_y_std', 'gyro_z_mean', 'gyro_z_std']:
                features[key] = np.nan
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                       acc_cols: Tuple[str, str, str] = ('x', 'y', 'z'),
                                       gyro_cols: Optional[Tuple[str, str, str]] = None) -> Dict[str, float]:
        """
        Extract features from a DataFrame with IMU data.
        
        Args:
            df: DataFrame with accelerometer (and optionally gyroscope) data
            acc_cols: Column names for acceleration (x, y, z)
            gyro_cols: Optional column names for gyroscope (x, y, z)
            
        Returns:
            Dictionary of features
        """
        acc_x = df[acc_cols[0]].values
        acc_y = df[acc_cols[1]].values
        acc_z = df[acc_cols[2]].values
        
        gyro_x, gyro_y, gyro_z = None, None, None
        if gyro_cols is not None:
            if all(col in df.columns for col in gyro_cols):
                gyro_x = df[gyro_cols[0]].values
                gyro_y = df[gyro_cols[1]].values
                gyro_z = df[gyro_cols[2]].values
        
        return self.extract_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)


def load_corsano_acc(filepath: str) -> pd.DataFrame:
    """
    Load accelerometer data from Corsano sensor CSV/gzipped CSV.
    
    Args:
        filepath: Path to accelerometer CSV (e.g., corsano_wrist_acc/2025-12-04.csv.gz)
        
    Returns:
        DataFrame with accelerometer data
    """
    logger.info(f"Loading accelerometer from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples with columns: {df.columns.tolist()}")
    return df
