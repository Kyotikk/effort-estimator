#!/usr/bin/env python3
"""
Extract features from all sensors for each ADL segment and align with RMSSD labels.

This script:
1. Loads ADL timeline
2. For each ADL segment:
   - Extracts PPG features (heart rate level)
   - Extracts IMU features (movement)
   - Extracts EDA features (sympathetic arousal)
3. Matches with RMSSD-based effort labels
4. Outputs aligned dataset ready for training

Usage:
    python scripts/align_features_labels.py \\
        --session-path /path/to/parsingsim3/sim_elderly3 \\
        --output data/aligned/parsingsim3_sim_elderly3_aligned.csv \\
        --session-id parsingsim3_sim_elderly3
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.adl_parser import parse_adl_timeline
from features.ppg_features import PPGFeatureExtractor
from features.imu_features import IMUFeatureExtractor
from features.eda_features import EDAFeatureExtractor
from ecg.preprocessing import ECGPreprocessor, load_vivalnk_ecg
from ml.labels.rmssd_label import RMSSDLabeler


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_sensor_data_for_window(filepath: str, start_time: float, end_time: float,
                                sensor_start_time: float,
                                time_col: str = 'time') -> pd.DataFrame:
    """
    Load sensor data for a specific time window.
    
    Args:
        filepath: Path to sensor CSV file
        start_time: Window start time (relative seconds from session start)
        end_time: Window end time (relative seconds from session start)
        sensor_start_time: Sensor's absolute start time (Unix timestamp)
        time_col: Name of timestamp column
        
    Returns:
        DataFrame with data in time window
    """
    df = pd.read_csv(filepath)
    
    if time_col not in df.columns:
        # Try to find time column
        for col in ['timestamp', 'unix_time', 'time_s']:
            if col in df.columns:
                time_col = col
                break
    
    # Convert relative times to absolute times based on sensor start
    abs_start = sensor_start_time + start_time
    abs_end = sensor_start_time + end_time
    
    # Filter by time window
    mask = (df[time_col] >= abs_start) & (df[time_col] <= abs_end)
    return df[mask].copy()


def extract_adl_features(session_path: Path, adl_row: pd.Series, 
                        sensor_start_times: dict) -> dict:
    """
    Extract all features for a single ADL segment.
    
    Args:
        session_path: Path to session directory
        adl_row: Row from ADL dataframe with start_time, end_time, etc.
        sensor_start_times: Dict mapping sensor name to Unix start time
        
    Returns:
        Dictionary of features
    """
    logger = logging.getLogger(__name__)
    
    # ADL times are relative (seconds from session start)
    start_rel = adl_row['start_time']
    end_rel = adl_row['end_time']
    
    features = {
        'adl_id': adl_row['adl_id'],
        'adl_name': adl_row['adl_name'],
        'start_time': start_rel,
        'end_time': end_rel,
        'duration_sec': adl_row['duration_sec'],
        'borg_rpe': adl_row.get('borg_rpe', np.nan)
    }
    
    # === PPG FEATURES ===
    try:
        ppg_file = session_path / 'corsano_wrist_ppg2_green_6' / '2025-12-04.csv.gz'
        if ppg_file.exists():
            ppg_df = load_sensor_data_for_window(
                str(ppg_file), start_rel, end_rel, 
                sensor_start_times['ppg']
            )
            
            if len(ppg_df) > 100:  # Need sufficient data
                # Convert value column to numeric
                ppg_df['value'] = pd.to_numeric(ppg_df['value'], errors='coerce')
                ppg_df = ppg_df.dropna(subset=['value'])
                
                extractor = PPGFeatureExtractor(sampling_rate=64.0)
                ppg_features = extractor.extract_features(ppg_df['value'].values)
                features.update(ppg_features)
                logger.debug(f"Extracted PPG features for ADL {adl_row['adl_id']}")
            else:
                logger.warning(f"Insufficient PPG data for ADL {adl_row['adl_id']} ({len(ppg_df)} samples)")
                for key in ['ppg_hr_mean', 'ppg_hr_max', 'ppg_hr_min', 'ppg_hr_std', 
                           'ppg_hr_slope', 'ppg_hr_range', 'ppg_signal_quality', 'ppg_n_beats']:
                    features[key] = np.nan
    except Exception as e:
        logger.error(f"PPG extraction failed for ADL {adl_row['adl_id']}: {e}")
        for key in ['ppg_hr_mean', 'ppg_hr_max', 'ppg_hr_min', 'ppg_hr_std', 
                   'ppg_hr_slope', 'ppg_hr_range', 'ppg_signal_quality', 'ppg_n_beats']:
            features[key] = np.nan
    
    # === IMU FEATURES ===
    try:
        imu_file = session_path / 'corsano_wrist_acc' / '2025-12-04.csv.gz'
        if imu_file.exists():
            imu_df = load_sensor_data_for_window(
                str(imu_file), start_rel, end_rel,
                sensor_start_times['imu']
            )
            
            if len(imu_df) > 100:
                # Convert acc columns to numeric
                for col in ['accX', 'accY', 'accZ']:
                    if col in imu_df.columns:
                        imu_df[col] = pd.to_numeric(imu_df[col], errors='coerce')
                imu_df = imu_df.dropna(subset=['accX', 'accY', 'accZ'])
                
                extractor = IMUFeatureExtractor(sampling_rate=50.0)
                imu_features = extractor.extract_features(
                    imu_df['accX'].values,
                    imu_df['accY'].values,
                    imu_df['accZ'].values
                )
                features.update(imu_features)
                logger.debug(f"Extracted IMU features for ADL {adl_row['adl_id']}")
            else:
                logger.warning(f"Insufficient IMU data for ADL {adl_row['adl_id']} ({len(imu_df)} samples)")
                for key in ['acc_raw_mean', 'acc_mag_mean', 'acc_mag_std', 'acc_mag_max', 
                           'acc_mag_integral', 'steps_sum', 'cadence_mean', 'movement_duration']:
                    features[key] = np.nan
    except Exception as e:
        logger.error(f"IMU extraction failed for ADL {adl_row['adl_id']}: {e}")
        for key in ['acc_raw_mean', 'acc_mag_mean', 'acc_mag_std', 'acc_mag_max', 
                   'acc_mag_integral', 'steps_sum', 'cadence_mean', 'movement_duration']:
            features[key] = np.nan
    
    # === EDA FEATURES ===
    try:
        eda_file = session_path / 'corsano_bioz_emography' / '2025-12-04.csv.gz'
        if eda_file.exists():
            eda_df = load_sensor_data_for_window(
                str(eda_file), start_rel, end_rel,
                sensor_start_times['eda']
            )
            
            if len(eda_df) > 2:  # EDA has low sampling rate, need at least 2-3 samples
                # Use stress_skin column
                if 'stress_skin' in eda_df.columns:
                    eda_df['stress_skin'] = pd.to_numeric(eda_df['stress_skin'], errors='coerce')
                    eda_df = eda_df.dropna(subset=['stress_skin'])
                    
                    if len(eda_df) > 2:
                        extractor = EDAFeatureExtractor(sampling_rate=0.05)  # ~1 sample per 20 sec
                        eda_features = extractor.extract_features(eda_df['stress_skin'].values)
                        features.update(eda_features)
                        logger.debug(f"Extracted EDA features for ADL {adl_row['adl_id']} ({len(eda_df)} samples)")
                    else:
                        logger.debug(f"Insufficient valid EDA data for ADL {adl_row['adl_id']}")
                        for key in ['eda_mean', 'eda_std', 'eda_slope', 'eda_scr_count', 'eda_scr_rate']:
                            features[key] = np.nan
                else:
                    for key in ['eda_mean', 'eda_std', 'eda_slope', 'eda_scr_count', 'eda_scr_rate']:
                        features[key] = np.nan
            else:
                logger.debug(f"No EDA samples for ADL {adl_row['adl_id']} ({len(eda_df)} samples)")
                for key in ['eda_mean', 'eda_std', 'eda_slope', 'eda_scr_count', 'eda_scr_rate']:
                    features[key] = np.nan
    except Exception as e:
        logger.error(f"EDA extraction failed for ADL {adl_row['adl_id']}: {e}")
        for key in ['eda_mean', 'eda_std', 'eda_slope', 'eda_scr_count', 'eda_scr_rate']:
            features[key] = np.nan
    
    return features


def compute_adl_rmssd_labels(ecg_path: str, adl_df: pd.DataFrame, 
                             session_id: str) -> pd.DataFrame:
    """
    Compute RMSSD-based effort labels for each ADL segment.
    
    Args:
        ecg_path: Path to ECG file
        adl_df: DataFrame with ADL segments
        session_id: Session identifier
        
    Returns:
        DataFrame with RMSSD labels per ADL
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing RMSSD labels for ADL segments...")
    
    # Load and process ECG
    ecg_signal, sampling_rate = load_vivalnk_ecg(ecg_path)
    preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)
    r_peaks, clean_rr, valid_mask = preprocessor.process_ecg(ecg_signal)
    
    # Compute RR times (seconds from start)
    rr_times = (r_peaks[:-1] / sampling_rate)[valid_mask]
    
    # Compute RMSSD for each ADL
    labeler = RMSSDLabeler(window_size_sec=60.0)
    labels = labeler.create_session_labels(
        rr_intervals=clean_rr,
        rr_times=rr_times,
        adl_segments=adl_df,
        session_id=session_id
    )
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Align sensor features with RMSSD labels for each ADL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--session-path', required=True, type=str,
                       help='Path to session directory (e.g., .../parsingsim3/sim_elderly3)')
    parser.add_argument('--output', required=True, type=str,
                       help='Output CSV file for aligned dataset')
    parser.add_argument('--session-id', required=True, type=str,
                       help='Session identifier')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    session_path = Path(args.session_path)
    
    logger.info("="*60)
    logger.info("FEATURE EXTRACTION AND LABEL ALIGNMENT")
    logger.info("="*60)
    logger.info(f"Session: {args.session_id}")
    logger.info(f"Path: {session_path}")
    
    # Step 1: Parse ADL timeline
    logger.info("\n[1/4] Parsing ADL timeline...")
    adl_file = session_path / 'scai_app' / 'ADLs_1.csv'
    if not adl_file.exists():
        logger.error(f"ADL file not found: {adl_file}")
        sys.exit(1)
    
    adl_df = parse_adl_timeline(str(adl_file))
    logger.info(f"Parsed {len(adl_df)} ADL segments")
    
    # Step 2: Compute RMSSD labels
    logger.info("\n[2/4] Computing RMSSD-based effort labels...")
    ecg_file = session_path / 'vivalnk_vv330_ecg' / 'data_1.csv.gz'
    if not ecg_file.exists():
        logger.error(f"ECG file not found: {ecg_file}")
        sys.exit(1)
    
    # Get ECG start time for alignment - use this as the reference for all sensors
    ecg_full = pd.read_csv(ecg_file)
    ecg_start_time = ecg_full['time'].iloc[0]
    ecg_end_time = ecg_full['time'].iloc[-1]
    
    logger.info(f"ECG time range: {ecg_start_time:.2f} - {ecg_end_time:.2f}")
    logger.info(f"ECG duration: {(ecg_end_time - ecg_start_time)/60:.1f} minutes")
    
    # CRITICAL FIX: ADL timeline has different clock/timezone than sensors
    # Use ECG timing as ground truth and map ADL relative times to it
    # ADL times are relative (0 to ~2000 seconds), ECG times are absolute Unix timestamps
    
    rmssd_labels = compute_adl_rmssd_labels(str(ecg_file), adl_df, args.session_id)
    logger.info(f"Computed RMSSD for {len(rmssd_labels)} ADL segments")
    
    # Step 3: Extract features for each ADL
    logger.info("\n[3/4] Extracting features for each ADL...")
    
    # CRITICAL: Get ADL session start time (corrected for 8-hour offset)
    # The ADL times in adl_df are relative (0 to ~2000 sec), we need the absolute start
    adl_csv = pd.read_csv(adl_file)
    start_str = adl_csv.iloc[0, 0]
    if 'Start of Recording:' in start_str:
        date_str = start_str.split('Start of Recording:')[1].strip()
        from utils.adl_parser import parse_timestamp
        adl_session_start = parse_timestamp(date_str, timezone_offset_hours=+8)
        logger.info(f"ADL session starts at: {adl_session_start:.2f} ({(adl_session_start - ecg_start_time)/60:.1f} min after ECG)")
    else:
        logger.error("Could not extract ADL session start time")
        sys.exit(1)
    
    # Use ADL session start as the reference for all sensors
    sensor_start_times = {}
    sensor_start_times['ppg'] = adl_session_start
    sensor_start_times['imu'] = adl_session_start
    sensor_start_times['eda'] = adl_session_start
    sensor_start_times['ecg'] = adl_session_start
    
    logger.info(f"Using ADL session start ({adl_session_start:.2f}) as reference for all sensors")
    
    all_features = []
    for idx, adl_row in adl_df.iterrows():
        logger.info(f"Processing ADL {adl_row['adl_id']}: {adl_row['adl_name']}")
        features = extract_adl_features(session_path, adl_row, sensor_start_times)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    logger.info(f"Extracted features for {len(features_df)} ADLs")
    
    # Step 4: Merge features with RMSSD labels
    logger.info("\n[4/4] Aligning features with labels...")
    
    # Merge on adl_id
    aligned_df = features_df.merge(
        rmssd_labels[['adl_id', 'rmssd', 'ln_rmssd', 'n_beats', 'mean_rr', 'std_rr']],
        on='adl_id',
        how='left'
    )
    
    # Add session identifier
    aligned_df['session_id'] = args.session_id
    
    # Reorder columns: metadata, features, labels
    metadata_cols = ['session_id', 'adl_id', 'adl_name', 'start_time', 'end_time', 
                    'duration_sec', 'borg_rpe']
    feature_cols = [col for col in aligned_df.columns 
                   if col.startswith(('ppg_', 'acc_', 'eda_', 'gyro_'))]
    label_cols = ['rmssd', 'ln_rmssd', 'n_beats', 'mean_rr', 'std_rr']
    
    final_cols = metadata_cols + feature_cols + label_cols
    aligned_df = aligned_df[[col for col in final_cols if col in aligned_df.columns]]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=False)
    
    logger.info(f"\nSaved aligned dataset to {output_path}")
    logger.info(f"Total ADLs: {len(aligned_df)}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Complete cases (no missing RMSSD): {aligned_df['rmssd'].notna().sum()}")
    
    # Summary statistics
    logger.info("\nRMSSD Summary:")
    logger.info(f"  Mean: {aligned_df['rmssd'].mean():.2f} ms")
    logger.info(f"  Std: {aligned_df['rmssd'].std():.2f} ms")
    logger.info(f"  Range: {aligned_df['rmssd'].min():.2f} - {aligned_df['rmssd'].max():.2f} ms")
    
    logger.info("\nFeature Completeness:")
    logger.info(f"  PPG: {aligned_df['ppg_hr_mean'].notna().sum()}/{len(aligned_df)}")
    logger.info(f"  IMU: {aligned_df['acc_mag_mean'].notna().sum()}/{len(aligned_df)}")
    logger.info(f"  EDA: {aligned_df['eda_mean'].notna().sum()}/{len(aligned_df)}")
    
    logger.info("\n" + "="*60)
    logger.info("ALIGNMENT COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()
