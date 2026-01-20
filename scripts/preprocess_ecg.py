#!/usr/bin/env python3
"""
CLI for ECG preprocessing and RMSSD label generation.

Usage:
    python scripts/preprocess_ecg.py \\
        --ecg-file /path/to/vivalnk_vv330_ecg/data_1.csv.gz \\
        --output /path/to/output_labels.csv \\
        --session-id sim_elderly3_2025-12-04

Optional:
    --adl-file /path/to/scai_app/ADLs_1.csv \\
        (if provided, generates per-ADL labels; otherwise generates windowed RMSSD)
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_adl_segments(adl_file: str) -> pd.DataFrame:
    """
    Load ADL segments from scai_app/ADLs_1.csv.
    
    Expected columns (adapt based on actual format):
    - activity_name or adl_name
    - start_time or timestamp_start
    - end_time or timestamp_end
    - (optional) phase: baseline, task, recovery
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading ADL segments from {adl_file}")
    
    df = pd.read_csv(adl_file)
    logger.info(f"Loaded {len(df)} ADL records with columns: {df.columns.tolist()}")
    
    # Standardize column names (adapt based on actual format)
    # This is a placeholder - adjust based on actual ADLs_1.csv structure
    required_cols = []
    
    # Try to identify timestamp columns
    if 'start_time' not in df.columns:
        for col in ['timestamp_start', 'time_start', 'start']:
            if col in df.columns:
                df['start_time'] = pd.to_datetime(df[col])
                break
    
    if 'end_time' not in df.columns:
        for col in ['timestamp_end', 'time_end', 'end']:
            if col in df.columns:
                df['end_time'] = pd.to_datetime(df[col])
                break
    
    # Convert to seconds from first timestamp
    if 'start_time' in df.columns and 'end_time' in df.columns:
        first_time = df['start_time'].min()
        df['start_time'] = (df['start_time'] - first_time).dt.total_seconds()
        df['end_time'] = (df['end_time'] - first_time).dt.total_seconds()
    
    # Add ADL ID if not present
    if 'adl_id' not in df.columns:
        df['adl_id'] = range(len(df))
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='ECG preprocessing and RMSSD label generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--ecg-file', required=True, type=str,
                       help='Path to ECG file (data_1.csv.gz)')
    parser.add_argument('--output', required=True, type=str,
                       help='Output CSV file for labels')
    parser.add_argument('--session-id', required=True, type=str,
                       help='Session identifier (e.g., sim_elderly3_2025-12-04)')
    parser.add_argument('--adl-file', type=str, default=None,
                       help='Optional: ADL segments file (ADLs_1.csv)')
    parser.add_argument('--sampling-rate', type=float, default=256.0,
                       help='ECG sampling rate (Hz, default: 256)')
    parser.add_argument('--window-size', type=float, default=60.0,
                       help='RMSSD window size (seconds, default: 60)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    ecg_path = Path(args.ecg_file)
    if not ecg_path.exists():
        logger.error(f"ECG file not found: {ecg_path}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("ECG PREPROCESSING AND RMSSD LABEL GENERATION")
    logger.info("="*60)
    logger.info(f"Session ID: {args.session_id}")
    logger.info(f"ECG file: {ecg_path}")
    logger.info(f"Output: {args.output}")
    
    # Step 1: Load ECG data
    logger.info("\n[1/4] Loading ECG data...")
    ecg_signal, detected_sr = load_vivalnk_ecg(str(ecg_path))
    sampling_rate = args.sampling_rate if detected_sr == 256.0 else detected_sr
    logger.info(f"Loaded {len(ecg_signal)} samples at {sampling_rate:.2f} Hz")
    logger.info(f"Duration: {len(ecg_signal) / sampling_rate / 60:.2f} minutes")
    
    # Step 2: Process ECG
    logger.info("\n[2/4] Processing ECG (R-peak detection, RR intervals)...")
    preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)
    r_peaks, clean_rr, valid_mask = preprocessor.process_ecg(ecg_signal)
    
    logger.info(f"Detected {len(r_peaks)} R-peaks")
    logger.info(f"Clean RR intervals: {len(clean_rr)}")
    logger.info(f"Mean RR: {np.mean(clean_rr):.2f} ms (HR: {60000 / np.mean(clean_rr):.1f} bpm)")
    
    # Step 3: Compute RR times
    rr_times = (r_peaks[:-1] / sampling_rate)[valid_mask]  # Times of RR intervals
    
    # Step 4: Compute RMSSD labels
    logger.info("\n[3/4] Computing RMSSD-based effort labels...")
    labeler = RMSSDLabeler(window_size_sec=args.window_size)
    
    if args.adl_file:
        # Per-ADL labels
        adl_segments = load_adl_segments(args.adl_file)
        labels_df = labeler.create_session_labels(
            rr_intervals=clean_rr,
            rr_times=rr_times,
            adl_segments=adl_segments,
            session_id=args.session_id
        )
    else:
        # Windowed RMSSD
        labels_df = labeler.compute_windowed_rmssd(clean_rr, rr_times)
        labels_df['session_id'] = args.session_id
    
    logger.info(f"Generated {len(labels_df)} label records")
    
    # Step 5: Save output
    logger.info("\n[4/4] Saving labels...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Summary statistics
    if 'rmssd' in labels_df.columns:
        logger.info("\nRMSSD Summary:")
        logger.info(f"  Mean: {labels_df['rmssd'].mean():.2f} ms")
        logger.info(f"  Std: {labels_df['rmssd'].std():.2f} ms")
        logger.info(f"  Min: {labels_df['rmssd'].min():.2f} ms")
        logger.info(f"  Max: {labels_df['rmssd'].max():.2f} ms")
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()
