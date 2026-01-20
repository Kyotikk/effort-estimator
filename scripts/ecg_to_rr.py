#!/usr/bin/env python
"""
ECG preprocessing CLI: raw ECG → clean RR intervals.

Usage:
    python scripts/ecg_to_rr.py \\
        --ecg-csv data/raw/ecg_session1.csv \\
        --output-rr data/interim/rr_session1.csv \\
        --output-quality data/interim/rr_quality_session1.json \\
        --sampling-rate 250 \\
        --session-id session_001
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.ecg_preprocess import ECGProcessor, load_ecg_csv, save_rr_intervals, save_quality_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Process raw ECG → clean RR intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/ecg_to_rr.py \\
    --ecg-csv data/raw/ecg.csv \\
    --output-rr data/interim/rr.csv \\
    --sampling-rate 250

  # With session ID and custom quality output
  python scripts/ecg_to_rr.py \\
    --ecg-csv data/raw/ecg_subject1.csv \\
    --output-rr data/interim/rr_subject1.csv \\
    --output-quality data/interim/rr_quality_subject1.json \\
    --session-id subject_001 \\
    --sampling-rate 500 \\
    --verbose
        """
    )
    
    parser.add_argument('--ecg-csv', required=True, help='Input ECG CSV file')
    parser.add_argument('--output-rr', required=True, help='Output RR intervals CSV')
    parser.add_argument('--output-quality', default=None, help='Output quality summary JSON')
    parser.add_argument('--session-id', default=None, help='Session identifier (optional)')
    parser.add_argument('--sampling-rate', type=float, default=250, 
                       help='ECG sampling rate (Hz) [default: 250]')
    parser.add_argument('--time-column', default='time', 
                       help='Name of time column in ECG CSV [default: time]')
    parser.add_argument('--ecg-column', default='ecg', 
                       help='Name of ECG signal column [default: ecg]')
    parser.add_argument('--min-rr', type=float, default=300,
                       help='Minimum physiological RR interval (ms) [default: 300]')
    parser.add_argument('--max-rr', type=float, default=2000,
                       help='Maximum physiological RR interval (ms) [default: 2000]')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output_rr).parent.mkdir(parents=True, exist_ok=True)
    if args.output_quality:
        Path(args.output_quality).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting ECG → RR pipeline")
    logger.info(f"  Input ECG: {args.ecg_csv}")
    logger.info(f"  Sampling rate: {args.sampling_rate} Hz")
    logger.info(f"  Session ID: {args.session_id or 'auto'}")
    
    # Load ECG
    try:
        ecg_df = load_ecg_csv(args.ecg_csv, time_column=args.time_column, 
                             ecg_column=args.ecg_column)
    except Exception as e:
        logger.error(f"Failed to load ECG CSV: {e}")
        return 1
    
    # Process ECG
    processor = ECGProcessor(sampling_rate=args.sampling_rate, verbose=args.verbose)
    try:
        rr_df, quality = processor.process_ecg(
            ecg_df, 
            ecg_column=args.ecg_column,
            time_column=args.time_column,
            session_id=args.session_id
        )
    except Exception as e:
        logger.error(f"ECG processing failed: {e}")
        return 1
    
    # Save outputs
    try:
        save_rr_intervals(rr_df, args.output_rr)
        logger.info(f"✓ RR intervals saved: {args.output_rr}")
    except Exception as e:
        logger.error(f"Failed to save RR intervals: {e}")
        return 1
    
    if args.output_quality:
        try:
            save_quality_summary(quality, args.output_quality)
            logger.info(f"✓ Quality summary saved: {args.output_quality}")
        except Exception as e:
            logger.error(f"Failed to save quality summary: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ECG PROCESSING SUMMARY")
    logger.info("="*70)
    logger.info(f"Input ECG samples: {len(ecg_df)}")
    logger.info(f"R-peaks detected: {len(rr_df) + 1}")  # +1 because df has intervals, not peaks
    logger.info(f"RR intervals extracted: {len(rr_df)}")
    logger.info(f"Valid intervals kept: {quality['n_valid']}/{quality['n_input']} ({quality['pct_removed']:.1f}% removed)")
    logger.info(f"Mean RR (valid): {quality['mean_rr_kept']:.1f} ± {quality['std_rr_kept']:.1f} ms")
    logger.info(f"RR range (valid): {quality['min_rr_kept']:.0f} - {quality['max_rr_kept']:.0f} ms")
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
