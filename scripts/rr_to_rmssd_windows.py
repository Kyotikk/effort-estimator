#!/usr/bin/env python
"""
RR intervals → windowed RMSSD computation CLI.

Usage:
    python scripts/rr_to_rmssd_windows.py \\
        --rr-csv data/interim/rr_session1.csv \\
        --output-windows data/interim/rmssd_windows_session1.csv \\
        --window-length 60 \\
        --overlap 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.ecg_hrv_features import RMSSDWindower, load_rr_intervals, save_rmssd_windows

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compute windowed RMSSD from RR intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 60s windows, 50% overlap
  python scripts/rr_to_rmssd_windows.py \\
    --rr-csv data/interim/rr.csv \\
    --output-windows data/interim/rmssd_windows.csv \\
    --window-length 60 \\
    --overlap 0.5

  # 5-minute windows for longer sessions
  python scripts/rr_to_rmssd_windows.py \\
    --rr-csv data/interim/rr.csv \\
    --output-windows data/interim/rmssd_windows_5min.csv \\
    --window-length 300 \\
    --overlap 0.5 \\
    --min-rr-per-window 5 \\
    --verbose
        """
    )
    
    parser.add_argument('--rr-csv', required=True, help='Input RR intervals CSV')
    parser.add_argument('--output-windows', required=True, help='Output RMSSD windows CSV')
    parser.add_argument('--window-length', type=float, default=60,
                       help='Window duration (seconds) [default: 60]')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap (0.0-1.0) [default: 0.5]')
    parser.add_argument('--min-rr-per-window', type=int, default=3,
                       help='Minimum valid RR intervals per window [default: 3]')
    parser.add_argument('--session-id', default=None, 
                       help='Override session_id from RR CSV (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output_windows).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting RR → RMSSD windowing pipeline")
    logger.info(f"  Input RR: {args.rr_csv}")
    logger.info(f"  Window: {args.window_length}s, overlap: {args.overlap*100:.0f}%")
    
    # Load RR intervals
    try:
        rr_df = load_rr_intervals(args.rr_csv)
    except Exception as e:
        logger.error(f"Failed to load RR CSV: {e}")
        return 1
    
    if len(rr_df) == 0:
        logger.error("RR DataFrame is empty")
        return 1
    
    # Create windower
    windower = RMSSDWindower(
        window_length_s=args.window_length,
        overlap_frac=args.overlap,
        min_rr_per_window=args.min_rr_per_window,
        verbose=args.verbose
    )
    
    # Compute windowed RMSSD
    try:
        df_windows = windower.window_rmssd(rr_df, session_id=args.session_id)
    except Exception as e:
        logger.error(f"Windowing failed: {e}")
        return 1
    
    if len(df_windows) == 0:
        logger.error("No windows produced")
        return 1
    
    # Save output
    try:
        save_rmssd_windows(df_windows, args.output_windows)
        logger.info(f"✓ RMSSD windows saved: {args.output_windows}")
    except Exception as e:
        logger.error(f"Failed to save RMSSD windows: {e}")
        return 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("RMSSD WINDOWING SUMMARY")
    logger.info("="*70)
    logger.info(f"Input RR intervals: {len(rr_df)}")
    logger.info(f"  Valid: {rr_df['is_valid'].sum()}")
    logger.info(f"  Invalid: {(~rr_df['is_valid']).sum()}")
    logger.info(f"Windows created: {len(df_windows)}")
    logger.info(f"Windows with valid RMSSD: {df_windows['rmssd'].notna().sum()}")
    
    # Statistics on RMSSD
    valid_rmssd = df_windows['rmssd'].dropna()
    if len(valid_rmssd) > 0:
        logger.info(f"RMSSD statistics (ms):")
        logger.info(f"  Mean: {valid_rmssd.mean():.1f}")
        logger.info(f"  Median: {valid_rmssd.median():.1f}")
        logger.info(f"  Std: {valid_rmssd.std():.1f}")
        logger.info(f"  Range: {valid_rmssd.min():.1f} - {valid_rmssd.max():.1f}")
        
        valid_ln_rmssd = df_windows['ln_rmssd'].dropna()
        logger.info(f"lnRMSSD statistics:")
        logger.info(f"  Mean: {valid_ln_rmssd.mean():.3f}")
        logger.info(f"  Std: {valid_ln_rmssd.std():.3f}")
    
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
