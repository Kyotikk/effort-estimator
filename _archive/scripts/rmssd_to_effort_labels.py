#!/usr/bin/env python
"""
RMSSD windows → Effort labels CLI.

Usage:
    python scripts/rmssd_to_effort_labels.py \\
        --rmssd-csv data/interim/rmssd_windows.csv \\
        --output-labels data/interim/effort_labels.csv \\
        --strategy rmssd_recovery \\
        --baseline-windows 3 \\
        --threshold -20
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.effort_labels import EffortLabelGenerator, load_rmssd_windows, save_effort_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate effort labels from RMSSD windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: use RMSSD recovery strategy with defaults
  python scripts/rmssd_to_effort_labels.py \\
    --rmssd-csv data/interim/rmssd_windows.csv \\
    --output-labels data/interim/effort_labels.csv

  # Custom baseline and threshold
  python scripts/rmssd_to_effort_labels.py \\
    --rmssd-csv data/interim/rmssd_windows.csv \\
    --output-labels data/interim/effort_labels.csv \\
    --strategy rmssd_recovery \\
    --baseline-windows 5 \\
    --threshold -25 \\
    --verbose

  # Z-score based strategy
  python scripts/rmssd_to_effort_labels.py \\
    --rmssd-csv data/interim/rmssd_windows.csv \\
    --output-labels data/interim/effort_labels.csv \\
    --strategy rmssd_zscore
        """
    )
    
    parser.add_argument('--rmssd-csv', required=True, 
                       help='Input RMSSD windows CSV (from Stage 2)')
    parser.add_argument('--output-labels', required=True, 
                       help='Output effort labels CSV')
    parser.add_argument('--strategy', default='rmssd_recovery',
                       choices=['rmssd_recovery', 'rmssd_zscore'],
                       help='Label generation strategy [default: rmssd_recovery]')
    parser.add_argument('--baseline-windows', type=int, default=3,
                       help='Number of initial windows for baseline [default: 3]')
    parser.add_argument('--threshold', type=float, default=-20,
                       help='RMSSD drop threshold (%) for effort label [default: -20]')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output_labels).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting RMSSD → Effort Labels pipeline")
    logger.info(f"  Input RMSSD: {args.rmssd_csv}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Baseline windows: {args.baseline_windows}")
    logger.info(f"  Threshold: {args.threshold}%")
    
    # Load RMSSD windows
    try:
        df_rmssd = load_rmssd_windows(args.rmssd_csv)
    except Exception as e:
        logger.error(f"Failed to load RMSSD CSV: {e}")
        return 1
    
    if len(df_rmssd) == 0:
        logger.error("RMSSD DataFrame is empty")
        return 1
    
    # Create label generator
    generator = EffortLabelGenerator(
        baseline_window_count=args.baseline_windows,
        recovery_threshold_pct=args.threshold,
        verbose=args.verbose
    )
    
    # Generate labels
    try:
        df_labels = generator.generate_labels(df_rmssd, strategy=args.strategy)
    except Exception as e:
        logger.error(f"Label generation failed: {e}")
        return 1
    
    # Save output
    try:
        save_effort_labels(df_labels, args.output_labels)
        logger.info(f"✓ Effort labels saved: {args.output_labels}")
    except Exception as e:
        logger.error(f"Failed to save effort labels: {e}")
        return 1
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("EFFORT LABEL SUMMARY")
    logger.info("="*70)
    logger.info(f"Total windows: {len(df_labels)}")
    
    effort_count = (df_labels['effort_binary'] == 1).sum()
    rest_count = (df_labels['effort_binary'] == 0).sum()
    logger.info(f"  Effort windows: {effort_count} ({100*effort_count/len(df_labels):.1f}%)")
    logger.info(f"  Rest windows: {rest_count} ({100*rest_count/len(df_labels):.1f}%)")
    
    # Baseline statistics
    if not pd.isna(generator.baseline_rmssd_mean):
        logger.info(f"\nBaseline (rest state):")
        logger.info(f"  RMSSD: {generator.baseline_rmssd_mean:.1f} ± {generator.baseline_rmssd_std:.1f} ms")
        logger.info(f"  N windows: {args.baseline_windows}")
    
    # Effort window statistics
    effort_windows = df_labels[df_labels['effort_binary'] == 1]
    if len(effort_windows) > 0:
        logger.info(f"\nEffort windows (RMSSD drops {args.threshold}%):")
        logger.info(f"  RMSSD: {effort_windows['rmssd'].mean():.1f} ± {effort_windows['rmssd'].std():.1f} ms")
        logger.info(f"  Confidence: {effort_windows['effort_confidence'].mean():.2f} ± {effort_windows['effort_confidence'].std():.2f}")
        logger.info(f"  % change: {effort_windows['rmssd_pct_change'].mean():.1f}% ± {effort_windows['rmssd_pct_change'].std():.1f}%")
    
    # Label distribution over time
    window_ids = df_labels['window_id'].values
    effort_binary = df_labels['effort_binary'].values
    
    # Find transitions (rest → effort, effort → rest)
    transitions = np.diff(effort_binary)
    n_rest_to_effort = (transitions == 1).sum()
    n_effort_to_rest = (transitions == -1).sum()
    
    logger.info(f"\nLabel transitions:")
    logger.info(f"  Rest → Effort: {n_rest_to_effort} times")
    logger.info(f"  Effort → Rest: {n_effort_to_rest} times")
    
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())
