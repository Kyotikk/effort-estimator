"""
HRV Recovery Estimation Pipeline - Main Entrypoint

Orchestrates all 6 modules to build and train a model predicting HRV recovery from physiological features.
"""

import logging
import argparse
import yaml
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from hrv_recovery_pipeline.module1_ibi import extract_ibi_timeseries
from hrv_recovery_pipeline.module2_rmssd import compute_rmssd_windows, get_rmssd_in_interval
from hrv_recovery_pipeline.module3_bouts import get_effort_bouts
from hrv_recovery_pipeline.module4_labels import compute_bout_labels
from hrv_recovery_pipeline.module5_features import build_model_table
from hrv_recovery_pipeline.module6_training import evaluate_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_hrv_pipeline(config_path: Path, output_dir: Optional[Path] = None):
    """
    Execute HRV recovery estimation pipeline.
    
    Args:
        config_path: Path to pipeline config YAML
        output_dir: Output directory (default: from config)
    """
    
    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    if output_dir is None:
        output_dir = Path(config.get('output_dir', './hrv_output'))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # =========================================================================
    # MODULE 1: Extract IBI from green PPG
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 1: Extract IBI from green PPG")
    logger.info("="*80)
    
    ppg_path = config['input_paths'].get('ppg_green')
    if not ppg_path:
        raise ValueError("ppg_green path not in config")
    
    ppg_df = pd.read_csv(ppg_path)
    logger.info(f"Loaded PPG: {len(ppg_df)} samples")
    
    ibi_params = config.get('ibi', {})
    ibi_df = extract_ibi_timeseries(ppg_df, **ibi_params)
    
    ibi_output = output_dir / "ibi_timeseries.csv"
    ibi_df.to_csv(ibi_output, index=False)
    logger.info(f"Saved IBI to {ibi_output}")
    
    # =========================================================================
    # MODULE 2: Compute windowed RMSSD
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 2: Compute windowed RMSSD")
    logger.info("="*80)
    
    rmssd_params = config.get('rmssd', {})
    rmssd_df = compute_rmssd_windows(ibi_df, **rmssd_params)
    
    rmssd_output = output_dir / "rmssd_windows.csv"
    rmssd_df.to_csv(rmssd_output, index=False)
    logger.info(f"Saved RMSSD windows to {rmssd_output}")
    
    # =========================================================================
    # MODULE 3: Define effort bouts
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 3: Define effort bouts")
    logger.info("="*80)
    
    adl_path = config['input_paths'].get('adl')
    imu_path = config['input_paths'].get('imu_features')
    
    bouts_df = get_effort_bouts(
        adl_path=adl_path,
        imu_features_df=pd.read_csv(imu_path) if imu_path else None,
        **config.get('bouts', {})
    )
    
    if bouts_df.empty:
        raise ValueError("No effort bouts detected")
    
    bouts_output = output_dir / "effort_bouts.csv"
    bouts_df.to_csv(bouts_output, index=False)
    logger.info(f"Saved bouts to {bouts_output}")
    
    # =========================================================================
    # MODULE 4: Compute HRV recovery labels
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 4: Compute HRV recovery labels")
    logger.info("="*80)
    
    labels_df = compute_bout_labels(
        rmssd_df,
        bouts_df,
        **config.get('labels', {})
    )
    
    labels_output = output_dir / "bout_labels.csv"
    labels_df.to_csv(labels_output, index=False)
    logger.info(f"Saved labels to {labels_output}")
    
    # =========================================================================
    # MODULE 5: Extract features and build model table
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 5: Extract features and build model table")
    logger.info("="*80)
    
    # Load additional feature sources
    imu_features_df = None
    eda_features_df = None
    
    if imu_path and Path(imu_path).exists():
        imu_features_df = pd.read_csv(imu_path)
        logger.info(f"Loaded IMU features: {len(imu_features_df)} windows")
    
    eda_path = config['input_paths'].get('eda_features')
    if eda_path and Path(eda_path).exists():
        eda_features_df = pd.read_csv(eda_path)
        logger.info(f"Loaded EDA features: {len(eda_features_df)} windows")
    
    model_table = build_model_table(
        bouts_df,
        labels_df,
        imu_features_df=imu_features_df,
        eda_features_df=eda_features_df,
        ibi_df=ibi_df,
        session_id=config.get('session_id', 'unknown')
    )
    
    model_table_output = output_dir / "model_table.csv"
    model_table.to_csv(model_table_output, index=False)
    logger.info(f"Saved model table to {model_table_output}")
    
    # =========================================================================
    # MODULE 6: Train and evaluate models
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODULE 6: Train and evaluate models")
    logger.info("="*80)
    
    target_col = config.get('target_col', 'delta_rmssd')
    results = evaluate_models(
        model_table,
        target_col=target_col,
        output_dir=output_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Output files in: {output_dir}")
    logger.info(f"  ibi_timeseries.csv (n={len(ibi_df)})")
    logger.info(f"  rmssd_windows.csv (n={len(rmssd_df)})")
    logger.info(f"  effort_bouts.csv (n={len(bouts_df)})")
    logger.info(f"  bout_labels.csv (n={len(labels_df)}, {labels_df['qc_ok'].sum()} QC pass)")
    logger.info(f"  model_table.csv (n={len(model_table)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="HRV Recovery Estimation Pipeline"
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to pipeline config YAML'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    run_hrv_pipeline(args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
