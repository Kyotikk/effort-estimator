"""
Clean Pipeline Orchestrator
============================

Runs the 7-phase effort estimation pipeline with callable functions.

Phases:
1. Preprocessing - Load & clean raw signals (7 modalities)
2. Windowing - Create sliding windows + QC
3. Features - Extract features (IMU, PPG, RR, EDA)
4. Fusion - Combine all modality features
5. Alignment - Add effort labels
6. Selection - Select top 50 features
7. Training - Train XGBoost model

Usage:
    python pipeline/run_clean_pipeline.py --config config/pipeline.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
import pandas as pd
import logging

# Import all 7 phases
from pipeline.phase1_preprocessing import (
    preprocess_imu, preprocess_ppg, preprocess_eda, preprocess_rr
)
from pipeline.phase2_windowing import create_windows, quality_check_windows
from pipeline.phase3_features import (
    extract_imu_features, extract_ppg_features, extract_rr_features, extract_eda_features
)
from pipeline.phase4_fusion import fuse_modalities
from pipeline.phase5_alignment import align_with_targets
from pipeline.phase6_selection import select_features
from pipeline.phase7_training import train_model, evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_single_subject(config: dict, subject: str) -> dict:
    """
    Process a single subject through all 7 phases.
    
    Args:
        config: Pipeline configuration dict
        subject: Subject name/ID
    
    Returns:
        Dictionary with results from each phase
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing subject: {subject}")
    logger.info(f"{'='*60}")
    
    results = {}
    
    # =========================================================================
    # PHASE 1: PREPROCESSING
    # =========================================================================
    logger.info("\n[PHASE 1] Preprocessing - Load & clean raw signals")
    
    dataset = config["datasets"][subject]
    fs_config = config["preprocessing"]
    
    # Preprocess each modality
    imu_bioz_df = preprocess_imu(dataset["imu_bioz"], fs=fs_config["imu_bioz"]["fs"])
    imu_wrist_df = preprocess_imu(dataset["imu_wrist"], fs=fs_config["imu_wrist"]["fs"])
    ppg_green_df = preprocess_ppg(dataset["ppg_green"], fs=fs_config["ppg_green"]["fs"])
    ppg_infra_df = preprocess_ppg(dataset["ppg_infra"], fs=fs_config["ppg_infra"]["fs"])
    ppg_red_df = preprocess_ppg(dataset["ppg_red"], fs=fs_config["ppg_red"]["fs"])
    eda_df = preprocess_eda(dataset["eda"], fs=fs_config["eda"]["fs"])
    rr_df = preprocess_rr(dataset["rr"], fs=fs_config["rr"]["fs"])
    
    results["preprocessed"] = {
        "imu_bioz": imu_bioz_df,
        "imu_wrist": imu_wrist_df,
        "ppg_green": ppg_green_df,
        "ppg_infra": ppg_infra_df,
        "ppg_red": ppg_red_df,
        "eda": eda_df,
        "rr": rr_df,
    }
    
    logger.info(f"✓ Preprocessed 7 modalities")
    
    # =========================================================================
    # PHASE 2: WINDOWING
    # =========================================================================
    logger.info("\n[PHASE 2] Windowing - Create sliding windows")
    
    win_config = config["windowing"]
    window_length = win_config["window_lengths_sec"][0]  # Use first window size
    overlap = win_config["overlap"]
    
    windows_dict = {}
    for modality_name, df in results["preprocessed"].items():
        fs = config["preprocessing"][modality_name]["fs"]
        windows_dict[modality_name] = create_windows(
            df,
            fs=fs,
            win_sec=window_length,
            overlap=overlap
        )
    
    results["windows"] = windows_dict
    logger.info(f"✓ Created windows for all modalities")
    
    # =========================================================================
    # PHASE 3: FEATURE EXTRACTION
    # =========================================================================
    logger.info("\n[PHASE 3] Feature Extraction - Extract from each modality")
    
    features_dict = {}
    
    # IMU features
    imu_bioz_feat = extract_imu_features(imu_bioz_df, windows_dict["imu_bioz"])
    imu_wrist_feat = extract_imu_features(imu_wrist_df, windows_dict["imu_wrist"])
    
    features_dict["imu_bioz"] = imu_bioz_feat
    features_dict["imu_wrist"] = imu_wrist_feat
    
    # PPG features
    ppg_green_feat = extract_ppg_features(ppg_green_df, windows_dict["ppg_green"])
    ppg_infra_feat = extract_ppg_features(ppg_infra_df, windows_dict["ppg_infra"])
    ppg_red_feat = extract_ppg_features(ppg_red_df, windows_dict["ppg_red"])
    
    features_dict["ppg_green"] = ppg_green_feat
    features_dict["ppg_infra"] = ppg_infra_feat
    features_dict["ppg_red"] = ppg_red_feat
    
    # RR & EDA features
    rr_feat = extract_rr_features(rr_df, windows_dict["rr"])
    eda_feat = extract_eda_features(eda_df, windows_dict["eda"])
    
    features_dict["rr"] = rr_feat
    features_dict["eda"] = eda_feat
    
    results["features"] = features_dict
    logger.info(f"✓ Extracted features from all modalities")
    
    # =========================================================================
    # PHASE 4: FUSION
    # =========================================================================
    logger.info("\n[PHASE 4] Fusion - Combine all modality features")
    
    fused_df = fuse_modalities(features_dict, on="t_start", method="inner")
    results["fused"] = fused_df
    logger.info(f"✓ Fused data: {fused_df.shape[0]} windows x {fused_df.shape[1]} features")
    
    # =========================================================================
    # PHASE 5: ALIGNMENT
    # =========================================================================
    logger.info("\n[PHASE 5] Alignment - Add effort labels")
    
    # Load target labels
    targets_df = pd.read_csv(dataset["adl"])
    aligned_df = align_with_targets(fused_df, targets_df, time_col="t_start")
    results["aligned"] = aligned_df
    logger.info(f"✓ Aligned {aligned_df['effort'].notna().sum()}/{len(aligned_df)} windows with labels")
    
    # =========================================================================
    # PHASE 6: FEATURE SELECTION
    # =========================================================================
    logger.info("\n[PHASE 6] Feature Selection - Select top 50 features")
    
    # Separate features from metadata
    feature_cols = [c for c in aligned_df.columns 
                   if c not in ["t_start", "t_center", "t_end", "effort", "modality_source"]]
    X = aligned_df[feature_cols].fillna(0)
    
    selection_result = select_features(X, n_features=50)
    results["selection"] = selection_result
    logger.info(f"✓ Selected {len(selection_result['selected_features'])} top features")
    
    # =========================================================================
    # PHASE 7: TRAINING
    # =========================================================================
    logger.info("\n[PHASE 7] Training - Train XGBoost model")
    
    X_selected = selection_result["X_selected"]
    y = aligned_df["effort"].dropna()
    
    train_result = train_model(X_selected, y, test_size=0.2)
    results["model"] = train_result
    
    metrics = evaluate_model(train_result)
    results["metrics"] = metrics
    logger.info(f"✓ Model trained - R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject} - COMPLETE")
    logger.info(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Clean modular effort estimation pipeline")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml", help="Config file path")
    parser.add_argument("--subject", type=str, default=None, help="Process single subject (default: all)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Process subject(s)
    subjects = [args.subject] if args.subject else list(config["datasets"].keys())
    
    all_results = {}
    for subject in subjects:
        try:
            results = process_single_subject(config, subject)
            all_results[subject] = results
        except Exception as e:
            logger.error(f"Error processing {subject}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    
    # Print summary
    for subject, results in all_results.items():
        if "metrics" in results:
            metrics = results["metrics"]
            logger.info(f"{subject}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
