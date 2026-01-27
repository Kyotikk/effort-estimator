#!/usr/bin/env python3
"""
Effort Prediction Script - Predict Borg effort from sensor data using trained model.

This script takes raw sensor data from a NEW patient (no Borg labels needed)
and predicts effort scores using the pre-trained XGBoost model.

Usage:
    python predict_effort.py --data_dir /path/to/patient/data

The data directory should contain Corsano sensor files:
    - corsano_wrist_ppg2_green*.csv
    - corsano_bioz_emography*.csv  
    - corsano_bioz_acc*.csv
    - corsano_wrist_acc*.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

# Import preprocessing and feature extraction modules
from preprocessing.ppg import preprocess_ppg
from preprocessing.eda import preprocess_eda
from preprocessing.imu import preprocess_imu
from windowing.windows import create_windows
from features.manual_features_imu import compute_top_imu_features_from_windows
from features.hrv_features import extract_hrv_features
from features.eda_advanced_features import extract_eda_advanced_features
from features.ppg_features import extract_ppg_features
from features.eda_features import extract_eda_features


# Default paths
DEFAULT_MODEL_PATH = "/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models/xgboost_multisub_10.0s.json"
DEFAULT_FEATURES_PATH = "/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/qc_10.0s/features_selected_pruned.csv"


def find_file(data_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Find a file matching any of the patterns in the directory."""
    for pattern in patterns:
        matches = list(data_dir.glob(f"*{pattern}*"))
        if matches:
            # Prefer uncompressed over compressed
            uncompressed = [m for m in matches if not str(m).endswith('.gz')]
            if uncompressed:
                return sorted(uncompressed, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


def extract_all_features(
    ppg_path: Path,
    eda_path: Path,
    imu_bioz_path: Path,
    imu_wrist_path: Path,
    output_dir: Path,
    window_sec: float = 10.0,
    overlap: float = 0.7,
    fs_ppg: float = 32.0,
    fs_eda: float = 32.0,
    fs_imu: float = 32.0,
) -> pd.DataFrame:
    """
    Extract all features from raw sensor data.
    
    Returns a DataFrame with one row per window, containing all features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print(f"{'='*60}")
    
    # 1. Preprocess PPG
    print("▶ Preprocessing PPG...")
    ppg_clean_path = output_dir / "ppg_preprocessed.csv"
    preprocess_ppg(
        in_path=str(ppg_path),
        out_path=str(ppg_clean_path),
        fs=fs_ppg,
        time_col="time",
        do_resample=True,
    )
    ppg_df = pd.read_csv(ppg_clean_path)
    print(f"  ✓ PPG: {len(ppg_df)} samples")
    
    # 2. Preprocess EDA
    print("▶ Preprocessing EDA...")
    eda_clean_path = output_dir / "eda_preprocessed.csv"
    preprocess_eda(
        in_path=str(eda_path),
        out_path=str(eda_clean_path),
        fs=fs_eda,
        time_col="time",
        do_resample=True,
    )
    eda_df = pd.read_csv(eda_clean_path)
    print(f"  ✓ EDA: {len(eda_df)} samples")
    
    # 3. Preprocess IMU (bioz)
    print("▶ Preprocessing IMU (bioz)...")
    imu_bioz_df = preprocess_imu(
        path=str(imu_bioz_path),
        fs_out=fs_imu,
        noise_cutoff=5.0,
        gravity_cutoff=0.3,
    )
    imu_bioz_clean_path = output_dir / "imu_bioz_preprocessed.csv"
    imu_bioz_df.to_csv(imu_bioz_clean_path, index=False)
    print(f"  ✓ IMU bioz: {len(imu_bioz_df)} samples")
    
    # 4. Preprocess IMU (wrist)
    print("▶ Preprocessing IMU (wrist)...")
    imu_wrist_df = preprocess_imu(
        path=str(imu_wrist_path),
        fs_out=fs_imu,
        noise_cutoff=5.0,
        gravity_cutoff=0.3,
    )
    imu_wrist_clean_path = output_dir / "imu_wrist_preprocessed.csv"
    imu_wrist_df.to_csv(imu_wrist_clean_path, index=False)
    print(f"  ✓ IMU wrist: {len(imu_wrist_df)} samples")
    
    print(f"\n{'='*60}")
    print("WINDOWING")
    print(f"{'='*60}")
    
    # Create windows for each modality
    ppg_windows = create_windows(ppg_df, fs=fs_ppg, win_sec=window_sec, overlap=overlap)
    eda_windows = create_windows(eda_df, fs=fs_eda, win_sec=window_sec, overlap=overlap)
    imu_bioz_windows = create_windows(imu_bioz_df, fs=fs_imu, win_sec=window_sec, overlap=overlap)
    imu_wrist_windows = create_windows(imu_wrist_df, fs=fs_imu, win_sec=window_sec, overlap=overlap)
    
    # Save windows
    ppg_windows.to_csv(output_dir / "ppg_windows.csv", index=False)
    eda_windows.to_csv(output_dir / "eda_windows.csv", index=False)
    imu_bioz_windows.to_csv(output_dir / "imu_bioz_windows.csv", index=False)
    imu_wrist_windows.to_csv(output_dir / "imu_wrist_windows.csv", index=False)
    
    print(f"  ✓ PPG windows: {len(ppg_windows)}")
    print(f"  ✓ EDA windows: {len(eda_windows)}")
    print(f"  ✓ IMU bioz windows: {len(imu_bioz_windows)}")
    print(f"  ✓ IMU wrist windows: {len(imu_wrist_windows)}")
    
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION")
    print(f"{'='*60}")
    
    # Extract PPG statistical features
    print("▶ Extracting PPG features...")
    ppg_feat_path = output_dir / "ppg_features.csv"
    extract_ppg_features(
        ppg_csv=str(ppg_clean_path),
        windows_csv=str(output_dir / "ppg_windows.csv"),
        time_col="t_sec",
        signal_col="value",
        prefix="ppg_green_",
    ).to_csv(ppg_feat_path, index=False)
    
    # Extract HRV features from PPG
    print("▶ Extracting HRV features...")
    hrv_feat_path = output_dir / "hrv_features.csv"
    extract_hrv_features(
        ppg_csv=str(ppg_clean_path),
        windows_csv=str(output_dir / "ppg_windows.csv"),
        out_path=str(hrv_feat_path),
        fs=fs_ppg,
        prefix="ppg_green_",
    )
    
    # Extract EDA basic features
    print("▶ Extracting EDA features...")
    eda_feat_path = output_dir / "eda_features.csv"
    extract_eda_features(
        eda_csv=str(eda_clean_path),
        windows_csv=str(output_dir / "eda_windows.csv"),
        time_col="t_sec",
        cc_col="eda_cc",
        stress_col="eda_stress_skin",
        prefix="eda_",
    ).to_csv(eda_feat_path, index=False)
    
    # Extract advanced EDA features
    print("▶ Extracting advanced EDA features...")
    eda_adv_feat_path = output_dir / "eda_advanced_features.csv"
    extract_eda_advanced_features(
        eda_csv=str(eda_clean_path),
        windows_csv=str(output_dir / "eda_windows.csv"),
        out_path=str(eda_adv_feat_path),
        fs=fs_eda,
        prefix="eda_",
    )
    
    # Extract IMU features
    print("▶ Extracting IMU bioz features...")
    imu_bioz_feats = compute_top_imu_features_from_windows(
        data=imu_bioz_df,
        windows=imu_bioz_windows,
        signal_cols=["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"],
    )
    # Rename IMU features to include bioz prefix
    imu_bioz_feats = imu_bioz_feats.rename(columns={
        c: f"imu_bioz_{c}" if not c.startswith(("window_id", "start_idx", "end_idx", "t_")) else c
        for c in imu_bioz_feats.columns
    })
    imu_bioz_feats.to_csv(output_dir / "imu_bioz_features.csv", index=False)
    
    print("▶ Extracting IMU wrist features...")
    imu_wrist_feats = compute_top_imu_features_from_windows(
        data=imu_wrist_df,
        windows=imu_wrist_windows,
        signal_cols=["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"],
    )
    # Rename IMU features to include wrist prefix
    imu_wrist_feats = imu_wrist_feats.rename(columns={
        c: f"imu_wrist_{c}" if not c.startswith(("window_id", "start_idx", "end_idx", "t_")) else c
        for c in imu_wrist_feats.columns
    })
    imu_wrist_feats.to_csv(output_dir / "imu_wrist_features.csv", index=False)
    
    print(f"\n{'='*60}")
    print("FUSING FEATURES")
    print(f"{'='*60}")
    
    # Load all feature tables
    ppg_feats = pd.read_csv(ppg_feat_path)
    hrv_feats = pd.read_csv(hrv_feat_path)
    eda_feats = pd.read_csv(eda_feat_path)
    eda_adv_feats = pd.read_csv(eda_adv_feat_path)
    
    # Merge all features on t_center (time-based alignment)
    # Start with PPG as base (has most windows typically)
    fused = ppg_feats.copy()
    
    # Merge HRV features
    hrv_cols = [c for c in hrv_feats.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_end"]]
    fused = pd.merge_asof(
        fused.sort_values("t_center"),
        hrv_feats[hrv_cols + ["t_center"]].sort_values("t_center"),
        on="t_center",
        direction="nearest",
        tolerance=2.0,
    )
    
    # Merge EDA features
    eda_cols = [c for c in eda_feats.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_end"]]
    fused = pd.merge_asof(
        fused.sort_values("t_center"),
        eda_feats[eda_cols + ["t_center"]].sort_values("t_center"),
        on="t_center",
        direction="nearest",
        tolerance=2.0,
        suffixes=("", "_eda"),
    )
    
    # Merge advanced EDA features
    eda_adv_cols = [c for c in eda_adv_feats.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_end"]]
    fused = pd.merge_asof(
        fused.sort_values("t_center"),
        eda_adv_feats[eda_adv_cols + ["t_center"]].sort_values("t_center"),
        on="t_center",
        direction="nearest",
        tolerance=2.0,
        suffixes=("", "_eda_adv"),
    )
    
    # Merge IMU bioz features
    imu_bioz_cols = [c for c in imu_bioz_feats.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_end"]]
    fused = pd.merge_asof(
        fused.sort_values("t_center"),
        imu_bioz_feats[imu_bioz_cols + ["t_center"]].sort_values("t_center"),
        on="t_center",
        direction="nearest",
        tolerance=2.0,
        suffixes=("", "_bioz"),
    )
    
    # Merge IMU wrist features
    imu_wrist_cols = [c for c in imu_wrist_feats.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_end"]]
    fused = pd.merge_asof(
        fused.sort_values("t_center"),
        imu_wrist_feats[imu_wrist_cols + ["t_center"]].sort_values("t_center"),
        on="t_center",
        direction="nearest",
        tolerance=2.0,
        suffixes=("", "_wrist"),
    )
    
    # Save fused features
    fused.to_csv(output_dir / "fused_features.csv", index=False)
    print(f"  ✓ Fused features: {len(fused)} windows, {len(fused.columns)} columns")
    
    return fused


def predict_effort(
    features_df: pd.DataFrame,
    model_path: str,
    selected_features_path: str,
) -> pd.DataFrame:
    """
    Predict effort scores using the trained model.
    
    Args:
        features_df: DataFrame with extracted features
        model_path: Path to trained XGBoost model
        selected_features_path: Path to list of features used by model
    
    Returns:
        DataFrame with predictions added
    """
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")
    
    # Load model
    print(f"▶ Loading model: {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Load selected features
    print(f"▶ Loading feature list: {selected_features_path}")
    selected_features = pd.read_csv(selected_features_path, header=None)[0].tolist()
    print(f"  Model expects {len(selected_features)} features")
    
    # Check which features are available
    available = [f for f in selected_features if f in features_df.columns]
    missing = [f for f in selected_features if f not in features_df.columns]
    
    print(f"  Available: {len(available)}/{len(selected_features)}")
    if missing:
        print(f"  ⚠ Missing features: {missing[:10]}...")  # Show first 10
    
    if len(available) < len(selected_features) * 0.8:
        print(f"\n⚠ WARNING: Only {len(available)}/{len(selected_features)} features available.")
        print("  Predictions may be less accurate.")
    
    # Prepare feature matrix
    X = features_df[available].values
    
    # Handle missing features by filling with 0 (not ideal but works)
    if missing:
        X_full = np.zeros((len(features_df), len(selected_features)))
        for i, feat in enumerate(selected_features):
            if feat in available:
                X_full[:, i] = features_df[feat].values
        X = X_full
    
    # Replace NaN with 0 (or could use mean imputation)
    X = np.nan_to_num(X, nan=0.0)
    
    # Predict
    print("▶ Running prediction...")
    predictions = model.predict(X)
    
    # Clip to valid Borg range [0, 10]
    predictions = np.clip(predictions, 0, 10)
    
    # Add predictions to dataframe
    result = features_df.copy()
    result["predicted_effort"] = predictions
    
    print(f"\n✓ Predictions complete!")
    print(f"  Windows: {len(predictions)}")
    print(f"  Effort range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"  Mean effort: {predictions.mean():.2f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict effort from sensor data using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict from a directory containing sensor files
    python predict_effort.py --data_dir /path/to/patient/data
    
    # Specify individual files
    python predict_effort.py --ppg ppg.csv --eda eda.csv --imu_bioz imu_bioz.csv --imu_wrist imu_wrist.csv
    
    # Use custom model
    python predict_effort.py --data_dir /path/to/data --model /path/to/model.json
        """
    )
    
    parser.add_argument("--data_dir", type=str, help="Directory containing sensor CSV files")
    parser.add_argument("--ppg", type=str, help="Path to PPG CSV file")
    parser.add_argument("--eda", type=str, help="Path to EDA CSV file")
    parser.add_argument("--imu_bioz", type=str, help="Path to IMU bioz CSV file")
    parser.add_argument("--imu_wrist", type=str, help="Path to IMU wrist CSV file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES_PATH, help="Path to selected features list")
    parser.add_argument("--output", type=str, default="./effort_predictions", help="Output directory")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Window length in seconds")
    
    args = parser.parse_args()
    
    # Find sensor files
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"✗ Directory not found: {data_dir}")
            sys.exit(1)
        
        ppg_path = find_file(data_dir, ["ppg2_green", "ppg_green"])
        eda_path = find_file(data_dir, ["emography", "eda"])
        imu_bioz_path = find_file(data_dir, ["bioz_acc"])
        imu_wrist_path = find_file(data_dir, ["wrist_acc"])
        
        if not ppg_path:
            print(f"✗ PPG file not found in {data_dir}")
            sys.exit(1)
        if not eda_path:
            print(f"✗ EDA file not found in {data_dir}")
            sys.exit(1)
        if not imu_bioz_path:
            print(f"✗ IMU bioz file not found in {data_dir}")
            sys.exit(1)
        if not imu_wrist_path:
            print(f"✗ IMU wrist file not found in {data_dir}")
            sys.exit(1)
    else:
        # Use individual file paths
        if not all([args.ppg, args.eda, args.imu_bioz, args.imu_wrist]):
            print("✗ Please provide either --data_dir or all individual file paths")
            sys.exit(1)
        
        ppg_path = Path(args.ppg)
        eda_path = Path(args.eda)
        imu_bioz_path = Path(args.imu_bioz)
        imu_wrist_path = Path(args.imu_wrist)
    
    print(f"\n{'='*60}")
    print("EFFORT PREDICTION PIPELINE")
    print(f"{'='*60}")
    print(f"\nInput files:")
    print(f"  PPG:       {ppg_path}")
    print(f"  EDA:       {eda_path}")
    print(f"  IMU bioz:  {imu_bioz_path}")
    print(f"  IMU wrist: {imu_wrist_path}")
    print(f"\nModel: {args.model}")
    print(f"Window: {args.window_sec}s")
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"\n✗ Model not found: {args.model}")
        print(f"  Run the training pipeline first:")
        print(f"    python run_multisub_pipeline.py")
        print(f"    python train_multisub_xgboost.py")
        sys.exit(1)
    
    # Check features file exists
    if not Path(args.features).exists():
        print(f"\n✗ Features file not found: {args.features}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    features_df = extract_all_features(
        ppg_path=ppg_path,
        eda_path=eda_path,
        imu_bioz_path=imu_bioz_path,
        imu_wrist_path=imu_wrist_path,
        output_dir=output_dir / "intermediate",
        window_sec=args.window_sec,
    )
    
    # Predict effort
    result = predict_effort(
        features_df=features_df,
        model_path=args.model,
        selected_features_path=args.features,
    )
    
    # Save results
    output_path = output_dir / "effort_predictions.csv"
    result.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  Full results: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EFFORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows: {len(result)}")
    print(f"  Duration: {len(result) * args.window_sec * (1 - 0.7):.1f} seconds")
    print(f"\n  Effort Statistics:")
    print(f"    Mean:   {result['predicted_effort'].mean():.2f}")
    print(f"    Std:    {result['predicted_effort'].std():.2f}")
    print(f"    Min:    {result['predicted_effort'].min():.2f}")
    print(f"    Max:    {result['predicted_effort'].max():.2f}")
    print(f"    Median: {result['predicted_effort'].median():.2f}")
    
    # Categorize effort levels
    effort_bins = [0, 2, 4, 6, 8, 10]
    effort_labels = ["Rest (0-2)", "Light (2-4)", "Moderate (4-6)", "Hard (6-8)", "Very Hard (8-10)"]
    result["effort_category"] = pd.cut(result["predicted_effort"], bins=effort_bins, labels=effort_labels)
    
    print(f"\n  Effort Distribution:")
    for cat in effort_labels:
        count = (result["effort_category"] == cat).sum()
        pct = count / len(result) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Done! Predictions saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
