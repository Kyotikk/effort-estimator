#!/usr/bin/env python3
"""
Clean Pipeline Orchestrator - Using EXACT working code from pascal_update
No modifications to preprocessing/windowing/features logic
"""
import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import EXACT working preprocessing from phase1
from pipeline.phase1_preprocessing.imu import preprocess_imu
from pipeline.phase1_preprocessing.ppg import preprocess_ppg  
from pipeline.phase1_preprocessing.eda import preprocess_eda
from pipeline.phase1_preprocessing.rr import preprocess_rr

# Import EXACT working windowing from phase2
from pipeline.phase2_windowing.windows import create_windows

# Feature extraction will come later - just do preprocessing + windowing for now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    parser.add_argument("--subject", help="Specific subject to process")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    datasets = config["datasets"]
    if args.subject:
        datasets = [ds for ds in datasets if ds["name"] == args.subject]
    
    for dataset in datasets:
        subject = dataset["name"]
        print(f"\n{'='*60}")
        print(f"Processing: {subject}")
        print(f"{'='*60}")
        
        output_dir = Path(config["project"]["output_dir"]) / subject
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Preprocess each modality
        print("\n[PHASE 1] Preprocessing...")
        import pandas as pd
        
        # IMU bioz - returns DataFrame
        imu_bioz_df = preprocess_imu(
            path=dataset["imu_bioz"]["path"],
            fs_out=dataset["imu_bioz"]["fs_out"],
        )
        print(f"  IMU bioz: {len(imu_bioz_df)} samples")
        
        # IMU wrist - returns DataFrame
        imu_wrist_df = preprocess_imu(
            path=dataset["imu_wrist"]["path"],
            fs_out=dataset["imu_wrist"]["fs_out"],
        )
        print(f"  IMU wrist: {len(imu_wrist_df)} samples")
        
        # PPG variants - write to file
        ppg_green_path = output_dir / "ppg_green_preprocessed.csv"
        if not ppg_green_path.exists():
            preprocess_ppg(
                in_path=dataset["ppg_green"]["path"],
                out_path=str(ppg_green_path),
                fs=dataset["ppg_green"]["fs_out"],
            )
        ppg_green_df = pd.read_csv(ppg_green_path)
        print(f"  PPG green: {len(ppg_green_df)} samples")
        
        ppg_infra_path = output_dir / "ppg_infra_preprocessed.csv"
        if not ppg_infra_path.exists():
            preprocess_ppg(
                in_path=dataset["ppg_infra"]["path"],
                out_path=str(ppg_infra_path),
                fs=dataset["ppg_infra"]["fs_out"],
            )
        ppg_infra_df = pd.read_csv(ppg_infra_path)
        print(f"  PPG infra: {len(ppg_infra_df)} samples")
        
        ppg_red_path = output_dir / "ppg_red_preprocessed.csv"
        if not ppg_red_path.exists():
            preprocess_ppg(
                in_path=dataset["ppg_red"]["path"],
                out_path=str(ppg_red_path),
                fs=dataset["ppg_red"]["fs_out"],
            )
        ppg_red_df = pd.read_csv(ppg_red_path)
        print(f"  PPG red: {len(ppg_red_df)} samples")
        
        # EDA - write to file
        eda_path = output_dir / "eda_preprocessed.csv"
        if not eda_path.exists():
            preprocess_eda(
                in_path=dataset["eda"]["path"],
                out_path=str(eda_path),
                fs=dataset["eda"]["fs_out"],
                do_resample=True,
            )
        eda_df = pd.read_csv(eda_path)
        print(f"  EDA: {len(eda_df)} samples")
        
        # RR - write to file
        rr_path = output_dir / "rr_preprocessed.csv"
        if not rr_path.exists():
            preprocess_rr(
                in_path=dataset["rr"]["path"],
                out_path=str(rr_path),
            )
        rr_df = pd.read_csv(rr_path)
        print(f"  RR: {len(rr_df)} samples")
        
        print("✓ Preprocessing complete")
        
        # Phase 2: Windowing
        print("\n[PHASE 2] Windowing...")
        win_sec = config["windowing"]["window_lengths_sec"][0]
        overlap = config["windowing"]["overlap"]
        
        imu_bioz_windows = create_windows(imu_bioz_df, fs=dataset["imu_bioz"]["fs_out"], win_sec=win_sec, overlap=overlap)
        imu_wrist_windows = create_windows(imu_wrist_df, fs=dataset["imu_wrist"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_green_windows = create_windows(ppg_green_df, fs=dataset["ppg_green"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_infra_windows = create_windows(ppg_infra_df, fs=dataset["ppg_infra"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_red_windows = create_windows(ppg_red_df, fs=dataset["ppg_red"]["fs_out"], win_sec=win_sec, overlap=overlap)
        eda_windows = create_windows(eda_df, fs=dataset["eda"]["fs_out"], win_sec=win_sec, overlap=overlap)
        # Skip RR for now - irregular sampling
        
        print(f"✓ Created windows: IMU bioz={len(imu_bioz_windows)}, wrist={len(imu_wrist_windows)}, PPG={len(ppg_green_windows)}, EDA={len(eda_windows)}")
        
        print("\n✅ Pipeline working with EXACT code from pascal_update!")
        print("   Phases 1-2 complete, ready to add 3-7")


if __name__ == "__main__":
    main()
