#!/usr/bin/env python3
"""
Multi-subject pipeline runner.

Processes multiple subjects (sim_elderly3, sim_healthy3, sim_severe3)
through the full pipeline, then combines and trains a multi-subject model.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from pipeline.feature_selection_and_qc import select_and_prune_features, perform_pca_analysis, save_feature_selection_results

DATA_ROOT = "/Users/pascalschlegel/data/interim/parsingsim3"
SUBJECTS = ["sim_elderly3", "sim_healthy3", "sim_severe3"]
WINDOW_LENGTH = 10.0


def find_file(subject_path, pattern_parts, exclude_gz=False):
    """Find a file matching pattern parts in subject directory.
    
    For multiple matches, prefers the most recently modified file.
    If exclude_gz=True, prioritizes uncompressed .csv over .csv.gz files.
    """
    base = Path(subject_path)
    for i, part in enumerate(pattern_parts):
        matches = list(base.glob(f"*{part}*"))
        if not matches:
            return None
        
        # If this is the last part AND exclude_gz is True, filter out .gz files
        if i == len(pattern_parts) - 1 and exclude_gz:
            matches = [m for m in matches if not str(m).endswith('.gz')]
        
        # If no matches after filtering, return None
        if not matches:
            return None
        
        # Sort by modification time and pick the most recent
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        base = matches[0]
    
    # If we end up with a directory, find the actual file inside
    if base.is_dir():
        if exclude_gz:
            # Prioritize uncompressed .csv over .csv.gz
            csv_files = [f for f in base.glob("*.csv") if not str(f).endswith('.csv.gz')]
            if csv_files:
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(csv_files[0])
            return None
        
        # Standard search: try uncompressed .csv first, then .csv.gz
        uncompressed = [f for f in base.glob("*.csv") if not str(f).endswith('.csv.gz')]
        if uncompressed:
            uncompressed.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(uncompressed[0])
        
        gz_files = sorted(base.glob("*.csv.gz"), key=lambda x: x.stat().st_mtime, reverse=True)
        if gz_files:
            return str(gz_files[0])
        
        return None
    
    return str(base)


def generate_config(subject):
    """Generate pipeline config for a specific subject."""
    subject_path = Path(DATA_ROOT) / subject
    
    output_dir = subject_path / "effort_estimation_output"
    
    # Find all required data files
    imu_bioz_path = find_file(subject_path, ["corsano_bioz_acc"])
    imu_wrist_path = find_file(subject_path, ["corsano_wrist_acc"])
    ppg_green_path = find_file(subject_path, ["corsano_wrist_ppg2_green"])
    ppg_infra_path = find_file(subject_path, ["corsano_wrist_ppg2_infra"])
    ppg_red_path = find_file(subject_path, ["corsano_wrist_ppg2_red"])
    eda_path = find_file(subject_path, ["corsano_bioz_emography"])
    rr_path = find_file(subject_path, ["corsano_bioz_rr_interval"])
    adl_path = find_file(subject_path, ["scai_app", "ADLs"], exclude_gz=True)
    
    # Verify all files exist
    required_files = {
        "imu_bioz": imu_bioz_path,
        "imu_wrist": imu_wrist_path,
        "ppg_green": ppg_green_path,
        "ppg_infra": ppg_infra_path,
        "ppg_red": ppg_red_path,
        "eda": eda_path,
        "rr": rr_path,
        "adl": adl_path,
    }
    
    missing = [k for k, v in required_files.items() if not v]
    if missing:
        print(f"  ⚠ Missing files for {subject}: {missing}")
        return None
    
    config = {
        "project": {
            "name": "effort_estimation",
            "output_dir": str(output_dir),
        },
        "datasets": [
            {
                "name": f"parsingsim3_{subject}",
                "imu_bioz": {
                    "path": imu_bioz_path,
                    "fs_out": 32,
                },
                "imu_wrist": {
                    "path": imu_wrist_path,
                    "fs_out": 32,
                },
                "ppg_green": {
                    "path": ppg_green_path,
                    "fs_out": 32,
                },
                "ppg_infra": {
                    "path": ppg_infra_path,
                    "fs_out": 32,
                },
                "ppg_red": {
                    "path": ppg_red_path,
                    "fs_out": 32,
                },
                "eda": {
                    "path": eda_path,
                    "fs_out": 32,
                },
                "rr": {
                    "path": rr_path,
                    "fs_out": 1,
                },
            }
        ],
        "preprocessing": {
            "imu_bioz": {
                "noise_cutoff": 5.0,
                "gravity_cutoff": 0.3,
                "normalise": False,
            },
            "imu_wrist": {
                "noise_cutoff": 5.0,
                "gravity_cutoff": 0.3,
                "normalise": False,
            },
            "ppg_green": {
                "time_col": "time",
                "metric_id": "0x7e",
                "led_pd_pos": 6,
                "led": None,
                "do_resample": True,
                "apply_hpf": False,
            },
            "ppg_infra": {
                "time_col": "time",
                "metric_id": "0x7b",
                "led_pd_pos": 22,
                "led": None,
                "do_resample": True,
                "apply_hpf": True,
                "hpf_cutoff": 0.5,
            },
            "ppg_red": {
                "time_col": "time",
                "metric_id": "0x7c",
                "led_pd_pos": 182,
                "led": None,
                "do_resample": True,
                "apply_hpf": True,
                "hpf_cutoff": 0.5,
            },
            "rr": {
                "time_col": "time",
                "rr_col": "rr",
            },
            "eda": {
                "time_col": "time",
                "do_resample": True,
            },
        },
        "windowing": {
            "overlap": 0.7,
            "window_lengths_sec": [10.0, 5.0, 2.0],
        },
        "features": {
            "imu_bioz": {
                "modality": "imu",
                "signals": ["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"],
                "feature_set": "stat",
                "safe": True,
                "njobs": 1,
            },
            "imu_wrist": {
                "modality": "imu",
                "signals": ["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"],
                "feature_set": "stat",
                "safe": True,
                "njobs": 1,
            },
            "ppg_green": {
                "modality": "ppg_green",
                "prefix": "ppg_green_",
                "time_col": "t_sec",
                "signal_col": "value",
            },
            "ppg_infra": {
                "modality": "ppg_infra",
                "prefix": "ppg_infra_",
                "time_col": "t_sec",
                "signal_col": "value",
            },
            "ppg_red": {
                "modality": "ppg_red",
                "prefix": "ppg_red_",
                "time_col": "t_sec",
                "signal_col": "value",
            },
            "rr": {
                "modality": "rr",
                "prefix": "rr_",
                "time_col": "t_sec",
                "signal_col": "value",
            },
            "eda": {
                "modality": "eda",
                "prefix": "eda_",
                "time_col": "t_sec",
                "cc_col": "eda_cc",
                "stress_col": "eda_stress_skin",
            },
        },
        "targets": {
            "imu": {
                "adl_path": adl_path,
            }
        },
        "fusion": {
            "output_dir": str(output_dir / f"parsingsim3_{subject}"),
            "window_lengths_sec": [10.0, 5.0, 2.0],
            "tolerance_s": {
                "2.0": 2.0,
                "5.0": 2.0,
                "10.0": 2.0,
            },
            "modalities": {
                "imu_bioz": f"{output_dir}/parsingsim3_{subject}/imu_bioz/imu_features_{{window_length}}.csv",
                "imu_wrist": f"{output_dir}/parsingsim3_{subject}/imu_wrist/imu_features_{{window_length}}.csv",
                "ppg_green": f"{output_dir}/parsingsim3_{subject}/ppg_green/ppg_green_features_{{window_length}}.csv",
                "ppg_infra": f"{output_dir}/parsingsim3_{subject}/ppg_infra/ppg_infra_features_{{window_length}}.csv",
                "ppg_red": f"{output_dir}/parsingsim3_{subject}/ppg_red/ppg_red_features_{{window_length}}.csv",
                "eda": f"{output_dir}/parsingsim3_{subject}/eda/eda_features_{{window_length}}.csv",
            },
        },
        "logging": {
            "verbose": True,
        },
    }
    
    return config


def run_subject_pipeline(subject):
    """Run pipeline for a single subject."""
    print(f"\n{'='*70}")
    print(f"SUBJECT: {subject}")
    print(f"{'='*70}")
    
    # Generate config
    config = generate_config(subject)
    if not config:
        print(f"✗ Could not generate config for {subject}")
        return False
    
    # Write config to temporary file
    config_path = Path(f"/tmp/pipeline_{subject}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✓ Config: {config_path}")
    
    # Run pipeline
    print(f"▶ Running pipeline...")
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", str(config_path)],
        cwd="/Users/pascalschlegel/effort-estimator",
    )
    
    if result.returncode != 0:
        print(f"✗ Pipeline failed for {subject}")
        return False
    
    print(f"✓ Pipeline completed for {subject}")
    return True


def combine_datasets(subjects, window_length):
    """Combine aligned fused features from multiple subjects."""
    print(f"\n{'='*70}")
    print(f"COMBINING DATASETS (window={window_length}s)")
    print(f"{'='*70}")
    
    dfs = []
    for subject in subjects:
        aligned_path = (
            Path(DATA_ROOT)
            / subject
            / "effort_estimation_output"
            / f"parsingsim3_{subject}"
            / f"fused_aligned_{window_length:.1f}s.csv"
        )
        
        if not aligned_path.exists():
            print(f"  ⚠ Missing: {aligned_path}")
            continue
        
        df = pd.read_csv(aligned_path)
        df["subject"] = subject
        dfs.append(df)
        print(f"  ✓ {subject}: {len(df)} samples")
    
    if not dfs:
        print("✗ No data to combine!")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined: {len(combined)} total samples")
    print(f"  Labeled: {combined['borg'].notna().sum()}")
    
    return combined


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-subject pipeline runner")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=SUBJECTS,
        help="Subjects to process",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip individual subject pipelines (use cached results)",
    )
    args = parser.parse_args()
    
    # Run individual subject pipelines
    if not args.skip_pipeline:
        succeeded = []
        for subject in args.subjects:
            if run_subject_pipeline(subject):
                succeeded.append(subject)
        
        if not succeeded:
            print("\n✗ No subjects completed successfully!")
            return 1
        
        print(f"\n✓ Completed: {len(succeeded)}/{len(args.subjects)} subjects")
    else:
        succeeded = args.subjects
    
    # Combine datasets
    combined = combine_datasets(succeeded, WINDOW_LENGTH)
    if combined is None:
        return 1
    
    # Save combined dataset
    output_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined")
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_file = output_path / f"multisub_aligned_{WINDOW_LENGTH:.1f}s.csv"
    combined.to_csv(combined_file, index=False)
    print(f"\n✓ Saved combined dataset: {combined_file}")
    
    # Run feature selection on combined dataset
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION + QC")
    print(f"{'='*70}")
    
    df_labeled = combined.dropna(subset=["borg"]).copy()
    
    # Remove metadata columns
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            return True
        return False
    
    feature_cols = [col for col in df_labeled.columns if not is_metadata(col)]
    X = df_labeled[feature_cols].values
    y = df_labeled["borg"].values
    
    print(f"  ✓ {len(feature_cols)} features (after metadata removal)")
    print(f"  ✓ {len(df_labeled)} labeled samples for feature selection")
    
    # Feature selection + pruning
    print(f"\n  Selecting top 100 features by correlation...")
    pruned_indices, pruned_cols = select_and_prune_features(
        X, y, feature_cols, 
        corr_threshold=0.90, 
        top_n=100
    )
    
    # PCA analysis
    X_pruned = X[:, pruned_indices]
    explained_df, loadings_df, top_loadings_df, pcs_for_targets = perform_pca_analysis(
        X_pruned, pruned_cols
    )
    
    # Save feature selection outputs
    feature_qc_dir = output_path / f"qc_{WINDOW_LENGTH:.1f}s"
    save_feature_selection_results(
        str(feature_qc_dir), 
        pruned_cols, 
        explained_df, 
        loadings_df, 
        top_loadings_df
    )
    
    print(f"  ✓ Feature selection complete: {len(pruned_cols)} features selected")
    print(f"  ✓ QC outputs saved to {feature_qc_dir}")
    
    # Summary stats
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for subject in succeeded:
        n_samples = len(combined[combined["subject"] == subject])
        n_labeled = combined[combined["subject"] == subject]["borg"].notna().sum()
        print(f"  {subject}: {n_samples} samples ({n_labeled} labeled)")
    
    print(f"\nTotal: {len(combined)} samples ({combined['borg'].notna().sum()} labeled)")
    print(f"Features before selection: {len(feature_cols)}")
    print(f"Features after selection: {len(pruned_cols)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
