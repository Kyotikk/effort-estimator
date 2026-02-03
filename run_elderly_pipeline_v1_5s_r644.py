#!/usr/bin/env python3
"""
Pipeline runner for elderly patients across parsingsim 3, 4, and 5.

Runs preprocessing, feature extraction, alignment, and training for:
- sim_elderly3 (parsingsim3)
- sim_elderly4 (parsingsim4)  
- sim_elderly5 (parsingsim5)

Then combines and trains XGBoost (and optionally linear regression) model.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from ml.feature_selection_and_qc import select_and_prune_features, perform_pca_analysis, save_feature_selection_results

# Configuration for 3 elderly patients
SUBJECTS_CONFIG = {
    "sim_elderly3": {
        "data_root": "/Users/pascalschlegel/data/interim/parsingsim3",
        "label": "elderly3",
    },
    "sim_elderly4": {
        "data_root": "/Users/pascalschlegel/data/interim/parsingsim4",
        "label": "elderly4",
    },
    "sim_elderly5": {
        "data_root": "/Users/pascalschlegel/data/interim/parsingsim5",
        "label": "elderly5",
    },
}

WINDOW_LENGTHS = [10.0, 5.0, 2.0]  # Try in order until we find data


def find_file(subject_path, pattern_parts, exclude_gz=False):
    """Find a file matching pattern parts in subject directory."""
    base = Path(subject_path)
    for i, part in enumerate(pattern_parts):
        matches = list(base.glob(f"*{part}*"))
        if not matches:
            return None
        
        if i == len(pattern_parts) - 1 and exclude_gz:
            matches = [m for m in matches if not str(m).endswith('.gz')]
        
        if not matches:
            return None
        
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        base = matches[0]
    
    if base.is_dir():
        all_csv = list(base.glob("*.csv")) + list(base.glob("*.csv.gz"))
        
        if not all_csv:
            return None
        
        derived_patterns = ['rr_interval', 'shifted', 'processed', 'result', 'peak', 'baevsky']
        
        def is_raw_data(f):
            fname = f.name.lower()
            return not any(p in fname for p in derived_patterns)
        
        raw_files = [f for f in all_csv if is_raw_data(f)]
        candidates = raw_files if raw_files else all_csv
        
        if exclude_gz:
            uncompressed = [f for f in candidates if not str(f).endswith('.csv.gz')]
            if uncompressed:
                uncompressed.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(uncompressed[0])
            return None
        
        uncompressed = [f for f in candidates if not str(f).endswith('.csv.gz')]
        if uncompressed:
            uncompressed.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(uncompressed[0])
        
        gz_files = sorted([f for f in candidates if str(f).endswith('.csv.gz')], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if gz_files:
            return str(gz_files[0])
        
        return None
    
    return str(base)


def generate_config(subject, data_root):
    """Generate pipeline config for a specific subject."""
    subject_path = Path(data_root) / subject
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
        for k, v in required_files.items():
            print(f"    {k}: {v}")
        return None
    
    print(f"  Found files for {subject}:")
    print(f"    ADL (with Borg): {adl_path}")
    
    config = {
        "project": {
            "name": "effort_estimation",
            "output_dir": str(output_dir),
        },
        "datasets": [
            {
                "name": f"elderly_{subject}",
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
            "overlap": 0.1,
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
            "output_dir": str(output_dir / f"elderly_{subject}"),
            "window_lengths_sec": [10.0, 5.0, 2.0],
            "tolerance_s": {
                "2.0": 2.0,
                "5.0": 2.0,
                "10.0": 2.0,
            },
            "modalities": {
                "imu_bioz": f"{output_dir}/elderly_{subject}/imu_bioz/imu_features_{{window_length}}.csv",
                "imu_wrist": f"{output_dir}/elderly_{subject}/imu_wrist/imu_features_{{window_length}}.csv",
                "ppg_green": f"{output_dir}/elderly_{subject}/ppg_green/ppg_green_features_{{window_length}}.csv",
                "ppg_green_hrv": f"{output_dir}/elderly_{subject}/ppg_green/ppg_green_hrv_features_{{window_length}}.csv",
                "ppg_infra": f"{output_dir}/elderly_{subject}/ppg_infra/ppg_infra_features_{{window_length}}.csv",
                "ppg_infra_hrv": f"{output_dir}/elderly_{subject}/ppg_infra/ppg_infra_hrv_features_{{window_length}}.csv",
                "ppg_red": f"{output_dir}/elderly_{subject}/ppg_red/ppg_red_features_{{window_length}}.csv",
                "ppg_red_hrv": f"{output_dir}/elderly_{subject}/ppg_red/ppg_red_hrv_features_{{window_length}}.csv",
                "eda": f"{output_dir}/elderly_{subject}/eda/eda_features_{{window_length}}.csv",
                "eda_advanced": f"{output_dir}/elderly_{subject}/eda/eda_advanced_features_{{window_length}}.csv",
            },
        },
        "logging": {
            "verbose": True,
        },
    }
    
    return config


def run_subject_pipeline(subject, data_root):
    """Run pipeline for a single subject."""
    print(f"\n{'='*70}")
    print(f"SUBJECT: {subject} (from {data_root})")
    print(f"{'='*70}")
    
    # Generate config
    config = generate_config(subject, data_root)
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


def combine_datasets(subjects_config, window_length):
    """Combine aligned fused features from multiple subjects."""
    print(f"\n{'='*70}")
    print(f"COMBINING DATASETS (window={window_length}s)")
    print(f"{'='*70}")
    
    dfs = []
    for subject, cfg in subjects_config.items():
        data_root = cfg["data_root"]
        aligned_path = (
            Path(data_root)
            / subject
            / "effort_estimation_output"
            / f"elderly_{subject}"
            / f"fused_aligned_{window_length:.1f}s.csv"
        )
        
        if not aligned_path.exists():
            print(f"  ⚠ Missing: {aligned_path}")
            continue
        
        df = pd.read_csv(aligned_path)
        df["subject"] = subject
        df["label"] = cfg["label"]
        dfs.append(df)
        print(f"  ✓ {subject}: {len(df)} samples, {df['borg'].notna().sum()} labeled")
    
    if not dfs:
        print("✗ No data to combine!")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined: {len(combined)} total samples")
    print(f"  Labeled: {combined['borg'].notna().sum()}")
    
    return combined


def train_xgboost(df_labeled, pruned_cols, output_dir):
    """Train XGBoost with proper GroupKFold CV."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from scipy.stats import pearsonr
    import xgboost as xgb
    
    print(f"\n{'='*70}")
    print(f"TRAINING: XGBoost with GroupKFold CV")
    print(f"{'='*70}")
    
    X = df_labeled[pruned_cols].values
    y = df_labeled["borg"].values
    
    # Create activity groups from subject + borg changes
    activity_ids = []
    current_id = 0
    prev_subject = None
    prev_borg = None
    
    for i, (subj, borg) in enumerate(zip(df_labeled["subject"], y)):
        if subj != prev_subject or borg != prev_borg:
            current_id += 1
        activity_ids.append(current_id)
        prev_subject = subj
        prev_borg = borg
    
    groups = np.array(activity_ids)
    n_activities = len(np.unique(groups))
    print(f"  Activities detected: {n_activities}")
    print(f"  Samples: {len(y)}")
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    
    # GroupKFold CV
    n_splits = min(5, n_activities)
    cv = GroupKFold(n_splits=n_splits)
    
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
    
    # Metrics
    r, p = pearsonr(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"\n  Results (GroupKFold CV, {n_splits} folds):")
    print(f"    Pearson r: {r:.3f} (p={p:.2e})")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE: {mae:.2f}")
    
    # Feature importance
    model.fit(X, y)
    importance = pd.DataFrame({
        "feature": pruned_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\n  Top 10 features by importance:")
    for i, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save results
    results_dir = output_dir / "xgboost_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    
    results_df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred,
        "subject": df_labeled["subject"].values,
        "activity_id": groups,
    })
    results_df.to_csv(results_dir / "predictions.csv", index=False)
    
    # Summary
    summary = {
        "model": "XGBoost",
        "cv_method": f"GroupKFold ({n_splits} folds)",
        "n_samples": len(y),
        "n_activities": n_activities,
        "n_features": len(pruned_cols),
        "pearson_r": r,
        "p_value": p,
        "rmse": rmse,
        "mae": mae,
    }
    
    with open(results_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    
    print(f"\n  ✓ Results saved to {results_dir}")
    
    return model, r, rmse


def train_linear_regression(df_labeled, pruned_cols, output_dir):
    """Train Linear Regression with proper GroupKFold CV."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Linear Regression (Ridge) with GroupKFold CV")
    print(f"{'='*70}")
    
    X = df_labeled[pruned_cols].values
    y = df_labeled["borg"].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create activity groups
    activity_ids = []
    current_id = 0
    prev_subject = None
    prev_borg = None
    
    for i, (subj, borg) in enumerate(zip(df_labeled["subject"], y)):
        if subj != prev_subject or borg != prev_borg:
            current_id += 1
        activity_ids.append(current_id)
        prev_subject = subj
        prev_borg = borg
    
    groups = np.array(activity_ids)
    n_activities = len(np.unique(groups))
    
    # Ridge regression (with regularization)
    model = Ridge(alpha=1.0)
    
    # GroupKFold CV
    n_splits = min(5, n_activities)
    cv = GroupKFold(n_splits=n_splits)
    
    y_pred = cross_val_predict(model, X_scaled, y, groups=groups, cv=cv)
    
    # Metrics
    r, p = pearsonr(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    
    print(f"\n  Results (GroupKFold CV, {n_splits} folds):")
    print(f"    Pearson r: {r:.3f} (p={p:.2e})")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE: {mae:.2f}")
    
    # Feature coefficients
    model.fit(X_scaled, y)
    coefficients = pd.DataFrame({
        "feature": pruned_cols,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_)
    }).sort_values("abs_coefficient", ascending=False)
    
    print(f"\n  Top 10 features by |coefficient|:")
    for i, row in coefficients.head(10).iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")
    
    # Save results
    results_dir = output_dir / "linear_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    coefficients.to_csv(results_dir / "coefficients.csv", index=False)
    
    results_df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred,
        "subject": df_labeled["subject"].values,
        "activity_id": groups,
    })
    results_df.to_csv(results_dir / "predictions.csv", index=False)
    
    summary = {
        "model": "Ridge Regression",
        "cv_method": f"GroupKFold ({n_splits} folds)",
        "n_samples": len(y),
        "n_activities": n_activities,
        "n_features": len(pruned_cols),
        "pearson_r": r,
        "p_value": p,
        "rmse": rmse,
        "mae": mae,
    }
    
    with open(results_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    
    print(f"\n  ✓ Results saved to {results_dir}")
    
    return model, r, rmse


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Elderly patients pipeline (parsingsim 3, 4, 5)")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip individual subject pipelines (use cached results)",
    )
    parser.add_argument(
        "--xgboost",
        action="store_true",
        default=True,
        help="Train XGBoost model (default: True)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Also train linear regression model",
    )
    args = parser.parse_args()
    
    print("="*70)
    print("ELDERLY PATIENTS PIPELINE")
    print("Subjects: sim_elderly3, sim_elderly4, sim_elderly5")
    print("="*70)
    
    # Run individual subject pipelines
    succeeded = {}
    if not args.skip_pipeline:
        for subject, cfg in SUBJECTS_CONFIG.items():
            if run_subject_pipeline(subject, cfg["data_root"]):
                succeeded[subject] = cfg
            else:
                print(f"  Note: {subject} pipeline had issues, checking if partial data exists...")
                # Check if partial data exists (even if pipeline didn't fully complete)
                for wl in WINDOW_LENGTHS:
                    aligned_path = (
                        Path(cfg["data_root"])
                        / subject
                        / "effort_estimation_output"
                        / f"elderly_{subject}"
                        / f"fused_aligned_{wl:.1f}s.csv"
                    )
                    if aligned_path.exists():
                        print(f"  ✓ Found partial data for {subject} at {wl:.1f}s, including it")
                        succeeded[subject] = cfg
                        break
        
        if not succeeded:
            print("\n✗ No subjects completed successfully!")
            return 1
        
        print(f"\n✓ Completed: {len(succeeded)}/{len(SUBJECTS_CONFIG)} subjects")
    else:
        succeeded = SUBJECTS_CONFIG
        print("Skipping pipeline (using cached results)")
    
    # Try to combine datasets at each window length until we find one that works
    combined = None
    window_length = None
    for wl in WINDOW_LENGTHS:
        combined = combine_datasets(succeeded, wl)
        if combined is not None and len(combined) > 0:
            window_length = wl
            break
        print(f"  (No valid data at {wl:.1f}s, trying next...)")
    
    if combined is None or len(combined) == 0:
        print("\n✗ No data found at any window length!")
        return 1
    
    # Save combined dataset
    output_path = Path("/Users/pascalschlegel/data/interim/elderly_combined")
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_file = output_path / f"elderly_aligned_{window_length:.1f}s.csv"
    combined.to_csv(combined_file, index=False)
    print(f"\n✓ Saved combined dataset: {combined_file}")
    
    # Feature selection
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION + QC")
    print(f"{'='*70}")
    
    df_labeled = combined.dropna(subset=["borg"]).copy()
    
    # Remove metadata columns
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "label", "borg",
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
    feature_qc_dir = output_path / f"qc_{window_length:.1f}s"
    save_feature_selection_results(
        str(feature_qc_dir), 
        pruned_cols, 
        explained_df, 
        loadings_df, 
        top_loadings_df
    )
    
    print(f"  ✓ Feature selection complete: {len(pruned_cols)} features selected")
    print(f"  ✓ QC outputs saved to {feature_qc_dir}")
    
    # Train models
    if args.xgboost or not args.linear:
        train_xgboost(df_labeled, pruned_cols, output_path)
    
    if args.linear:
        train_linear_regression(df_labeled, pruned_cols, output_path)
    
    # Summary stats
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    for subject in succeeded:
        n_samples = len(combined[combined["subject"] == subject])
        n_labeled = combined[combined["subject"] == subject]["borg"].notna().sum()
        print(f"  {subject}: {n_samples} samples ({n_labeled} labeled)")
    
    print(f"\nTotal: {len(combined)} samples ({combined['borg'].notna().sum()} labeled)")
    print(f"Features before selection: {len(feature_cols)}")
    print(f"Features after selection: {len(pruned_cols)}")
    print(f"\nOutput directory: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
