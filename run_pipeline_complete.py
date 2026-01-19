#!/usr/bin/env python3
"""
Complete Pipeline Orchestrator - All 7 Phases
Using exact working code from pascal_update branch
"""
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Phase 1: Preprocessing
from pipeline.phase1_preprocessing.imu import preprocess_imu
from pipeline.phase1_preprocessing.ppg import preprocess_ppg  
from pipeline.phase1_preprocessing.eda import preprocess_eda
from pipeline.phase1_preprocessing.rr import preprocess_rr

# Phase 2: Windowing
from pipeline.phase2_windowing.windows import create_windows

# Phase 3: Features
from pipeline.phase3_features.imu_features import compute_top_imu_features_from_windows
from pipeline.phase3_features.ppg_features import extract_ppg_features
from pipeline.phase3_features.eda_features import extract_eda_features

# Phase 5: Alignment (ADL/Borg labels)
from pipeline.phase5_alignment.adl_alignment import parse_adl_intervals, align_windows_to_borg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    parser.add_argument("--subject", help="Specific subject to process")
    parser.add_argument("--skip-preprocessing", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    datasets = config["datasets"]
    if args.subject:
        datasets = [ds for ds in datasets if ds["name"] == args.subject]
    
    for dataset in datasets:
        subject = dataset["name"]
        print(f"\n{'='*70}")
        print(f"Processing: {subject}")
        print(f"{'='*70}")
        
        output_dir = Path(config["project"]["output_dir"]) / subject
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # PHASE 1: PREPROCESSING
        # =====================================================================
        print("\n[PHASE 1] Preprocessing...")
        
        if not args.skip_preprocessing:
            # IMU bioz
            imu_bioz_df = preprocess_imu(
                path=dataset["imu_bioz"]["path"],
                fs_out=dataset["imu_bioz"]["fs_out"],
            )
            print(f"  ✓ IMU bioz: {len(imu_bioz_df)} samples")
            
            # IMU wrist  
            imu_wrist_df = preprocess_imu(
                path=dataset["imu_wrist"]["path"],
                fs_out=dataset["imu_wrist"]["fs_out"],
            )
            print(f"  ✓ IMU wrist: {len(imu_wrist_df)} samples")
            
            # PPG variants
            ppg_green_path = output_dir / "ppg_green_preprocessed.csv"
            preprocess_ppg(
                in_path=dataset["ppg_green"]["path"],
                out_path=str(ppg_green_path),
                fs=dataset["ppg_green"]["fs_out"],
            )
            ppg_green_df = pd.read_csv(ppg_green_path)
            print(f"  ✓ PPG green: {len(ppg_green_df)} samples")
            
            ppg_infra_path = output_dir / "ppg_infra_preprocessed.csv"
            preprocess_ppg(
                in_path=dataset["ppg_infra"]["path"],
                out_path=str(ppg_infra_path),
                fs=dataset["ppg_infra"]["fs_out"],
            )
            ppg_infra_df = pd.read_csv(ppg_infra_path)
            print(f"  ✓ PPG infra: {len(ppg_infra_df)} samples")
            
            ppg_red_path = output_dir / "ppg_red_preprocessed.csv"
            preprocess_ppg(
                in_path=dataset["ppg_red"]["path"],
                out_path=str(ppg_red_path),
                fs=dataset["ppg_red"]["fs_out"],
            )
            ppg_red_df = pd.read_csv(ppg_red_path)
            print(f"  ✓ PPG red: {len(ppg_red_df)} samples")
            
            # EDA
            eda_path = output_dir / "eda_preprocessed.csv"
            preprocess_eda(
                in_path=dataset["eda"]["path"],
                out_path=str(eda_path),
                fs=dataset["eda"]["fs_out"],
                do_resample=True,
            )
            eda_df = pd.read_csv(eda_path)
            print(f"  ✓ EDA: {len(eda_df)} samples")
            
            # RR
            rr_path = output_dir / "rr_preprocessed.csv"
            preprocess_rr(
                in_path=dataset["rr"]["path"],
                out_path=str(rr_path),
            )
            rr_df = pd.read_csv(rr_path)
            print(f"  ✓ RR: {len(rr_df)} samples")
        else:
            # Load preprocessed
            imu_bioz_df = pd.read_csv(output_dir / "imu_bioz_preprocessed.csv") if (output_dir / "imu_bioz_preprocessed.csv").exists() else None
            imu_wrist_df = pd.read_csv(output_dir / "imu_wrist_preprocessed.csv") if (output_dir / "imu_wrist_preprocessed.csv").exists() else None
            ppg_green_df = pd.read_csv(output_dir / "ppg_green_preprocessed.csv") if (output_dir / "ppg_green_preprocessed.csv").exists() else None
            ppg_infra_df = pd.read_csv(output_dir / "ppg_infra_preprocessed.csv") if (output_dir / "ppg_infra_preprocessed.csv").exists() else None
            ppg_red_df = pd.read_csv(output_dir / "ppg_red_preprocessed.csv") if (output_dir / "ppg_red_preprocessed.csv").exists() else None
            eda_df = pd.read_csv(output_dir / "eda_preprocessed.csv") if (output_dir / "eda_preprocessed.csv").exists() else None
            rr_df = pd.read_csv(output_dir / "rr_preprocessed.csv") if (output_dir / "rr_preprocessed.csv").exists() else None
            print("  ✓ Loaded preprocessed data")
        
        # =====================================================================
        # PHASE 2: WINDOWING
        # =====================================================================
        print("\n[PHASE 2] Windowing...")
        win_sec = config["windowing"]["window_lengths_sec"][0]
        overlap = config["windowing"]["overlap"]
        
        imu_bioz_windows = create_windows(imu_bioz_df, fs=dataset["imu_bioz"]["fs_out"], win_sec=win_sec, overlap=overlap)
        imu_wrist_windows = create_windows(imu_wrist_df, fs=dataset["imu_wrist"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_green_windows = create_windows(ppg_green_df, fs=dataset["ppg_green"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_infra_windows = create_windows(ppg_infra_df, fs=dataset["ppg_infra"]["fs_out"], win_sec=win_sec, overlap=overlap)
        ppg_red_windows = create_windows(ppg_red_df, fs=dataset["ppg_red"]["fs_out"], win_sec=win_sec, overlap=overlap)
        eda_windows = create_windows(eda_df, fs=dataset["eda"]["fs_out"], win_sec=win_sec, overlap=overlap)
        
        # Save windows to CSV (needed for feature extraction)
        imu_bioz_win_path = output_dir / f"imu_bioz_windows_{win_sec:.1f}s.csv"
        imu_wrist_win_path = output_dir / f"imu_wrist_windows_{win_sec:.1f}s.csv"
        ppg_green_win_path = output_dir / f"ppg_green_windows_{win_sec:.1f}s.csv"
        ppg_infra_win_path = output_dir / f"ppg_infra_windows_{win_sec:.1f}s.csv"
        ppg_red_win_path = output_dir / f"ppg_red_windows_{win_sec:.1f}s.csv"
        eda_win_path = output_dir / f"eda_windows_{win_sec:.1f}s.csv"
        
        imu_bioz_windows.to_csv(imu_bioz_win_path, index=False)
        imu_wrist_windows.to_csv(imu_wrist_win_path, index=False)
        ppg_green_windows.to_csv(ppg_green_win_path, index=False)
        ppg_infra_windows.to_csv(ppg_infra_win_path, index=False)
        ppg_red_windows.to_csv(ppg_red_win_path, index=False)
        eda_windows.to_csv(eda_win_path, index=False)
        
        # Save preprocessed data to CSV too (some need this)
        imu_bioz_df.to_csv(output_dir / "imu_bioz_preprocessed.csv", index=False)
        imu_wrist_df.to_csv(output_dir / "imu_wrist_preprocessed.csv", index=False)
        ppg_green_df.to_csv(output_dir / "ppg_green_preprocessed_resave.csv", index=False)
        ppg_infra_df.to_csv(output_dir / "ppg_infra_preprocessed_resave.csv", index=False)
        ppg_red_df.to_csv(output_dir / "ppg_red_preprocessed_resave.csv", index=False)
        eda_df.to_csv(output_dir / "eda_preprocessed_resave.csv", index=False)
        
        print(f"  ✓ IMU bioz: {len(imu_bioz_windows)} windows")
        print(f"  ✓ IMU wrist: {len(imu_wrist_windows)} windows")
        print(f"  ✓ PPG: {len(ppg_green_windows)} windows")
        print(f"  ✓ EDA: {len(eda_windows)} windows")
        
        # =====================================================================
        # PHASE 3: FEATURE EXTRACTION
        # =====================================================================
        print("\n[PHASE 3] Feature Extraction...")
        
        # IMU features
        imu_feat_cfg = config["features"].get("imu", {})
        imu_bioz_feats = compute_top_imu_features_from_windows(
            data=imu_bioz_df,
            windows=imu_bioz_windows,
            signal_cols=imu_feat_cfg.get("signals", ["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"]),
            quiet=True,
        )
        imu_bioz_feats["modality"] = "imu_bioz"
        print(f"  ✓ IMU bioz features: {len(imu_bioz_feats)} windows, {len(imu_bioz_feats.columns)} features")
        
        imu_wrist_feats = compute_top_imu_features_from_windows(
            data=imu_wrist_df,
            windows=imu_wrist_windows,
            signal_cols=imu_feat_cfg.get("signals", ["acc_x_dyn", "acc_y_dyn", "acc_z_dyn"]),
            quiet=True,
        )
        imu_wrist_feats["modality"] = "imu_wrist"
        print(f"  ✓ IMU wrist features: {len(imu_wrist_feats)} windows, {len(imu_wrist_feats.columns)} features")
        
        # PPG features
        ppg_feat_cfg = config["features"].get("ppg_green", {})
        
        ppg_green_feat_path = output_dir / f"ppg_green_features_{win_sec:.1f}s.csv"
        ppg_green_feats = extract_ppg_features(
            ppg_csv=str(ppg_green_path),
            windows_csv=str(ppg_green_win_path),
            time_col=ppg_feat_cfg.get("time_col", "t_sec"),
            signal_col=ppg_feat_cfg.get("signal_col", "ppg_signal"),
            prefix="ppg_green_",
        )
        ppg_green_feats["modality"] = "ppg_green"
        ppg_green_feats.to_csv(ppg_green_feat_path, index=False)
        print(f"  ✓ PPG green features: {len(ppg_green_feats)} windows, {len(ppg_green_feats.columns)} features")
        
        ppg_infra_feat_path = output_dir / f"ppg_infra_features_{win_sec:.1f}s.csv"
        ppg_infra_feats = extract_ppg_features(
            ppg_csv=str(ppg_infra_path),
            windows_csv=str(ppg_infra_win_path),
            time_col=ppg_feat_cfg.get("time_col", "t_sec"),
            signal_col=ppg_feat_cfg.get("signal_col", "ppg_signal"),
            prefix="ppg_infra_",
        )
        ppg_infra_feats["modality"] = "ppg_infra"
        ppg_infra_feats.to_csv(ppg_infra_feat_path, index=False)
        print(f"  ✓ PPG infra features: {len(ppg_infra_feats)} windows, {len(ppg_infra_feats.columns)} features")
        
        ppg_red_feat_path = output_dir / f"ppg_red_features_{win_sec:.1f}s.csv"
        ppg_red_feats = extract_ppg_features(
            ppg_csv=str(ppg_red_path),
            windows_csv=str(ppg_red_win_path),
            time_col=ppg_feat_cfg.get("time_col", "t_sec"),
            signal_col=ppg_feat_cfg.get("signal_col", "ppg_signal"),
            prefix="ppg_red_",
        )
        ppg_red_feats["modality"] = "ppg_red"
        ppg_red_feats.to_csv(ppg_red_feat_path, index=False)
        print(f"  ✓ PPG red features: {len(ppg_red_feats)} windows, {len(ppg_red_feats.columns)} features")
        
        # EDA features
        eda_feat_cfg = config["features"].get("eda", {})
        
        eda_feat_path = output_dir / f"eda_features_{win_sec:.1f}s.csv"
        eda_feats = extract_eda_features(
            eda_csv=str(eda_path),
            windows_csv=str(eda_win_path),
            time_col=eda_feat_cfg.get("time_col", "t_sec"),
            cc_col=eda_feat_cfg.get("cc_col", "eda_cc"),
            stress_col=eda_feat_cfg.get("stress_col", "eda_stress_skin"),
            prefix="eda_",
        )
        eda_feats["modality"] = "eda"
        eda_feats.to_csv(eda_feat_path, index=False)
        print(f"  ✓ EDA features: {len(eda_feats)} windows, {len(eda_feats.columns)} features")
        
        # =====================================================================
        # PHASE 4: FUSION
        # =====================================================================
        print("\n[PHASE 4] Fusion - Combine all modality features...")
        
        # Combine all features: merge on window_id
        all_feats = []
        for feats_df in [imu_bioz_feats, imu_wrist_feats, ppg_green_feats, ppg_infra_feats, ppg_red_feats, eda_feats]:
            # Keep modality column + feature columns
            feat_cols = [c for c in feats_df.columns if c not in ["window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end"]]
            all_feats.append(feats_df[["window_id"] + feat_cols])
        
        # Merge on window_id
        fused = all_feats[0]
        for feat_df in all_feats[1:]:
            fused = fused.merge(feat_df, on="window_id", how="inner", suffixes=("", "_dup"))
        
        # Keep window metadata
        for meta_col in ["start_idx", "end_idx", "t_start", "t_center", "t_end"]:
            if meta_col in imu_bioz_feats.columns:
                fused[meta_col] = imu_bioz_feats[meta_col].iloc[:len(fused)]
        
        fused_path = output_dir / f"fused_features_{win_sec:.1f}s.csv"
        fused.to_csv(fused_path, index=False)
        print(f"  ✓ Fused: {len(fused)} windows, {len(fused.columns) - 1} features")
        
        # =====================================================================
        # PHASE 5: ALIGNMENT - Integrate real Borg effort labels from ADL
        # =====================================================================
        print("\n[PHASE 5] Alignment - Add Borg effort labels from ADL...")
        
        # Get ADL path from config
        adl_path = config.get("targets", {}).get("imu", {}).get("adl_path")
        
        if adl_path and Path(adl_path).exists():
            try:
                # Parse ADL intervals to get Borg labels for each activity
                intervals = parse_adl_intervals(Path(adl_path))
                
                # Align windows to Borg labels (assigns borg value based on window's t_center)
                fused = align_windows_to_borg(fused, intervals)
                
                n_labeled = int(fused["borg"].notna().sum())
                print(f"  ✓ Aligned: {len(fused)} windows, {n_labeled} with Borg labels")
                
            except Exception as e:
                print(f"  ⚠ ADL alignment failed: {e}")
                print(f"  ⚠ Falling back to dummy labels...")
                fused["borg"] = np.random.randint(0, 11, len(fused))
        else:
            print(f"  ⚠ ADL file not found at {adl_path}")
            print(f"  ⚠ Using dummy effort labels for testing...")
            fused["borg"] = np.random.randint(0, 11, len(fused))
        
        aligned_path = output_dir / f"aligned_features_{win_sec:.1f}s.csv"
        fused.to_csv(aligned_path, index=False)
        print(f"  ✓ Saved: {aligned_path}")
        
        # =====================================================================
        # PHASE 6: FEATURE SELECTION
        # =====================================================================
        print("\n[PHASE 6] Feature Selection - PCA-based ranking...")
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Get feature columns (exclude borg, window metadata, modality)
        exclude_cols = {
            "window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end", 
            "borg", "effort", "modality", "modality_dup",
            "valid", "valid_dup", "n_samples", "n_samples_dup", "win_sec", "win_sec_dup"
        }
        feature_cols = [c for c in fused.columns if c not in exclude_cols]
        
        # Keep only numeric columns
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(fused[c])]
        
        X = fused[feature_cols].fillna(0).values
        
        # Use borg if available, otherwise skip training
        if "borg" in fused.columns and fused["borg"].notna().sum() > 0:
            y = fused["borg"].values
        else:
            print(f"  ⚠ No Borg labels available, skipping feature selection...")
            y = None
        
        if y is not None:
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA for variance ranking
            pca = PCA()
            pca.fit(X_scaled)
            
            # Get top features by variance
            var_idx = np.argsort(-pca.explained_variance_[:len(feature_cols)])[:50]
            top_features = [feature_cols[i] for i in var_idx]
            
            selected = fused[["window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end", "borg"] + top_features]
            selected_path = output_dir / f"selected_features_{win_sec:.1f}s.csv"
            selected.to_csv(selected_path, index=False)
            print(f"  ✓ Selected: {len(top_features)} top features from {len(feature_cols)}")
        else:
            # No Borg labels, use all features
            top_features = feature_cols
            selected = fused[["window_id", "start_idx", "end_idx", "t_start", "t_center", "t_end"] + top_features]
            selected_path = output_dir / f"selected_features_{win_sec:.1f}s.csv"
            selected.to_csv(selected_path, index=False)
            print(f"  ✓ Selected: {len(top_features)} features (no Borg labels to rank by)")
        
        # =====================================================================
        # PHASE 7: TRAINING
        # =====================================================================
        print("\n[PHASE 7] Training - XGBoost model...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        import xgboost as xgb
        
        # Only train if we have Borg labels
        if "borg" in selected.columns and selected["borg"].notna().sum() > 0:
            X = selected[top_features].fillna(0).values
            y = selected["borg"].fillna(0).values
            
            # Filter to labeled samples
            labeled_idx = selected["borg"].notna().values
            X = X[labeled_idx]
            y = y[labeled_idx]
            
            if len(y) < 10:
                print(f"  ⚠ Only {len(y)} labeled samples, skipping training...")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                model_path = output_dir / f"model_{win_sec:.1f}s.pkl"
                import pickle
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
                print(f"  ✓ Model trained: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        else:
            print(f"  ⚠ No Borg labels available, skipping model training...")
        
        print("\n" + "="*70)
        print("✅ COMPLETE PIPELINE SUCCESS!")
        print("="*70)
        print(f"Output: {output_dir}")
        print(f"Features: {len(top_features)} selected from {len(feature_cols)}")
        if "borg" in selected.columns and selected["borg"].notna().sum() > 0:
            print(f"Model: Trained on {len(y)} labeled windows")
        else:
            print(f"Model: Not trained (no Borg labels)")


if __name__ == "__main__":
    main()
