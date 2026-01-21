#!/usr/bin/env python3
# run_pipeline.py

import yaml
from pathlib import Path
import sys
import pandas as pd
import subprocess

from phases.phase1_preprocessing.imu import preprocess_imu
from phases.phase1_preprocessing.ppg import preprocess_ppg
from phases.phase1_preprocessing.eda import preprocess_eda
from phases.phase1_preprocessing.rr import preprocess_rr
from phases.phase1_preprocessing.ecg import preprocess_ecg

from phases.phase2_windowing.windows import create_windows
from phases.phase3_features.manual_features_imu import compute_top_imu_features_from_windows

from phases.phase5_alignment.run_target_alignment import run_alignment
from phases.phase5_alignment.adl_alignment import parse_adl_intervals
from phases.phase5_alignment.compute_hrv_recovery import align_windows_to_hrv_recovery, align_windows_to_hrv_recovery_rr
from phases.phase4_fusion.run_fusion import main as run_fusion
from phases.phase6_feature_selection.feature_selection_and_qc import main as run_feature_selection


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ensure_modality_col(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if "modality" not in df.columns:
        df["modality"] = modality
    return df


def run_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)

    output_root = Path(cfg["project"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    overlap = float(cfg["windowing"]["overlap"])
    win_lengths = cfg["windowing"]["window_lengths_sec"]
    
    # Get target type, RR path, ADL offset, and recovery params from config
    target_type = cfg.get("targets", {}).get("target_type", "borg")
    rr_path = cfg.get("targets", {}).get("imu", {}).get("rr_path", None)
    rmssd_windows_path = cfg.get("targets", {}).get("imu", {}).get("rmssd_windows_path", None)
    adl_offset_hours = float(cfg.get("targets", {}).get("imu", {}).get("adl_offset_hours", 0.0))
    rec_cfg = cfg.get("targets", {}).get("recovery", {})
    rec_start = float(rec_cfg.get("rec_start", 10.0))
    rec_end = float(rec_cfg.get("rec_end", 60.0))

    for dataset in cfg["datasets"]:
        name = dataset["name"]
        print(f"\n=== Processing dataset: {name} ===")

        ds_out = output_root / name
        ds_out.mkdir(parents=True, exist_ok=True)

        # ---------- MODALITY: IMU (multiple types: bioz, wrist) ----------
        imu_types = ["imu_bioz", "imu_wrist"]
        
        for imu_type in imu_types:
            if imu_type not in dataset:
                print(f"⚠ Skipping {imu_type}: not in dataset")
                continue
                
            if imu_type not in cfg["preprocessing"] or imu_type not in cfg["features"]:
                print(f"⚠ Skipping {imu_type}: missing config")
                continue
                
            print(f"▶ {imu_type}: preprocessing")
            imu_cfg = cfg["preprocessing"][imu_type]
            feat_cfg_imu = cfg["features"][imu_type]

            imu_path = dataset[imu_type]["path"]
            fs_imu = float(dataset[imu_type]["fs_out"])

            imu_out_dir = ds_out / imu_type
            imu_out_dir.mkdir(parents=True, exist_ok=True)

            imu_clean_path = imu_out_dir / "imu_preprocessed.csv"
            if imu_clean_path.exists():
                imu_df = pd.read_csv(imu_clean_path)
                print(f"  Loaded {imu_clean_path}")
            else:
                imu_df = preprocess_imu(
                    path=imu_path,
                    fs_out=fs_imu,
                    noise_cutoff=imu_cfg["noise_cutoff"],
                    gravity_cutoff=imu_cfg["gravity_cutoff"],
                )
                imu_df.to_csv(imu_clean_path, index=False)
                print(f"  Saved {imu_clean_path}")

            # Windowing
            print(f"  Windowing {imu_type}...")
            window_length_sec = float(win_lengths[0])
            windows_path = imu_out_dir / f"imu_windows_{window_length_sec:.1f}s.csv"
            
            if not windows_path.exists():
                windows_df = create_windows(
                    df=imu_df,
                    fs=fs_imu,
                    win_sec=window_length_sec,
                    overlap=overlap,
                )
                windows_df.to_csv(windows_path, index=False)
            else:
                windows_df = pd.read_csv(windows_path)
            
            n_windows = len(windows_df)
            print(f"    ✓ Created {n_windows} windows ({window_length_sec:.1f}s)")

            # Feature extraction
            print(f"  Extracting features for {imu_type}...")
            signal_cols = feat_cfg_imu["signals"]
            
            for win_sec in win_lengths:
                win_sec = float(win_sec)
                windows_path_sec = imu_out_dir / f"imu_windows_{win_sec:.1f}s.csv"
                features_path = imu_out_dir / f"imu_features_{win_sec:.1f}s.csv"
                
                if not windows_path_sec.exists():
                    windows_df_sec = create_windows(
                        df=imu_df,
                        fs=fs_imu,
                        win_sec=win_sec,
                        overlap=overlap,
                    )
                    windows_df_sec.to_csv(windows_path_sec, index=False)
                else:
                    windows_df_sec = pd.read_csv(windows_path_sec)
                
                if not features_path.exists():
                    feat_df = compute_top_imu_features_from_windows(
                        data=imu_df,
                        windows=windows_df_sec,
                        signal_cols=signal_cols,
                    )
                    feat_df.to_csv(features_path, index=False)
                    print(f"    ✓ Features {win_sec:.1f}s: {len(feat_df)} windows, {len(feat_df.columns)-1} features")

        # ---------- MODALITY: PPG variants (ppg_green, ppg_infra, ppg_red) ----------
        ppg_variants = {k: v for k, v in dataset.items() if k.startswith("ppg_")}
        for ppg_key, ppg_data in ppg_variants.items():
            if ppg_key not in cfg["preprocessing"] or ppg_key not in cfg["features"]:
                print(f"⚠ Skipping {ppg_key}: missing config")
                continue
            
            print(f"▶ {ppg_key}: preprocessing")
            ppg_cfg = cfg["preprocessing"][ppg_key]
            feat_cfg_ppg = cfg["features"][ppg_key]
            
            ppg_path = ppg_data["path"]
            fs_ppg = float(ppg_data["fs_out"])
            
            ppg_out_dir = ds_out / ppg_key
            ppg_out_dir.mkdir(parents=True, exist_ok=True)
            
            ppg_clean_path = ppg_out_dir / f"{ppg_key}_preprocessed.csv"
            if ppg_clean_path.exists():
                ppg_df = pd.read_csv(ppg_clean_path)
                print(f"  Loaded {ppg_clean_path}")
            else:
                preprocess_ppg(
                    in_path=ppg_path,
                    out_path=str(ppg_clean_path),
                    fs=fs_ppg,
                    time_col=ppg_cfg.get("time_col", "time"),
                    metric_id=ppg_cfg.get("metric_id", None),
                    led_pd_pos=ppg_cfg.get("led_pd_pos", None),
                    led=ppg_cfg.get("led", None),
                    do_resample=ppg_cfg.get("do_resample", True),
                    apply_hpf=ppg_cfg.get("apply_hpf", False),
                    hpf_cutoff=ppg_cfg.get("hpf_cutoff", 0.5),
                )
                ppg_df = pd.read_csv(ppg_clean_path)
                print(f"  Saved {ppg_clean_path}")

        # ---------- MODALITY: ECG → RR (optional) ----------
        # If ECG is provided, process it to extract RR intervals
        has_ecg = "ecg" in dataset
        if has_ecg:
            if "ecg" in cfg["preprocessing"]:
                print("▶ ECG: preprocessing → extracting RR intervals")
                ecg_cfg = cfg["preprocessing"]["ecg"]

                ecg_path = dataset["ecg"]["path"]
                fs_ecg = float(dataset["ecg"].get("fs", 125.0))  # Default 125 Hz for VitalNK

                ecg_out_dir = ds_out / "ecg"
                ecg_out_dir.mkdir(parents=True, exist_ok=True)

                # Output goes to rr_preprocessed.csv (same format as RR modality)
                rr_out_dir = ds_out / "rr"
                rr_out_dir.mkdir(parents=True, exist_ok=True)
                rr_clean_path = rr_out_dir / "rr_preprocessed.csv"
                
                if rr_clean_path.exists():
                    print(f"  RR intervals already exist: {rr_clean_path}")
                else:
                    preprocess_ecg(
                        in_path=ecg_path,
                        out_path=str(rr_clean_path),
                        time_col=ecg_cfg.get("time_col", "time"),
                        ecg_col=ecg_cfg.get("ecg_col", "ecg"),
                        fs=fs_ecg,
                        min_rr_ms=ecg_cfg.get("min_rr_ms", 300),
                        max_rr_ms=ecg_cfg.get("max_rr_ms", 2000),
                    )
                    print(f"  ECG → RR intervals saved: {rr_clean_path}")
                
                # Mark that we now have RR data (from ECG)
                has_rr = True

        # ---------- MODALITY: RR (optional) ----------
        has_rr = "rr" in dataset or has_ecg
        if has_rr:
            if "rr" in cfg["preprocessing"] and "rr" in cfg["features"]:
                print("▶ RR: preprocessing")
                rr_cfg = cfg["preprocessing"]["rr"]
                feat_cfg_rr = cfg["features"]["rr"]

                rr_path = dataset["rr"]["path"]
                fs_rr = float(dataset["rr"]["fs_out"])

                rr_out_dir = ds_out / "rr"
                rr_out_dir.mkdir(parents=True, exist_ok=True)

                rr_clean_path = rr_out_dir / "rr_preprocessed.csv"
                if rr_clean_path.exists():
                    rr_df = pd.read_csv(rr_clean_path)
                    print(f"  Loaded {rr_clean_path}")
                else:
                    preprocess_rr(
                        in_path=rr_path,
                        out_path=str(rr_clean_path),
                        time_col=rr_cfg.get("time_col", "time"),
                        rr_col=rr_cfg.get("rr_col", "rr"),
                    )
                    rr_df = pd.read_csv(rr_clean_path)
                    print(f"  Saved {rr_clean_path}")

        # ---------- MODALITY: EDA (optional) ----------
        has_eda = "eda" in dataset
        if has_eda:
            print("▶ EDA: preprocessing")
            eda_cfg = cfg["preprocessing"]["eda"]
            feat_cfg_eda = cfg["features"]["eda"]

            eda_path = dataset["eda"]["path"]
            fs_eda = float(dataset["eda"]["fs_out"])

            eda_out_dir = ds_out / "eda"
            eda_out_dir.mkdir(parents=True, exist_ok=True)

            eda_clean_path = eda_out_dir / "eda_preprocessed.csv"
            if eda_clean_path.exists():
                eda_df = pd.read_csv(eda_clean_path)
                print(f"  Loaded {eda_clean_path}")
            else:
                preprocess_eda(
                    in_path=eda_path,
                    out_path=str(eda_clean_path),
                    fs=fs_eda,
                    time_col=eda_cfg.get("time_col", "time"),
                    do_resample=bool(eda_cfg.get("do_resample", True)),
                )
                eda_df = pd.read_csv(eda_clean_path)
                print(f"  Saved {eda_clean_path}")

        # ---------- WINDOWS + FEATURES + QC + ALIGNMENT PER WINDOW LENGTH ----------
        for win_sec in win_lengths:
            win_sec = float(win_sec)
            print(f"\n▶ Window length: {win_sec}s | overlap={overlap}")

            # ---- IMU windows ----
            imu_win_path = imu_out_dir / f"imu_windows_{win_sec:.1f}s.csv"
            if imu_win_path.exists():
                imu_windows = pd.read_csv(imu_win_path)
            else:
                imu_windows = create_windows(df=imu_df, fs=fs_imu, win_sec=win_sec, overlap=overlap)
                imu_windows.to_csv(imu_win_path, index=False)

            # ---- IMU features (TOP only, no TIFEX) ----
            imu_feat_path = imu_out_dir / f"imu_features_{win_sec:.1f}s.csv"
            if imu_feat_path.exists():
                imu_feats = pd.read_csv(imu_feat_path)
            else:
                imu_feats = compute_top_imu_features_from_windows(
                    data=imu_df,
                    windows=imu_windows,
                    signal_cols=feat_cfg_imu["signals"],
                    quiet=True,
                )
                imu_feats = _ensure_modality_col(imu_feats, feat_cfg_imu.get("modality", "imu"))
                imu_feats.to_csv(imu_feat_path, index=False)

            # ---- IMU alignment ----
            adl_path = cfg["targets"]["imu"]["adl_path"]
            imu_aligned_path = imu_out_dir / f"imu_aligned_{win_sec:.1f}s.csv"
            if not imu_aligned_path.exists():
                run_alignment(
                    features_path=str(imu_feat_path),
                    windows_path=str(imu_win_path),
                    adl_path=adl_path,
                    out_path=str(imu_aligned_path),
                    target_type=target_type,
                    rr_path=Path(rr_path) if rr_path else None,
                    rmssd_windows_path=Path(rmssd_windows_path) if rmssd_windows_path else None,
                    rec_start=rec_start,
                    rec_end=rec_end,
                    adl_offset_hours=adl_offset_hours,
                )

            # ---- PPG variants windows/features/QC/alignment ----
            for ppg_key in ppg_variants.keys():
                ppg_cfg_local = cfg["preprocessing"][ppg_key]
                feat_cfg_ppg_local = cfg["features"][ppg_key]
                ppg_out_dir_local = ds_out / ppg_key
                ppg_data_local = ppg_variants[ppg_key]
                fs_ppg_local = float(ppg_data_local["fs_out"])
                
                ppg_clean_path_local = ppg_out_dir_local / f"{ppg_key}_preprocessed.csv"
                ppg_df_local = pd.read_csv(ppg_clean_path_local)
                
                ppg_win_path = ppg_out_dir_local / f"{ppg_key}_windows_{win_sec:.1f}s.csv"
                if ppg_win_path.exists():
                    ppg_windows = pd.read_csv(ppg_win_path)
                else:
                    ppg_windows = create_windows(df=ppg_df_local, fs=fs_ppg_local, win_sec=win_sec, overlap=overlap)
                    ppg_windows.to_csv(ppg_win_path, index=False)

                ppg_feat_path = ppg_out_dir_local / f"{ppg_key}_features_{win_sec:.1f}s.csv"
                if not ppg_feat_path.exists():
                    subprocess.check_call(
                        [
                            sys.executable,
                            "phases/phase3_features/ppg_features.py",
                            "--ppg",
                            str(ppg_clean_path_local),
                            "--windows",
                            str(ppg_win_path),
                            "--out",
                            str(ppg_feat_path),
                            "--fs",
                            str(fs_ppg_local),
                            "--time_col",
                            feat_cfg_ppg_local.get("time_col", "t_sec"),
                            "--signal_col",
                            feat_cfg_ppg_local.get("signal_col", "value"),
                            "--prefix",
                            feat_cfg_ppg_local.get("prefix", f"{ppg_key}_"),
                        ]
                    )
                    tmp = pd.read_csv(ppg_feat_path)
                    tmp = _ensure_modality_col(tmp, feat_cfg_ppg_local.get("modality", ppg_key))
                    tmp.to_csv(ppg_feat_path, index=False)

                ppg_aligned_path = ppg_out_dir_local / f"{ppg_key}_aligned_{win_sec:.1f}s.csv"
                if not ppg_aligned_path.exists():
                    run_alignment(
                        features_path=str(ppg_feat_path),
                        windows_path=str(ppg_win_path),
                        adl_path=adl_path,
                        out_path=str(ppg_aligned_path),
                        target_type=target_type,
                        rr_path=Path(rr_path) if rr_path else None,
                        rmssd_windows_path=Path(rmssd_windows_path) if rmssd_windows_path else None,
                        rec_start=rec_start,
                        rec_end=rec_end,
                        adl_offset_hours=adl_offset_hours,
                    )

            # ---- RR windows/features/QC/alignment (SKIPPED - non-uniform sampling) ----
            # RR data has event-based timestamps, not uniform grid. Skipping windowed features for now.
            # TODO: Implement RR aggregation or resampling strategy

            # ---- EDA windows/features/QC/alignment ----
            if has_eda:
                eda_win_path = eda_out_dir / f"eda_windows_{win_sec:.1f}s.csv"
                if eda_win_path.exists():
                    eda_windows = pd.read_csv(eda_win_path)
                else:
                    eda_windows = create_windows(df=eda_df, fs=fs_eda, win_sec=win_sec, overlap=overlap)
                    eda_windows.to_csv(eda_win_path, index=False)

                eda_feat_path = eda_out_dir / f"eda_features_{win_sec:.1f}s.csv"
                if not eda_feat_path.exists():
                    subprocess.check_call(
                        [
                            sys.executable,
                            "phases/phase3_features/eda_features.py",
                            "--eda",
                            str(eda_clean_path),
                            "--windows",
                            str(eda_win_path),
                            "--out",
                            str(eda_feat_path),
                            "--fs",
                            str(fs_eda),
                            "--time_col",
                            feat_cfg_eda.get("time_col", "t_sec"),
                            "--cc_col",
                            feat_cfg_eda.get("cc_col", "eda_cc"),
                            "--stress_col",
                            feat_cfg_eda.get("stress_col", "eda_stress_skin"),
                            "--prefix",
                            feat_cfg_eda.get("prefix", "eda_"),
                        ]
                    )
                    tmp = pd.read_csv(eda_feat_path)
                    tmp = _ensure_modality_col(tmp, feat_cfg_eda.get("modality", "eda"))
                    tmp.to_csv(eda_feat_path, index=False)

                eda_aligned_path = eda_out_dir / f"eda_aligned_{win_sec:.1f}s.csv"
                if not eda_aligned_path.exists():
                    run_alignment(
                        features_path=str(eda_feat_path),
                        windows_path=str(eda_win_path),
                        adl_path=adl_path,
                        out_path=str(eda_aligned_path),
                        target_type=target_type,
                        rr_path=Path(rr_path) if rr_path else None,
                        rmssd_windows_path=Path(rmssd_windows_path) if rmssd_windows_path else None,
                        rec_start=rec_start,
                        rec_end=rec_end,
                        adl_offset_hours=adl_offset_hours,
                    )

        # ---------- FUSION ----------
        print("\n▶ Fusion step")
        if cfg.get("fusion", None) is not None:
            print("  Running fusion...")
            run_fusion(config_path=config_path)
            print("  ✓ Fusion complete")
        else:
            print("  (Fusion not configured in config)")
        
        # ---------- FUSED FEATURES ALIGNMENT WITH HRV RECOVERY ----------
        print("\n▶ Aligning fused features with HRV Recovery Rate labels")
        adl_path = cfg["targets"]["imu"]["adl_path"]
        
        for win_sec in win_lengths:
            win_sec = float(win_sec)
            
            fused_feat_path = ds_out / f"fused_features_{win_sec:.1f}s.csv"
            imu_win_path = ds_out / "imu_bioz" / f"imu_windows_{win_sec:.1f}s.csv"
            ppg_clean_path = ds_out / "ppg_green" / "ppg_green_preprocessed.csv"  # Use green PPG for HRV
            fused_aligned_path = ds_out / f"fused_aligned_{win_sec:.1f}s.csv"
            
            if fused_feat_path.exists() and not fused_aligned_path.exists():
                print(f"  Computing HRV recovery rates and aligning fused features ({win_sec:.1f}s windows)...")
                # Load windows and ADL data
                windows_df = pd.read_csv(imu_win_path)
                
                # Add window_id if missing (needed for merge)
                if 'window_id' not in windows_df.columns:
                    windows_df['window_id'] = windows_df.index
                
                adl_df = parse_adl_intervals(adl_path)

                if adl_offset_hours != 0:
                    delta = adl_offset_hours * 3600.0
                    adl_df["t_start"] = adl_df["t_start"] + delta
                    adl_df["t_end"] = adl_df["t_end"] + delta
                    print(f"  Shifted ADL intervals by {adl_offset_hours} h ({delta:.0f} s)")

                adl_t_start = adl_df["t_start"].min()
                adl_t_end = adl_df["t_end"].max()
                recovery_buffer = 300  # 5 minutes for HRV recovery phase
                windows_before = len(windows_df)
                windows_df = windows_df[
                    (windows_df["t_center"] >= adl_t_start) &
                    (windows_df["t_center"] <= adl_t_end + recovery_buffer)
                ].copy()

                print(f"  ADL recording time: {adl_t_start:.1f} to {adl_t_end:.1f} (+{recovery_buffer}s recovery buffer)")
                print(f"  Windows before filtering: {windows_before}")
                print(f"  Windows in ADL time range: {len(windows_df)}")

                # Use RR-based HRV recovery computation
                if not rr_path or not Path(rr_path).exists():
                    raise ValueError(f"RR path required for HRV recovery computation: {rr_path}")
                
                aligned_df = align_windows_to_hrv_recovery_rr(
                    windows=windows_df,
                    intervals=adl_df,
                    rr_path=rr_path
                )
                
                # Merge HRV recovery labels with fused features
                fused_df = pd.read_csv(fused_feat_path)
                
                # Build merge columns based on what's available
                merge_keys = ['window_id'] if 'window_id' in aligned_df.columns else ['start_idx', 'end_idx']
                hrv_cols = [c for c in ['hrv_recovery_rate', 'activity_borg'] if c in aligned_df.columns]
                merge_cols = merge_keys + hrv_cols
                
                merged_df = fused_df.merge(
                    aligned_df[merge_cols],
                    on=merge_keys,
                    how='inner'  # Only keep windows with HRV labels
                )
                merged_df.to_csv(fused_aligned_path, index=False)
                print(f"  ✓ HRV alignment complete. {len(merged_df)} windows with recovery rates")
        
        # ---------- FEATURE SELECTION + QC ----------
        print("\n▶ Feature selection with correlation pruning + quality checks")
        for win_sec in win_lengths:
            win_sec = float(win_sec)
            fused_aligned_path = ds_out / f"fused_aligned_{win_sec:.1f}s.csv"
            feature_qc_dir = ds_out / "feature_selection_qc" / f"qc_{win_sec:.1f}s"
            
            if fused_aligned_path.exists():
                print(f"  Running feature selection ({win_sec:.1f}s windows)...")
                run_feature_selection(
                    fused_aligned_path=str(fused_aligned_path),
                    output_dir=str(feature_qc_dir),
                    win_sec=win_sec
                )
                print(f"  ✓ Feature selection complete ({win_sec:.1f}s)")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    run_pipeline(config_path)
