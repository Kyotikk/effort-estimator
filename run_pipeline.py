import yaml
from pathlib import Path
import pandas as pd


from preprocessing.imu import preprocess_imu
from windowing.windows import create_windows
from features.tifex import run_tifex
from ml.targets.run_target_alignment import run_alignment
from ml.run_fusion import main as run_fusion

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---- Pipeline ----
def run_pipeline(config_path: str) -> None:
    """
    Docstring for run_pipeline
    
    :param config_path: Config file path
    :type config_path: str
    
    Runs the end-to-end data processing pipeline including:
    1. IMU Preprocessing
    2. Windowing
    3. Feature Extraction
    4. Target Alignment
    5. Fusion (if applicable)
    """

    cfg = load_config(config_path)

    output_root = Path(cfg["project"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in cfg["datasets"]:
        name = dataset["name"]
        print(f"\n=== Processing dataset: {name} ===")
        # -------------------------
        # 1. IMU Preprocessing
        # -------------------------
        print("▶ Preprocessing IMU")

        modality = "imu"  # hardcoded for now
        imu_cfg = cfg["preprocessing"]["imu"]
        imu_path = dataset["imu_bioz"]["path"]
        fs_out = dataset["imu_bioz"]["fs_out"]

        imu_out_dir = output_root / name / "imu_bioz"
        imu_out_dir.mkdir(parents=True, exist_ok=True)

        imu_clean_path = imu_out_dir / "imu_preprocessed.csv"

        if imu_clean_path.exists():
            print(f"  Preprocessed IMU already exists at {imu_clean_path}, loading...")
            imu_df = pd.read_csv(imu_clean_path)
        else:
            imu_df = preprocess_imu(
                path=imu_path,
                fs_out=fs_out,
                noise_cutoff=imu_cfg["noise_cutoff"],
                gravity_cutoff=imu_cfg["gravity_cutoff"],
            )

            imu_df.to_csv(imu_clean_path, index=False)
            print(f"  Saved preprocessed IMU to {imu_clean_path}")
        
                # -------------------------
        # 2-3. Windowing + Feature extraction
        # -------------------------
        print("▶ Creating windows and extracting features")
        feat_cfg = cfg["features"]["imu"]


        for win_sec in cfg["windowing"]["window_lengths_sec"]:

            win_path = imu_out_dir / f"{modality}_windows_{win_sec:.1f}s.csv"
            if win_path.exists():
                print(f"  Windows for {win_sec}s already exist at {win_path}, loading...")
                windows_df = pd.read_csv(win_path)
            else:
                print(f"  ▶ Window length: {win_sec}s")

                windows_df = create_windows(
                    df=imu_df,
                    fs=fs_out,
                    win_sec=win_sec,
                    overlap=cfg["windowing"]["overlap"],
                )
                print(f"    {len(windows_df)} windows created")
               
                windows_df.to_csv(win_path, index=False)
                print(f"  Saved windows to {win_path}")

            
            feat_path = imu_out_dir / f"{modality}_features_{win_sec:.1f}s.csv"
            if feat_path.exists():
                print(f"  Features for {win_sec}s already exist at {feat_path}, loading...")
                feats_df = pd.read_csv(feat_path)
            else:
                # ---- Feature extraction ----
                print(f"  ▶ Feature extraction with Tifex")
                feats_df = run_tifex(
                    data=imu_df,
                    windows=windows_df,
                    fs=fs_out,
                    signal_cols=feat_cfg["signals"],
                    feature_set=feat_cfg["feature_set"],
                    safe=feat_cfg["safe"],
                    njobs=feat_cfg["njobs"],
                )

                print(feats_df.head())
                print("Rows:", len(feats_df))
                print("Feature columns:", feats_df.filter(like="__").shape[1])
                
                # Add modality metadata column (from feature config if provided, otherwise 'imu')
                modality = feat_cfg.get("modality", "unknown")
                if "modality" not in feats_df.columns:
                    feats_df["modality"] = modality

                print(
                    f"  ✓ IMU | {win_sec}s | "
                    f"{len(windows_df)} windows | "
                    f"{feats_df.shape[1]} features | "
                    f"modality={modality}"
                )
                
                feats_df.to_csv(feat_path, index=False)
                print(f"  Saved features to {feat_path}")

            # -------------------------
            # 4. Target Alignment
            # -------------------------
            print("▶ Aligning targets")
            adl_path = cfg["targets"]["imu"]["adl_path"]
            adl_out_path = imu_out_dir / f"{modality}_aligned_{win_sec:.1f}s.csv"
            if adl_out_path.exists():
                print(f"  Aligned targets for {win_sec}s already exist at {adl_out_path}, skipping...")
                aligned_df = pd.read_csv(adl_out_path)
            else:
                run_alignment(
                    features_path=feat_path,
                    windows_path=win_path,
                    adl_path=adl_path,
                    out_path=adl_out_path,
                )
                aligned_df = pd.read_csv(adl_out_path)
            
        # -------------------------
        # 5. Fusion (if applicable)
        # ------------------------
        print("▶ Fusion step (if applicable)")
        fus_out_path = imu_out_dir / f"fused_{win_sec:.1f}s.csv"
        if fus_out_path.exists():
            print(f"  Fused features for {win_sec}s already exist at {fus_out_path}, skipping...")
        else:
            fusion_cfg = cfg.get("fusion", None)
            if fusion_cfg is not None:
                run_fusion(config_path=config_path)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline("config/pipeline.yaml")
