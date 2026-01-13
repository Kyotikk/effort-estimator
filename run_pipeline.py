import yaml
from pathlib import Path
import pandas as pd


from preprocessing.imu import preprocess_imu
from windowing.windows import create_windows
from features.tifex import run_tifex

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---- Pipeline ----
def run_pipeline(config_path: str) -> None:
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

        imu_cfg = cfg["preprocessing"]["imu"]
        imu_path = dataset["imu"]["path"]
        fs_out = dataset["imu"]["fs_out"]

        imu_df = preprocess_imu(
            path=imu_path,
            fs_out=fs_out,
            noise_cutoff=imu_cfg["noise_cutoff"],
            gravity_cutoff=imu_cfg["gravity_cutoff"],
        )

        imu_out_dir = output_root / name / "imu"
        imu_out_dir.mkdir(parents=True, exist_ok=True)

        imu_clean_path = imu_out_dir / "imu_preprocessed.csv"
        imu_df.to_csv(imu_clean_path, index=False)
        print(f"  Saved preprocessed IMU to {imu_clean_path}")

        # -------------------------
        # 2-3. Windowing + Feature extraction
        # -------------------------
        print("▶ Creating windows and extracting features")
        feat_cfg = cfg["features"]["imu"]

        for win_sec in cfg["windowing"]["window_lengths_sec"]:
            print(f"  ▶ Window length: {win_sec}s")

            windows_df = create_windows(
                df=imu_df,
                fs=fs_out,
                win_sec=win_sec,
                overlap=cfg["windowing"]["overlap"],
            )
        
            print(f"    {len(windows_df)} windows created")

            win_path = imu_out_dir / f"windows_{win_sec:.1f}s.csv"
            windows_df.to_csv(win_path, index=False)
            print(f"  Saved windows to {win_path}")

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

        # print(feats_df.shape[0], "feature rows created with", feats_df.shape[1], "features.")
            feat_path = imu_out_dir / f"features_{win_sec:.1f}s.csv"
            feats_df.to_csv(feat_path, index=False)

        print(f"  Saved features to {feat_path}")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline("config/pipeline.yaml")
