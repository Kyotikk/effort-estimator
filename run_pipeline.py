import os
import pandas as pd

from preprocessing.imu import preprocess_imu
from windowing.windows import create_windows
from features.tifex import main as run_tifex_cli

# Temporary config for testing
CONFIG = {
    "imu": {
        "fs": 32,
        "noise_cutoff": 5.0,
        "gravity_cutoff": 0.3,
        "signals_for_features": [
            "acc_x_dyn",
            "acc_y_dyn",
            "acc_z_dyn",
        ],
    },
    "window": {
        "length_sec": 5.0,
        "overlap": 0.7,
    },
    "tifex": {
        "features": "stat",
        "safe": True,
        "njobs": 1,
    },
}

# ---- Paths ----
# Need to iterate through multiple files later
RAW_IMU = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_healthy_3\\corsano_bioz_acc\\2025-12-04.csv.gz"

INTERIM_DIR = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\imu"
FEATURE_DIR = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\features\\imu"

IMU_PREPROCESSED = os.path.join(INTERIM_DIR, "imu_preprocessed.csv")
IMU_WINDOWS = os.path.join(INTERIM_DIR, "imu_windows.csv")
IMU_FEATURES = os.path.join(FEATURE_DIR, "imu_features.csv")

# path = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_healthy_3\\corsano_bioz_acc\\2025-12-04.csv.gz"

# ---- Pipeline ----
# include validation?
def run_imu_pipeline() -> None:
    os.makedirs(INTERIM_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # -------------------------
    # Preprocessing
    # -------------------------
    print("▶ Preprocessing IMU")

    df_imu = preprocess_imu(
        path=RAW_IMU,
        fs_out=CONFIG["imu"]["fs"],
        noise_cutoff=CONFIG["imu"]["noise_cutoff"],
        gravity_cutoff=CONFIG["imu"]["gravity_cutoff"],
    )

    df_imu.to_csv(IMU_PREPROCESSED, index=False)
    print(f"  Saved preprocessed IMU to {IMU_PREPROCESSED}")

    # -------------------------
    # Windowing
    # -------------------------
    print("▶ Creating windows")

    windows = create_windows(
        df=df_imu,
        fs=CONFIG["imu"]["fs"],
        win_sec=CONFIG["window"]["length_sec"],
        overlap=CONFIG["window"]["overlap"],
    )

    windows.to_csv(IMU_WINDOWS, index=False)
    print(f"  Saved windows to {IMU_WINDOWS}")

    # -------------------------
    # Feature extraction (TIFEX)
    # -------------------------
    print("▶ Extracting features with TIFEX")

    # We invoke tifex via its CLI-compatible main
    import sys
    sys.argv = [
        "tifex.py",
        "--input", IMU_PREPROCESSED,
        "--windows", IMU_WINDOWS,
        "--out", IMU_FEATURES,
        "--fs", str(CONFIG["imu"]["fs"]),
        "--signals", ",".join(CONFIG["imu"]["signals_for_features"]),
        "--features", CONFIG["tifex"]["features"],
        "--njobs", str(CONFIG["tifex"]["njobs"]),
    ]

    if CONFIG["tifex"]["safe"]:
        sys.argv.append("--safe")

    run_tifex_cli()

    print(f"  Saved IMU features to {IMU_FEATURES}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_imu_pipeline()
    

# def validate_preprocessing(df, fs):
#     dt = np.diff(df["t_sec"].values)
#     print("Unique dt:", np.unique(np.round(dt, 6)))

#     assert np.allclose(dt, 1.0 / fs, rtol=1e-3), "Sampling rate mismatch"

# def validate_windows(df, windows, fs, win_sec):
#     expected_len = int(round(fs * win_sec))

#     print("Window sample sizes:", windows["n_samples"].unique())
#     assert windows["n_samples"].nunique() == 1
#     assert windows["n_samples"].iloc[0] == expected_len

#     print("Max end_idx:", windows["end_idx"].max(), "Data length:", len(df))
#     assert windows["end_idx"].max() <= len(df)


# df = preprocess_imu(path, fs_out=fs)
# print("Preprocessed IMU data shape:", df.shape)

# # --- Validation: sampling ---
# validate_preprocessing(df, fs)

# # --- Windowing ---
# windows = create_windows(df, fs, win_sec, overlap)

# # --- Validation: windows ---
# validate_windows(df, windows, fs, win_sec)



# signals = config["imu"]["signals"]["dynamic"]

# if config["imu"]["signals"]["magnitude"]["enabled"]:
#     add_magnitude()

# def run_imu_pipeline(cfg):
#     df = preprocess_imu(
#         path=cfg["input"],
#         fs_out=cfg["imu"]["fs"]
#     )

#     df.to_csv(cfg["interim_imu"], index=False)

#     create_windows(
#         input_csv=cfg["interim_imu"],
#         fs=cfg["imu"]["fs"],
#         win_sec=cfg["window"]["length"],
#         overlap=cfg["window"]["overlap"],
#         out_csv=cfg["windows_imu"]
#     )

#     run_tifex(
#         input_csv=cfg["interim_imu"],
#         windows_csv=cfg["windows_imu"],
#         signals=cfg["imu"]["signals"]["dynamic"],
#         fs=cfg["imu"]["fs"],
#         out_csv=cfg["features_imu"]
#     )

