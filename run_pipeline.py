import numpy as np
import pandas as pd
import yaml
from preprocessing.imu import preprocess_imu
from windowing.windows import create_windows

def validate_preprocessing(df, fs):
    dt = np.diff(df["t_sec"].values)
    print("Unique dt:", np.unique(np.round(dt, 6)))

    assert np.allclose(dt, 1.0 / fs, rtol=1e-3), "Sampling rate mismatch"

def validate_windows(df, windows, fs, win_sec):
    expected_len = int(round(fs * win_sec))

    print("Window sample sizes:", windows["n_samples"].unique())
    assert windows["n_samples"].nunique() == 1
    assert windows["n_samples"].iloc[0] == expected_len

    print("Max end_idx:", windows["end_idx"].max(), "Data length:", len(df))
    assert windows["end_idx"].max() <= len(df)

fs = 32
win_sec = 5.0
overlap = 0.7
path = "C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_healthy_3\\corsano_bioz_acc\\2025-12-04.csv.gz"

df = preprocess_imu(path, fs_out=fs)
print("Preprocessed IMU data shape:", df.shape)

# --- Validation: sampling ---
validate_preprocessing(df, fs)

# --- Windowing ---
windows = create_windows(df, fs, win_sec, overlap)

# --- Validation: windows ---
validate_windows(df, windows, fs, win_sec)



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

