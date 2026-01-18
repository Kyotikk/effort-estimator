#!/usr/bin/env python3
import os
import subprocess
import pandas as pd

from preprocessing.ppg import preprocess_ppg
from windowing.windows import create_windows


# -----------------
# CONFIG (edit here)
# -----------------
FS_PPG  = 32.0

WIN_SEC = 10.0          # 10s windows
OVERLAP = 0.5           # 50% overlap (fraction in [0,1))

RAW_PPG = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_ppg2_green_6/2025-12-04.csv.gz"
OUT_DIR = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_ppg2_green_6"

DATE_STEM = "2025-12-04"


# -----------------
# OUTPUT PATHS
# -----------------
PPG_PREPROCESSED = os.path.join(OUT_DIR, f"{DATE_STEM}_ppg_preprocessed.csv")

PPG_WINDOWS  = os.path.join(
    OUT_DIR, f"{DATE_STEM}_ppg_windows_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol.csv"
)
PPG_FEATURES = os.path.join(
    OUT_DIR, f"{DATE_STEM}_ppg_features_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol.csv"
)

QC_OUT_DIR = os.path.join(
    "/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis",
    f"quality_ppg_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol",
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(QC_OUT_DIR, exist_ok=True)

    # 1) preprocess
    preprocess_ppg(
        in_path=RAW_PPG,
        out_path=PPG_PREPROCESSED,
        fs=FS_PPG,
        time_col="time",
        metric_id="0x7e",
        led_pd_pos=6,
        led=None,
        do_resample=True,
    )

    # 2) windows (10s, 50% overlap)
    df = pd.read_csv(PPG_PREPROCESSED)
    win = create_windows(df, fs=FS_PPG, win_sec=WIN_SEC, overlap=OVERLAP)
    win.to_csv(PPG_WINDOWS, index=False)
    print(f"✓ Wrote {len(win)} windows → {PPG_WINDOWS}")

    # 3) features
    subprocess.check_call([
        "python", "features/ppg_features.py",
        "--ppg", PPG_PREPROCESSED,
        "--windows", PPG_WINDOWS,
        "--out", PPG_FEATURES,
        "--fs", str(FS_PPG),
        "--time_col", "t_sec",
        "--signal_col", "value",
        "--prefix", "ppg_",
    ])

    # 4) quality check
    subprocess.check_call([
        "python", "windowing/feature_quality_check_any.py",
        "--in_csv", PPG_FEATURES,
        "--out_dir", QC_OUT_DIR,
    ])

    print("DONE")


if __name__ == "__main__":
    main()
