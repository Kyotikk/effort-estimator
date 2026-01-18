#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd

from preprocessing.eda import preprocess_eda
from windowing.windows import create_windows


# -----------------
# CONFIG
# -----------------
FS_EDA  = 32.0
WIN_SEC = 10.0
OVERLAP = 0.5

BASE = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3"
RAW_EDA = os.path.join(BASE, "corsano_bioz_emography", "2025-12-04.csv.gz")
DATE_STEM = "2025-12-04"

OUT_EDA = os.path.join(BASE, f"{DATE_STEM}_eda_preprocessed.csv")
EDA_WINDOWS = os.path.join(BASE, f"{DATE_STEM}_eda_windows_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol.csv")
EDA_FEATURES = os.path.join(BASE, f"{DATE_STEM}_eda_features_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol.csv")

QC_OUT_DIR = os.path.join(
    "/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis",
    f"quality_eda_{int(WIN_SEC)}s_{int(OVERLAP*100)}ol",
)


def main():
    os.makedirs(os.path.dirname(OUT_EDA), exist_ok=True)
    os.makedirs(QC_OUT_DIR, exist_ok=True)

    # 1) preprocess (resample to 32Hz)
    preprocess_eda(
        in_path=RAW_EDA,
        out_path=OUT_EDA,
        fs=FS_EDA,
        time_col="time",
        do_resample=True,
        # keep_cols default matches your file; you can shrink later if you want
    )

    # 2) windows (10s, 50% overlap)
    df = pd.read_csv(OUT_EDA)
    win = create_windows(df, fs=FS_EDA, win_sec=WIN_SEC, overlap=OVERLAP)
    win.to_csv(EDA_WINDOWS, index=False)
    print(f"✓ Wrote {len(win)} windows → {EDA_WINDOWS}")

    # 3) features (use venv python reliably)
    subprocess.check_call([
        sys.executable, "features/eda_features.py",
        "--eda", OUT_EDA,
        "--windows", EDA_WINDOWS,
        "--out", EDA_FEATURES,
        "--fs", str(FS_EDA),
        "--time_col", "t_sec",
        "--cc_col", "eda_cc",
        "--stress_col", "eda_stress_skin",
        "--prefix", "eda_",
    ])

    # 4) quality check
    subprocess.check_call([
        sys.executable, "windowing/feature_quality_check_any.py",
        "--in_csv", EDA_FEATURES,
        "--out_dir", QC_OUT_DIR,
    ])

    print("DONE")


if __name__ == "__main__":
    main()
