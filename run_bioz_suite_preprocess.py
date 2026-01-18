#!/usr/bin/env python3
import os

from preprocessing.eda import preprocess_eda
from preprocessing.bioz import preprocess_bioz
from preprocessing.temp import preprocess_temp
from preprocessing.rr import preprocess_rr


# -----------------
# INPUT PATHS (your exact folders)
# -----------------
BASE = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3"

PATH_BIOZ = os.path.join(BASE, "corsano_bioz_bioz", "2025-12-04.csv.gz")
PATH_EDA  = os.path.join(BASE, "corsano_bioz_emography", "2025-12-04.csv.gz")
PATH_TEMP = os.path.join(BASE, "corsano_bioz_temperature", "2025-12-04.csv.gz")
PATH_RR   = os.path.join(BASE, "corsano_bioz_rr_interval", "2025-11-19.csv.gz")  # your example

OUT_DIR = BASE  # keep outputs next to raw folders? (or make a dedicated output folder)


# -----------------
# OUTPUTS
# -----------------
OUT_BIOZ = os.path.join(OUT_DIR, "2025-12-04_bioz_preprocessed.csv")
OUT_EDA  = os.path.join(OUT_DIR, "2025-12-04_eda_preprocessed.csv")
OUT_TEMP = os.path.join(OUT_DIR, "2025-12-04_temp_preprocessed.csv")
OUT_RR   = os.path.join(OUT_DIR, "2025-11-19_rr_preprocessed.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # choose conservative fs guesses; adjust later if needed
    preprocess_bioz(PATH_BIOZ, OUT_BIOZ, fs=32.0, time_col="time", do_resample=True)
    preprocess_eda (PATH_EDA,  OUT_EDA,  fs=32.0, time_col="time", do_resample=True)
    preprocess_temp(PATH_TEMP, OUT_TEMP, time_col="time", do_resample=False)


    # RR: event-based, no resample
    preprocess_rr(PATH_RR, OUT_RR, time_col="time", rr_col="rr")

    print("DONE")


if __name__ == "__main__":
    main()
