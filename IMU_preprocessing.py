import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# Helpers
# ----- Butterworth filter functions -----
def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


# ----- Load IMU data -----
def load_imu_csv(path):
    df = pd.read_csv(path, compression="gzip")
    # print(df.columns)
    df = df[["time", "accX", "accY", "accZ"]]
    df[["accX", "accY", "accZ"]] = df[["accX", "accY", "accZ"]].apply(
        pd.to_numeric, errors="coerce"
    )

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    
    print(f"Loaded IMU data from {path} with {len(df)} samples.")
    return df

# ----- Optional resampling -----
def resample_imu(df, fs=32):
    period_ms = int(1000 / fs)

    df_resampled = (
        df.resample(f"{period_ms}ms")
          .mean()
          .interpolate(method="linear")
    )

    print(f"Resampled IMU data to {fs}Hz with {len(df_resampled)} samples.")
    return df_resampled

# ----- Noise filtering at 5Hz -----
def lowpass_filter_imu(df, fs=32, cutoff=5.0):
    df_filt = df.copy()

    for axis in ["accX", "accY", "accZ"]:
        df_filt[axis] = butter_lowpass(
            df_filt[axis].values, cutoff, fs
        )

    print(f"Applied lowpass filter at {cutoff}Hz.")
    return df_filt

# ----- Gravity estimation and removal -----
def remove_gravity(df, fs=32, gravity_cutoff=0.3):
    df_out = df.copy()

    for axis in ["accX", "accY", "accZ"]:
        gravity = butter_lowpass(
            df[axis].values, gravity_cutoff, fs
        )
        df_out[f"{axis}_dyn"] = df[axis] - gravity
        df_out[f"{axis}_grav"] = gravity

    print(f"Removed gravity component using lowpass filter at {gravity_cutoff}Hz.")
    return df_out

# ---- Signal Vector Magnitude computation -----
# def compute_svm(df):
#     df["svm_raw"] = np.sqrt(
#         df["accX"]**2 + df["accY"]**2 + df["accZ"]**2
#     )

#     df["svm_dyn"] = np.sqrt(
#         df["accX_dyn"]**2 +
#         df["accY_dyn"]**2 +
#         df["accZ_dyn"]**2
#     )

#     print("Computed Signal Vector Magnitude (SVM).")
#     return df

# ----- Z-score normalisation -----
def zscore_normalise(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print("Applied Z-score normalisation.")
    return df

# ----- Full preprocessing pipeline -----
def preprocess_imu(
    path,
    fs=32,
    noise_cutoff=5.0,
    gravity_cutoff=0.3
):
    # Load
    df = load_imu_csv(path)

    # Resample
    df = resample_imu(df, fs)

    # Noise filtering
    df = lowpass_filter_imu(df, fs, noise_cutoff)

    # Gravity removal
    df = remove_gravity(df, fs, gravity_cutoff)

    # SVM
    # df = compute_svm(df)

    # Normalisation
    norm_cols = [
        "accX_dyn", "accY_dyn", "accZ_dyn" #,
    #     "svm_dyn", "svm_raw"
    ]
    df = zscore_normalise(df, norm_cols)

    return df


import argparse
import os
import pandas as pd


def windowing(args):
    

    df = pd.read_csv(args.input)

    required = ["t_sec", "acc_x", "acc_y", "acc_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    fs = int(args.fs)
    win_len = int(round(fs * float(args.win_sec)))
    if win_len < 2:
        raise ValueError("win_len < 2. Increase --win_sec or --fs.")

    if not (0.0 <= float(args.overlap) < 1.0):
        raise ValueError("--overlap must be in [0,1).")

    hop = int(round(win_len * (1.0 - float(args.overlap))))
    hop = max(1, hop)

    n = len(df)
    rows = []
    for start in range(0, n - win_len + 1, hop):
        end = start + win_len  # python slicing end-exclusive
        rows.append({
            "start_idx": start,
            "end_idx": end,
            "t_start": float(df["t_sec"].iloc[start]),
            "t_end": float(df["t_sec"].iloc[end - 1]),
            "n_samples": win_len
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    expected = 0
    if n >= win_len:
        expected = 1 + (n - win_len) // hop

    print(f"Wrote {len(out_df)} windows (expected ~{expected}) to {args.out}")
    print("Example:", out_df.head(3).to_string(index=False))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw CSV with t_sec, acc_x, acc_y, acc_z")
    ap.add_argument("--out", required=True, help="Output CSV of window indices")
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--win_sec", type=float, required=True)
    ap.add_argument("--overlap", type=float, required=True)
    args = ap.parse_args()
    df_windowed = windowing(args)
    print(df_windowed.head())
    
    # Example usage
    path = "/Users/pascalschlegel/Sensor-Data-Parsing/data/interim/parsingsim3/sim_healthy3/corsano_bioz_acc/2025-12-04.csv.gz"
    df_processed = preprocess_imu(path)

    print(df_processed.head())
    print(df_processed)

    


if __name__ == "__main__":
    main()


