#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

WINDOW_DIR = "data/feature_extraction/windows"
OUT_DIR    = "data/feature_extraction/manual_out"

WINDOWS_S  = [2, 5, 10]

# Your dynamic accel columns from preprocessing
COLUMNS    = ["accX_dyn", "accY_dyn", "accZ_dyn"]

# Optional metadata to carry through if present in windows CSV
META_COLS  = ["timestamp_start", "timestamp_end", "timestamp_center"]

FS         = 32.0


def iqr(x):
    return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)


def tkeo(x):
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def median_frequency_from_psd(x, fs):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 4:
        return np.nan
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = np.sum(psd)
    if total <= 0 or not np.isfinite(total):
        return np.nan
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, 0.5 * total)
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def spectral_entropy_from_psd(x, fs):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 4:
        return np.nan
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / n
    s = np.sum(psd)
    if s <= 0 or not np.isfinite(s):
        return np.nan
    p = psd / s
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(psd)))


def spectral_slope_logarithmic(x, fs):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return np.nan
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    freqs = freqs[1:]
    psd = psd[1:]
    mask = (freqs > 0) & (psd > 0) & np.isfinite(psd)
    freqs = freqs[mask]
    psd = psd[mask]
    if len(freqs) < 3:
        return np.nan
    lx = np.log(freqs)
    ly = np.log(psd)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope)


def compute_features_for_window(X, fs):
    ax, ay, az = X[:, 0], X[:, 1], X[:, 2]
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    feats = {}

    # amplitude / magnitude
    feats["acc_mag__mean"] = float(np.mean(acc_mag))
    feats["acc_mag__std"]  = float(np.std(acc_mag, ddof=0))
    feats["acc_mag__min"]  = float(np.min(acc_mag))
    feats["acc_mag__max"]  = float(np.max(acc_mag))
    feats["acc_mag__iqr"]  = float(iqr(acc_mag))

    # directional variability
    feats["acc_x__std"] = float(np.std(ax, ddof=0))
    feats["acc_y__std"] = float(np.std(ay, ddof=0))
    feats["acc_z__std"] = float(np.std(az, ddof=0))

    # periodicity + spectrum
    feats["acc_mag__median_frequency"] = median_frequency_from_psd(acc_mag, fs)
    feats["acc_mag__spectral_entropy"] = spectral_entropy_from_psd(acc_mag, fs)
    feats["acc_mag__spectral_slope_logarithmic"] = spectral_slope_logarithmic(acc_mag, fs)

    # bursts
    tk = tkeo(acc_mag)
    feats["acc_mag__tkeo_mean"] = float(np.mean(tk)) if len(tk) else np.nan

    # extra “safe” stats
    feats["acc_mag__median"] = float(np.median(acc_mag))
    feats["acc_mag__mad"] = float(np.median(np.abs(acc_mag - np.median(acc_mag))))
    feats["acc_mag__range"] = float(np.max(acc_mag) - np.min(acc_mag))
    feats["acc_mag__p90"] = float(np.nanpercentile(acc_mag, 90))
    feats["acc_mag__p10"] = float(np.nanpercentile(acc_mag, 10))
    feats["acc_mag__rms"] = float(np.sqrt(np.mean(acc_mag**2)))

    # axis correlations (coordination)
    feats["acc_xy__corr"] = float(np.corrcoef(ax, ay)[0, 1]) if np.std(ax) > 0 and np.std(ay) > 0 else np.nan
    feats["acc_xz__corr"] = float(np.corrcoef(ax, az)[0, 1]) if np.std(ax) > 0 and np.std(az) > 0 else np.nan
    feats["acc_yz__corr"] = float(np.corrcoef(ay, az)[0, 1]) if np.std(ay) > 0 and np.std(az) > 0 else np.nan

    return feats


def extract_for_file(in_csv, out_csv):
    wdf = pd.read_csv(in_csv)

    needed = ["window_id", "sample"] + COLUMNS
    missing = [c for c in needed if c not in wdf.columns]
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

    # determine which meta cols we can carry
    meta_present = [c for c in META_COLS if c in wdf.columns]

    wdf[COLUMNS] = wdf[COLUMNS].apply(pd.to_numeric, errors="coerce")

    rows = []
    for wid, g in wdf.groupby("window_id", sort=True):
        g = g.sort_values("sample")

        # drop rows with NaNs in required signal columns
        Xdf = g[COLUMNS].dropna()
        if len(Xdf) < 8:
            continue

        X = Xdf.to_numpy(dtype=float)
        feats = compute_features_for_window(X, FS)

        # window metadata
        feats["window_id"] = int(wid)
        for c in meta_present:
            # window-level constant per group; just take first
            feats[c] = g[c].iloc[0]

        rows.append(feats)

    out = pd.DataFrame(rows)

    if len(out):
        # keep consistent column order: window_id + timestamps + features
        front = ["window_id"] + [c for c in META_COLS if c in out.columns]
        rest = [c for c in out.columns if c not in set(front)]
        out = out[front + rest]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} | windows={len(out)} | features={out.shape[1] - (1 + len([c for c in META_COLS if c in out.columns])) if len(out) else 0}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for win_s in WINDOWS_S:
        in_csv  = os.path.join(WINDOW_DIR, f"windows_{win_s}s_ol50.csv")
        out_csv = os.path.join(OUT_DIR, f"manual_features_{win_s}s_ol50.csv")
        if not os.path.exists(in_csv):
            raise FileNotFoundError(in_csv)
        extract_for_file(in_csv, out_csv)


if __name__ == "__main__":
    main()
