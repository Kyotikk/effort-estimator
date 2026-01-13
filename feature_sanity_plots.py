#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_YAML = "config/selected_features.yaml"

IN_CSV = "data/feature_extraction/manual_out/manual_features_5s_ol50.csv"
OUT_DIR = "data/feature_extraction/analysis/sanity_plots_5s_ol50"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(FEATURES_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    feats = cfg["features"]

    df = pd.read_csv(IN_CSV)

    # Basic integrity
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {IN_CSV}: {missing}")

    # If you have window_id, keep it around for ordering
    idx_col = "window_id" if "window_id" in df.columns else None

    # Summary table
    X = df[feats].copy()
    summary = pd.DataFrame({
        "n": X.shape[0],
        "nan_frac": X.isna().mean(),
        "mean": X.mean(numeric_only=True),
        "std": X.std(numeric_only=True),
        "min": X.min(numeric_only=True),
        "max": X.max(numeric_only=True),
    })
    summary.to_csv(os.path.join(OUT_DIR, "feature_summary.csv"))
    print(f"Wrote: {os.path.join(OUT_DIR, 'feature_summary.csv')}")

    # Plots per feature
    for feat in feats:
        s = df[feat]

        # Histogram
        plt.figure()
        s.dropna().hist(bins=50)
        plt.title(f"{feat} — histogram")
        plt.xlabel(feat)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{feat}__hist.png"), dpi=150)
        plt.close()

        # Time/sequence plot (window order)
        plt.figure()
        if idx_col:
            d = df[[idx_col, feat]].sort_values(idx_col)
            plt.plot(d[idx_col].values, d[feat].values)
            plt.xlabel(idx_col)
        else:
            plt.plot(s.values)
            plt.xlabel("index")

        plt.title(f"{feat} — over windows")
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{feat}__trace.png"), dpi=150)
        plt.close()

    print(f"✅ Wrote plots to: {OUT_DIR}")

if __name__ == "__main__":
    main()
