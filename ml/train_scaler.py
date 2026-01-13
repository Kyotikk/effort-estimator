import yaml
import numpy as np
import pandas as pd
from scalers.imu_scaler import (
    fit_imu_scaler,
    transform_imu_features,
    save_imu_scaler,
)

# ---------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    out = df.copy()
    out["split"] = "train"

    mask = rng.random(len(df))
    out.loc[mask > 0.8, "split"] = "test"
    out.loc[(mask > 0.7) & (mask <= 0.8), "split"] = "val"

    return out

def drop_nan_features(train_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    valid = [
        c for c in feature_cols
        if not train_df[c].isna().any()
    ]
    dropped = set(feature_cols) - set(valid)

    if dropped:
        print("Dropping features with NaNs:")
        for c in sorted(dropped):
            print("  -", c)

    return valid

# ---------------------------------------------------------------------
def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    df = pd.read_csv(cfg["dataset"]["imu_features_path"])

    df = assign_splits(df, seed=cfg["splits"]["random_seed"])

    split_col = cfg["splits"]["column"]

    train_df = df[df[split_col] == cfg["splits"]["train"]]
    val_df   = df[df[split_col] == cfg["splits"]["val"]]

    scaler, feature_cols = fit_imu_scaler(train_df)

    train_df = transform_imu_features(train_df, scaler, feature_cols)
    val_df   = transform_imu_features(val_df, scaler, feature_cols)

    save_imu_scaler(
        scaler,
        feature_cols,
        cfg["normalisation"]["imu"]["save_to"],
    )

    print("IMU scaler fitted and saved.")
    print(f"Features scaled: {len(feature_cols)}")

    assert abs(train_df[feature_cols].mean()).mean() < 1e-6
    stds = train_df[feature_cols].std(ddof=0)
    assert stds.between(0.98, 1.02).mean() > 0.95
    assert not val_df[feature_cols].isna().any().any()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main("config/training.yaml")
