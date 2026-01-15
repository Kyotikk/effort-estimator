from pathlib import Path
import pandas as pd
import yaml

from ml.targets.adl_alignment import (
    parse_adl_intervals,
    align_windows_to_borg,
)

def run_alignment(
    features_path: Path,
    windows_path: Path,
    adl_path: Path,
    out_path: Path,
) -> None:
    # -------------------------
    # Load inputs
    # -------------------------
    features = pd.read_csv(features_path)
    windows = pd.read_csv(windows_path)

    intervals = parse_adl_intervals(adl_path)

    # -------------------------
    # Align Borg labels
    # -------------------------
    windows_labeled = align_windows_to_borg(
        windows=windows,
        intervals=intervals,
    )

    # -------------------------
    # Merge with features
    # -------------------------
    Xy = features.merge(
        windows_labeled[["start_idx", "end_idx", "borg"]],
        on=["start_idx", "end_idx"],
        how="left",
    )

    # Drop unlabeled windows
    Xy = Xy.dropna(subset=["borg"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Xy.to_csv(out_path, index=False)

    print(
        f"Saved aligned dataset to {out_path} "
        f"({len(Xy)} windows with Borg labels)"
    )

# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    adl_path = cfg["targets"]["imu"]["adl_path"]

    run_alignment(
        features_path=Path("C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_elderly_3\\effort_estimation_output\\parsingsim3_sim_elderly3\\imu_bioz\\imu_features_10.0s.csv"),
        windows_path=Path("C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_elderly_3\\effort_estimation_output\\parsingsim3_sim_elderly3\\imu_bioz\\imu_windows_10.0s.csv"),
        adl_path=adl_path,
        out_path=Path("C:\\Users\\Nicla\\Documents\\ETHZ\\Lifelogging\\Data\\interim\\scai-ncgg\\parsingsim3\\sim_elderly_3\\effort_estimation_output\\parsingsim3_sim_elderly3\\imu_bioz\\imu_aligned_10.0s.csv"),
    )

if __name__ == "__main__":
    main("config/training.yaml")