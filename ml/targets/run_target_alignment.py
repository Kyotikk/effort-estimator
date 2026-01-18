from pathlib import Path
import pandas as pd
import yaml

from ml.targets.adl_alignment import (
    parse_adl_intervals,
    align_windows_to_borg,
)

FUSION_REQUIRED_COLS = [
    "window_id",
    "t_start",
    "t_center",
    "t_end",
    "modality",
    "borg",
]


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

    # -------------------------
    # Ensure window_id exists (required by fusion)
    # -------------------------
    if "window_id" not in windows.columns:
        windows = windows.reset_index(drop=True)
        windows["window_id"] = windows.index.astype(int)

    # sanity: must have timing fields for fusion
    for c in ["t_start", "t_center", "t_end"]:
        if c not in windows.columns:
            raise ValueError(f"Windows missing '{c}'. Found: {list(windows.columns)}")

    # -------------------------
    # Parse ADL intervals and align Borg labels onto windows
    # -------------------------
    intervals = parse_adl_intervals(adl_path)

    windows_labeled = align_windows_to_borg(
        windows=windows,
        intervals=intervals,
    )

    # -------------------------
    # Choose merge keys
    # Prefer start_idx/end_idx; fallback to window_id
    # -------------------------
    have_se = {"start_idx", "end_idx"}.issubset(features.columns) and {"start_idx", "end_idx"}.issubset(windows_labeled.columns)
    have_wid = ("window_id" in features.columns) and ("window_id" in windows_labeled.columns)

    if have_se:
        merge_keys = ["start_idx", "end_idx"]
        meta_cols = ["start_idx", "end_idx", "window_id", "t_start", "t_center", "t_end", "borg"]
    elif have_wid:
        merge_keys = ["window_id"]
        meta_cols = ["window_id", "t_start", "t_center", "t_end", "borg"]
    else:
        raise ValueError(
            "Cannot align: need either start_idx/end_idx in BOTH features & windows, "
            "or window_id in BOTH.\n"
            f"Features cols: {list(features.columns)}\n"
            f"Windows cols: {list(windows_labeled.columns)}"
        )

    # -------------------------
    # Merge: add borg + fusion metadata columns to features
    # Avoid duplicate columns -> no _x/_y suffixes
    # -------------------------
    cols_to_drop = [c for c in meta_cols if c in features.columns and c not in merge_keys]
    if cols_to_drop:
        features = features.drop(columns=cols_to_drop)

    Xy = features.merge(
        windows_labeled[meta_cols],
        on=merge_keys,
        how="left",
    )

    # Ensure modality exists (fusion requires it)
    if "modality" not in Xy.columns:
        Xy["modality"] = "fused"

    # Drop unlabeled windows (keep only windows with Borg labels)
    # Xy = Xy.dropna(subset=["borg"]).copy()

    # Final contract check for fusion
    missing = [c for c in FUSION_REQUIRED_COLS if c not in Xy.columns]
    if missing:
        raise ValueError(
            f"Aligned output missing fusion-required columns: {missing}. "
            f"Columns present: {list(Xy.columns)}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Xy.to_csv(out_path, index=False)

    n_labeled = int(Xy["borg"].notna().sum()) if "borg" in Xy.columns else 0
    print(f"Saved aligned dataset to {out_path} ({len(Xy)} windows | borg labeled: {n_labeled})")


# -----------------------------------------------------------------
# Entry point (optional utility)
# -----------------------------------------------------------------
def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    adl_path = cfg["targets"]["imu"]["adl_path"]

    # This CLI utility is not used by run_pipeline.py, but kept for manual testing.
    raise SystemExit(
        "run_target_alignment.py is intended to be called by run_pipeline.py via run_alignment(...). "
        "If you want a CLI, add args parsing here."
    )


if __name__ == "__main__":
    main("config/pipeline.yaml")
