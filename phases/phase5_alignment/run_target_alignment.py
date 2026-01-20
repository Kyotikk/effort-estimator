from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from phases.phase5_alignment.adl_alignment import (
    parse_adl_intervals,
    align_windows_to_borg,
)
from phases.phase5_alignment.compute_hrv_recovery import (
    align_windows_to_hrv_recovery_rr,
)

FUSION_REQUIRED_COLS = [
    "window_id",
    "t_start",
    "t_center",
    "t_end",
    "modality",
]


def run_alignment(
    features_path: Path,
    windows_path: Path,
    adl_path: Path,
    out_path: Path,
    target_type: str = "borg",
    rr_path: Path = None,
    rmssd_windows_path: Path = None,
    rec_start: float = 10.0,
    rec_end: float = 60.0,
    adl_offset_hours: float = 0.0,
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
    # Parse ADL intervals and align target labels onto windows
    # -------------------------
    intervals = parse_adl_intervals(adl_path)

    # Optional ADL time offset (e.g., ADL logger local time vs. ECG UTC)
    if adl_offset_hours != 0:
        delta = adl_offset_hours * 3600.0
        intervals['t_start'] = intervals['t_start'] + delta
        intervals['t_end'] = intervals['t_end'] + delta
        print(f"  Shifted ADL intervals by {adl_offset_hours} h ({delta:.0f} s)")

    # Filter windows to only those within ADL recording time range
    adl_t_start = intervals['t_start'].min()
    adl_t_end = intervals['t_end'].max()
    windows_in_range = windows[
        (windows['t_center'] >= adl_t_start) & 
        (windows['t_center'] <= adl_t_end)
    ].copy()
    
    print(f"  ADL recording time: {adl_t_start:.1f} to {adl_t_end:.1f}")
    print(f"  Windows before filtering: {len(windows)}")
    print(f"  Windows in ADL time range: {len(windows_in_range)}")
    print(f"  Target type: {target_type}")

    if target_type == "borg":
        windows_labeled = align_windows_to_borg(
            windows=windows_in_range,
            intervals=intervals,
        )
    elif target_type == "hrv_recovery_rate":
        if rr_path is None:
            raise ValueError("rr_path required when target_type='hrv_recovery_rate'")
        windows_labeled = align_windows_to_hrv_recovery_rr(
            windows=windows_in_range,
            intervals=intervals,
            rr_path=rr_path,
        )
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'borg' or 'hrv_recovery_rate'.")

    # -------------------------
    # Optional: attach RMSSD activity/recovery slopes to windows
    # -------------------------
    if rmssd_windows_path:
        try:
            rmssd_df = pd.read_csv(rmssd_windows_path)
            required_cols = {"t_center", "rmssd", "t_start", "t_end"}
            missing_rmssd = required_cols - set(rmssd_df.columns)
            if missing_rmssd:
                raise ValueError(f"RMSSD windows missing columns: {missing_rmssd}")

            # Add columns, default NaN
            for col in [
                "rmssd_during_mean",
                "rmssd_during_median",
                "activity_slope_ms_per_s",
                "recovery_slope_ms_per_s",
                "slope_recovery_minus_activity",
            ]:
                if col not in windows_labeled.columns:
                    windows_labeled[col] = np.nan

            # Compute per-activity metrics and assign to windows
            def _fit_slope(df_sub):
                if len(df_sub) < 2:
                    return np.nan
                x = df_sub["t_center"].values
                y = df_sub["rmssd"].values
                slope, _ = np.polyfit(x, y, 1)
                return slope

            for _, act in intervals.iterrows():
                t0, t1 = act["t_start"], act["t_end"]
                # during activity
                act_mask = (rmssd_df["t_center"] >= t0) & (rmssd_df["t_center"] <= t1)
                act_win = rmssd_df.loc[act_mask, ["t_center", "rmssd"]].dropna()
                act_mean = act_win["rmssd"].mean() if not act_win.empty else np.nan
                act_med = act_win["rmssd"].median() if not act_win.empty else np.nan
                act_slope = _fit_slope(act_win)

                # recovery window
                rec_mask = (rmssd_df["t_center"] >= t1 + rec_start) & (rmssd_df["t_center"] <= t1 + rec_end)
                rec_win = rmssd_df.loc[rec_mask, ["t_center", "rmssd"]].dropna()
                rec_slope = _fit_slope(rec_win)

                # write to windows within activity / recovery
                win_act_mask = (windows_labeled["t_center"] >= t0) & (windows_labeled["t_center"] <= t1)
                win_rec_mask = (windows_labeled["t_center"] >= t1 + rec_start) & (windows_labeled["t_center"] <= t1 + rec_end)

                if win_act_mask.any():
                    windows_labeled.loc[win_act_mask, "rmssd_during_mean"] = act_mean
                    windows_labeled.loc[win_act_mask, "rmssd_during_median"] = act_med
                    windows_labeled.loc[win_act_mask, "activity_slope_ms_per_s"] = act_slope
                if win_rec_mask.any():
                    windows_labeled.loc[win_rec_mask, "recovery_slope_ms_per_s"] = rec_slope
                    delta = rec_slope - act_slope if not np.isnan(rec_slope) and not np.isnan(act_slope) else np.nan
                    windows_labeled.loc[win_rec_mask, "slope_recovery_minus_activity"] = delta
        except Exception as e:
            print(f"  ⚠ RMSSD attachment failed: {e}")

    # -------------------------
    # Choose merge keys
    # Prefer start_idx/end_idx; fallback to window_id
    # -------------------------
    have_se = {"start_idx", "end_idx"}.issubset(features.columns) and {"start_idx", "end_idx"}.issubset(windows_labeled.columns)
    have_wid = ("window_id" in features.columns) and ("window_id" in windows_labeled.columns)

    # Dynamic target column name
    target_col = "hrv_recovery_rate" if target_type == "hrv_recovery_rate" else "borg"

    extra_cols = [
        c
        for c in [
            "rmssd_during_mean",
            "rmssd_during_median",
            "activity_slope_ms_per_s",
            "recovery_slope_ms_per_s",
            "slope_recovery_minus_activity",
        ]
        if c in windows_labeled.columns
    ]

    if have_se:
        merge_keys = ["start_idx", "end_idx"]
        meta_cols = ["start_idx", "end_idx", "window_id", "t_start", "t_center", "t_end", target_col] + extra_cols
    elif have_wid:
        merge_keys = ["window_id"]
        meta_cols = ["window_id", "t_start", "t_center", "t_end", target_col] + extra_cols
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

    # Also filter features to only those in time range (for fused data with merged features)
    if 't_center' in features.columns:
        features = features[
            (features['t_center'] >= adl_t_start) & 
            (features['t_center'] <= adl_t_end)
        ].copy()

    Xy = features.merge(
        windows_labeled[meta_cols],
        on=merge_keys,
        how="left",
    )

    # Ensure modality exists (fusion requires it)
    if "modality" not in Xy.columns:
        Xy["modality"] = "fused"

    # Add target column to required cols for validation
    fusion_cols_check = FUSION_REQUIRED_COLS + [target_col]

    # Drop unlabeled windows (keep only windows with target labels)
    # Xy = Xy.dropna(subset=[target_col]).copy()

    # Final contract check for fusion
    missing = [c for c in fusion_cols_check if c not in Xy.columns]
    if missing:
        raise ValueError(
            f"Aligned output missing fusion-required columns: {missing}. "
            f"Columns present: {list(Xy.columns)}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Xy.to_csv(out_path, index=False)

    n_labeled = int(Xy[target_col].notna().sum()) if target_col in Xy.columns else 0
    print(f"Saved aligned dataset to {out_path} ({len(Xy)} windows | {target_col} labeled: {n_labeled})")


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
