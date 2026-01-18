# ml/fusion/fuse_windows.py

from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd

# Columns describing the window itself, not physiological features
META_COLS = {
    "window_id",
    "start_idx",
    "end_idx",
    "t_start",
    "t_center",
    "t_end",
    "modality",
}


def _feature_cols(df: pd.DataFrame, join_cols: List[str]) -> List[str]:
    """
    Return true feature columns from a table (exclude join + meta).
    """
    drop = set(join_cols) | META_COLS
    return [c for c in df.columns if c not in drop]


def _fuse_two_feature_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_col: str = "t_center",
    tolerance_sec: Optional[float] = None,
    direction: str = "nearest",
    suffixes: Tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """
    Fuse two feature tables by a time-like join column (default: t_center).

    - Keeps left's meta columns as canonical.
    - Excludes window meta from the "right feature" subset, so we never
      drop based on start_idx / end_idx / t_* / window_id.
    - Handles suffixed right-side feature columns after merge_asof.
    """

    if join_col not in left.columns or join_col not in right.columns:
        raise RuntimeError(
            f"Missing join_col={join_col}. "
            f"left_has={join_col in left.columns} right_has={join_col in right.columns}"
        )

    # Ensure sorted by join column for merge_asof
    left = left.sort_values(join_col).reset_index(drop=True)
    right = right.sort_values(join_col).reset_index(drop=True)

    # Time-based merge
    merge_kwargs = dict(
        left=left,
        right=right,
        on=join_col,
        direction=direction,
        suffixes=suffixes,
    )
    if tolerance_sec is not None:
        merge_kwargs["tolerance"] = tolerance_sec

    fused = pd.merge_asof(**merge_kwargs)

    # Identify right-side true feature columns (no join/meta)
    right_feats = _feature_cols(right, join_cols=[join_col])

    # Map them to actual column names in the fused table
    existing = set(fused.columns)
    right_feats_in_fused: List[str] = []

    for c in right_feats:
        if c in existing:
            right_feats_in_fused.append(c)
            continue

        # Common case: merge_asof added suffix for right side
        cand = f"{c}{suffixes[1]}"
        if cand in existing:
            right_feats_in_fused.append(cand)

    # If we have right-side features, drop rows where ALL of them are NaN
    if right_feats_in_fused:
        fused = fused.dropna(subset=right_feats_in_fused, how="all").reset_index(drop=True)
    else:
        fused = fused.reset_index(drop=True)

    return fused


def fuse_feature_tables(
    tables: List[pd.DataFrame],
    join_col: str = "t_center",
    tolerance_sec: Optional[float] = None,
    direction: str = "nearest",
    suffixes: Tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """
    Backwards-compatible multi-table fusion API.

    This matches what `ml/run_fusion.py` expects:

        fused = fuse_feature_tables(
            tables=[imu_df, ppg_df, eda_df],
            join_col="t_center",
            tolerance_sec=...,
            direction="nearest",
        )

    Internally we fuse sequentially:
      (((table[0] ⨝ table[1]) ⨝ table[2]) ⨝ ...)

    Each step uses the improved _fuse_two_feature_tables that:
      - treats META_COLS as non-features
      - only drops rows where all right-side *features* are NaN
    """

    if not tables:
        raise ValueError("fuse_feature_tables: 'tables' list is empty")

    if len(tables) == 1:
        return tables[0].copy()

    fused = tables[0]
    for t in tables[1:]:
        fused = _fuse_two_feature_tables(
            left=fused,
            right=t,
            join_col=join_col,
            tolerance_sec=tolerance_sec,
            direction=direction,
            suffixes=suffixes,
        )

    return fused
