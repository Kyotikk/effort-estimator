# ml/fusion/fuse_windows.py

from typing import Dict, List
import pandas as pd


REQUIRED_WINDOW_COLS = [
    "window_id",
    "t_start",
    "t_center",
    "t_end",
    "modality",
    "borg",
]


def _validate_feature_table(df: pd.DataFrame, modality: str) -> None:
    missing = set(REQUIRED_WINDOW_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{modality}] Missing required window columns: {missing}"
        )

    if df["window_id"].duplicated().any():
        raise ValueError(
            f"[{modality}] Duplicate window_id entries detected"
        )


def _prefix_feature_columns(
    df: pd.DataFrame,
    modality: str,
    metadata_cols: List[str],
) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        if col in metadata_cols:
            continue
        rename_map[col] = f"{modality}__{col}"

    return df.rename(columns=rename_map)


def fuse_feature_tables(
    feature_tables: Dict[str, pd.DataFrame],
    metadata_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Fuse window-aligned feature tables from multiple modalities.

    Parameters
    ----------
    feature_tables : dict
        Mapping modality name -> feature DataFrame
    metadata_cols : list of str, optional
        Columns preserved without prefixing
        Defaults to window timing columns

    Returns
    -------
    pd.DataFrame
        Fused feature table (one row per window)
    """
    if not feature_tables:
        raise ValueError("No feature tables provided for fusion")

    if metadata_cols is None:
        metadata_cols = REQUIRED_WINDOW_COLS.copy()

    fused = None

    for modality, df in feature_tables.items():
        _validate_feature_table(df, modality)

        df = df.copy()

        # Keep metadata columns + features only
        keep_cols = metadata_cols + [
            c for c in df.columns if c not in metadata_cols
        ]
        df = df.loc[:, keep_cols]

        df = _prefix_feature_columns(
            df=df,
            modality=modality,
            metadata_cols=metadata_cols,
        )

        if fused is None:
            fused = df
        else:
            before = len(fused)
            fused = fused.merge(
                df,
                on=metadata_cols,
                how="inner",
                validate="one_to_one",
            )
            after = len(fused)

            if after == 0:
                raise RuntimeError(
                    f"Fusion failed: no overlapping windows after adding modality '{modality}'"
                )

            if after < before:
                print(
                    f"[fusion] Warning: dropped {before - after} windows "
                    f"when merging modality '{modality}'"
                )

    assert fused is not None
    return fused
