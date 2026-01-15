import pandas as pd
from typing import Tuple, List

META_COLS = {
    "window_id",
    "t_start",
    "t_center",
    "t_end",
    "borg",
    "split",
}

def select_imu_features(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("imu__") and c not in META_COLS
    ]

def build_xy(
    df: pd.DataFrame,
    split: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    df_split = df[df["split"] == split]

    feature_cols = select_imu_features(df_split)

    X = df_split[feature_cols]
    y = df_split["borg"]

    return X, y
