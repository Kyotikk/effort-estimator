# phases/phase4_fusion/sanitise.py

import pandas as pd
import numpy as np
from typing import Tuple, List


def sanitise_features(
    df: pd.DataFrame,
    *,
    drop_bool: bool = True,
    drop_non_numeric: bool = True,
    drop_nan_cols: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Enforce scalar numeric feature contract.

    Returns
    -------
    cleaned_df : pd.DataFrame
    dropped_cols : list of str
    """

    dropped = []
    out = df.copy()

    # --- Drop boolean columns ---
    if drop_bool:
        bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
        out = out.drop(columns=bool_cols)
        dropped.extend(bool_cols)

    # --- Drop non-numeric columns (arrays, strings, objects) ---
    if drop_non_numeric:
        non_numeric = out.select_dtypes(exclude=[np.number]).columns.tolist()
        out = out.drop(columns=non_numeric)
        dropped.extend(non_numeric)

    # --- Drop columns containing NaNs ---
    if drop_nan_cols:
        nan_cols = out.columns[out.isna().any()].tolist()
        out = out.drop(columns=nan_cols)
        dropped.extend(nan_cols)

    return out, sorted(set(dropped))
