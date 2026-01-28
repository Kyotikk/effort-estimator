# ml/features/sanitise.py

import pandas as pd
import numpy as np
from typing import Tuple, List
import re


# Metadata columns that should NEVER be features
# NOTE: t_center is kept for alignment but should be excluded at model training
METADATA_PATTERNS = [
    r'^t_start',      # t_start
    r'^t_end',        # t_end
    r'^window_id',    # window_id, window_id_r, etc.
    r'^start_idx',    # start_idx, start_idx_r
    r'^end_idx',      # end_idx, end_idx_r
    r'_idx$',         # anything ending in _idx
    r'_idx_r',        # merged index columns
    r'^modality',     # modality column
    r'^subject$',     # subject identifier
    r'^activity$',    # activity label
    r'^borg',         # target variable
]


def _is_metadata_col(col: str) -> bool:
    """Check if column name matches metadata patterns."""
    col_lower = col.lower()
    for pattern in METADATA_PATTERNS:
        if re.search(pattern, col_lower):
            return True
    return False


def sanitise_features(
    df: pd.DataFrame,
    *,
    drop_bool: bool = True,
    drop_non_numeric: bool = True,
    drop_nan_cols: bool = True,
    drop_metadata: bool = True,
    nan_threshold: float = 0.5,  # Only drop columns with >50% NaN
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Enforce scalar numeric feature contract.
    
    Removes:
    - Boolean columns
    - Non-numeric columns (strings, objects, arrays)
    - Columns with >nan_threshold NaN values
    - Metadata columns (time, window, index columns) - THESE ARE NOT FEATURES!

    Returns
    -------
    cleaned_df : pd.DataFrame
    dropped_cols : list of str
    """

    dropped = []
    out = df.copy()

    # --- Drop metadata columns (time, window, index - NOT features!) ---
    if drop_metadata:
        metadata_cols = [c for c in out.columns if _is_metadata_col(c)]
        out = out.drop(columns=metadata_cols)
        dropped.extend(metadata_cols)

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

    # --- Drop columns with too many NaNs (>threshold) ---
    if drop_nan_cols:
        nan_pct = out.isna().mean()
        high_nan_cols = nan_pct[nan_pct > nan_threshold].index.tolist()
        out = out.drop(columns=high_nan_cols)
        dropped.extend(high_nan_cols)

    return out, sorted(set(dropped))
