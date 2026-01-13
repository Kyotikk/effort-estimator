# ml/scalers/imu_scaler.py

from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

def get_imu_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select IMU feature columns.
    Convention: TIFEX features contain '__' in the column name.
    """
    return [c for c in df.columns if "__" in c]

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

def find_non_scalar_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    bad = []
    for c in feature_cols:
        val = df[c].dropna().iloc[0]
        if isinstance(val, str) and val.strip().startswith("["):
            bad.append(c)
    return bad

# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------

def fit_imu_scaler(
    train_df: pd.DataFrame,
) -> Tuple[StandardScaler, List[str]]:
    """
    Fit a StandardScaler on IMU features only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training feature DataFrame (IMU only)

    Returns
    -------
    scaler : StandardScaler
    feature_cols : list of str
        Column order used during fitting (must be reused!)
    """
    feature_cols = get_imu_feature_columns(train_df)

    feature_cols = drop_nan_features(train_df, feature_cols)

    bad_cols = find_non_scalar_features(train_df, feature_cols)

    print("Non-scalar feature columns:")
    for c in bad_cols:
        print("  -", c)

    train_df = train_df.drop(columns=bad_cols)
    feature_cols = [c for c in feature_cols if c not in bad_cols]

    if len(feature_cols) == 0:
        raise ValueError("No IMU feature columns found (missing '__').")

    X = train_df[feature_cols]

    if X.isna().any().any():
        raise ValueError("NaNs detected in IMU features before scaling.")

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler, feature_cols


# ---------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------

def transform_imu_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a fitted IMU scaler to a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (train, val, or test)
    scaler : StandardScaler
        Fitted scaler
    feature_cols : list of str
        Feature column order used during fitting

    Returns
    -------
    pd.DataFrame
        Copy of df with scaled IMU features
    """
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing IMU features at transform time: {missing}")

    out = df.copy()

    out = out.astype({c: "float64" for c in feature_cols})

    # IMPORTANT: pass DataFrame, not numpy array
    X_scaled = scaler.transform(out[feature_cols])

    out[feature_cols] = pd.DataFrame(
        X_scaled,
        columns=feature_cols,
        index=out.index,
    )

    return out


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def save_imu_scaler(
    scaler: StandardScaler,
    feature_cols: List[str],
    path: str,
) -> None:
    """
    Save IMU scaler + feature order.
    """
    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": feature_cols,
        },
        path,
    )


def load_imu_scaler(path: str) -> Tuple[StandardScaler, List[str]]:
    """
    Load IMU scaler + feature order.
    """
    obj = joblib.load(path)
    return obj["scaler"], obj["feature_cols"]
