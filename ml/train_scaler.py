import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scalers.imu_scaler import (
    fit_imu_scaler,
    transform_imu_features,
    save_imu_scaler,
)

# ---------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def assign_splits(df: pd.DataFrame, seed: int = 42, subject_col: str = None) -> pd.DataFrame:
    """
    Randomly assign train/val/test splits to dataframe.
    
    If subject_col is provided, performs within-subject shuffling to avoid
    temporal domain shift from chronologically ordered protocol. Otherwise,
    shuffles all samples globally.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    seed : int
        Random seed for reproducibility
    subject_col : str, optional
        Column name containing subject/session IDs. If provided, samples are
        shuffled within each subject before split assignment. Common choices:
        'session_id', 'subject_id', 'dataset_name', or similar metadata column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'split' column added (70% train, 15% val, 15% test)
    """
    rng = np.random.default_rng(seed)

    out = df.copy()
    
    # If subject_col specified, shuffle within subjects
    if subject_col is not None and subject_col in out.columns:
        print(f"Assigning splits with within-subject shuffling (subject_col='{subject_col}')")
        
        # Group by subject and shuffle within each group
        groups = []
        for subject, group in out.groupby(subject_col, sort=False):
            shuffled_indices = rng.permutation(group.index)
            groups.append(out.loc[shuffled_indices])
        
        out = pd.concat(groups, ignore_index=False).sort_index()
    else:
        # No subject column: shuffle all data globally
        print("Assigning splits with global shuffling (no subject column detected)")
        shuffled_indices = rng.permutation(out.index)
        out = out.loc[shuffled_indices]
    
    # Assign splits based on random draw
    out["split"] = "train"
    
    mask = rng.random(len(out))
    out.loc[mask > 0.85, "split"] = "test"
    out.loc[(mask > 0.7) & (mask <= 0.85), "split"] = "val"

    return out

# ---------------------------------------------------------------------
def main(cfg_path: str):
    """
    Fit and save scalers for fused features from the preprocessing pipeline.
    
    Loads fused features from each window length, assigns splits with optional
    within-subject shuffling, fits scaler on training data, and transforms/saves
    all splits.
    
    Parameters
    ----------
    cfg_path : str
        Path to training config file
    """
    cfg = load_cfg(cfg_path)
    
    # Get paths from config
    fused_features_path = cfg["dataset"]["imu_features_path"]
    output_dir = Path(cfg["normalisation"]["imu"]["save_to"]).parent
    
    print(f"\n=== Scaling Fused Features ===")
    print(f"Input: {fused_features_path}")
    print(f"Output dir: {output_dir}")
    
    # Load fused features from pipeline output
    df = pd.read_csv(fused_features_path)
    print(f"\nLoaded fused features: {len(df)} windows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # Detect subject/session identifier for within-subject shuffling
    # Try common column names
    subject_col = None
    for candidate in ["session_id", "subject_id", "dataset_name", "participant_id", "session"]:
        if candidate in df.columns:
            subject_col = candidate
            break
    
    # Assign train/val/test splits (with within-subject shuffling if subject column found)
    df = assign_splits(df, seed=cfg["splits"]["random_seed"], subject_col=subject_col)
    if subject_col:
        print(f"✓ Within-subject shuffling applied (detected: '{subject_col}')")
    
    split_col = cfg["splits"]["column"]
    train_split = cfg["splits"]["train"]
    val_split = cfg["splits"]["val"]
    test_split = cfg["splits"]["test"]
    
    train_df = df[df[split_col] == train_split].copy()
    val_df   = df[df[split_col] == val_split].copy()
    test_df  = df[df[split_col] == test_split].copy()
    
    print(f"\nSplit assignments:")
    print(f"  Train: {len(train_df)} windows")
    print(f"  Val:   {len(val_df)} windows")
    print(f"  Test:  {len(test_df)} windows")
    
    # Fit scaler on training data
    scaler, feature_cols = fit_imu_scaler(train_df)
    print(f"\nFitted scaler on {len(feature_cols)} features")
    
    # Transform all splits
    train_df = transform_imu_features(train_df, scaler, feature_cols)
    val_df   = transform_imu_features(val_df, scaler, feature_cols)
    test_df  = transform_imu_features(test_df, scaler, feature_cols)
    
    # Combine splits back together
    df_scaled = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save scaler
    scaler_path = cfg["normalisation"]["imu"]["save_to"]
    save_imu_scaler(
        scaler,
        feature_cols,
        scaler_path,
    )
    print(f"\n✓ Scaler saved to: {scaler_path}")
    
    # Save scaled features
    output_path = output_dir / "fused_features_scaled.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"✓ Scaled features saved to: {output_path}")
    
    print(f"\nFeatures scaled: {len(feature_cols)}")
    print(f"Mean of scaled features (should be ~0): {df_scaled[feature_cols].mean().mean():.2e}")
    print(f"Std of scaled features (should be ~1): {df_scaled[feature_cols].std(ddof=0).mean():.2f}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main("config/training.yaml")
