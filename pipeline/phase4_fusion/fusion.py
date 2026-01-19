# ml/fusion/run_fusion.py

from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from ml.fusion.fuse_windows import fuse_feature_tables
from ml.features.sanitise import sanitise_features


def _load_table(path: str, modality: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {modality} feature table: {p}")
    df = pd.read_csv(p)
    if "t_center" not in df.columns:
        raise RuntimeError(f"{modality} feature table missing t_center: {p} | cols={list(df.columns)[:40]}")
    return df


def _range_str(df: pd.DataFrame) -> str:
    t = pd.to_numeric(df["t_center"], errors="coerce").dropna()
    if t.empty:
        return "t_center empty"
    return f"t_center min={t.min():.3f} max={t.max():.3f} n={len(df)}"


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    window_lengths = cfg["fusion"]["window_lengths_sec"]
    output_dir = Path(cfg["fusion"]["output_dir"])
    modalities_cfg = cfg["fusion"]["modalities"]

    # tolerance config:
    # - either a single float: cfg["fusion"]["tolerance_s"] = 0.2
    # - or a dict by window length string: {"2.0": 0.1, "10.0": 0.3}
    tol_cfg = cfg["fusion"].get("tolerance_s", None)

    for win_len in window_lengths:
        win_len = float(win_len)
        print(f"\n▶ Fusing features for window length: {win_len:.1f}s")

        # resolve tolerance
        tolerance_s = None
        if isinstance(tol_cfg, (int, float)):
            tolerance_s = float(tol_cfg)
        elif isinstance(tol_cfg, dict):
            # keys can be "10.0" or 10.0
            key1 = f"{win_len:.1f}"
            key2 = str(win_len)
            if key1 in tol_cfg:
                tolerance_s = float(tol_cfg[key1])
            elif key2 in tol_cfg:
                tolerance_s = float(tol_cfg[key2])

        # load tables for this window length
        feature_tables: Dict[str, pd.DataFrame] = {}
        for modality, path_template in modalities_cfg.items():
            path = path_template.format(window_length=f"{win_len:.1f}s")
            df = _load_table(path, modality)
            feature_tables[modality] = df
            print(f"  Loaded {modality}: {len(df)} windows | {_range_str(df)}")

        # fuse
        # Convert dict to ordered list of dataframes
        tables_list = [feature_tables[modality] for modality in modalities_cfg.keys()]
        fused = fuse_feature_tables(
            tables=tables_list,
            join_col="t_center",
            tolerance_sec=tolerance_s,
        )

        # sanitise features
        fused, dropped_cols = sanitise_features(fused)
        print(f"  Dropped {len(dropped_cols)} invalid feature columns")

        # write outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"fused_features_{win_len:.1f}s.csv"
        dropped_path = output_dir / f"dropped_features_{win_len:.1f}s.csv"

        pd.Series(dropped_cols, name="dropped_feature").to_csv(dropped_path, index=False)
        fused.to_csv(out_path, index=False)

        print(f"  ✓ Wrote {len(fused)} fused windows → {out_path}")


if __name__ == "__main__":
    main("config/fusion.yaml")
