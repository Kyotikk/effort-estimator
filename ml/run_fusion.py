# ml/fusion/run_fusion.py

import yaml
import pandas as pd
from pathlib import Path

from ml.fusion.fuse_windows import fuse_feature_tables
from ml.features.sanitise import sanitise_features


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    window_lengths = cfg["fusion"]["window_lengths_sec"]
    output_dir = Path(cfg["fusion"]["output_dir"])

    for win_len in window_lengths:
        print(f"\n▶ Fusing features for window length: {win_len}s")
        
        feature_tables = {}

        # Load feature tables for this window length
        for modality, path_template in cfg["fusion"]["modalities"].items():
            # Replace placeholder with actual window length
            path = path_template.format(window_length=f"{win_len:.1f}s")
            df = pd.read_csv(path)
            feature_tables[modality] = df
            print(f"  Loaded {modality}: {len(df)} windows")

        fused = fuse_feature_tables(
            feature_tables=feature_tables
        )

        fused, dropped_cols = sanitise_features(fused)

        print(f"  Dropped {len(dropped_cols)} invalid feature columns")

        # Create output path with window length
        out_path = output_dir / f"fused_features_{win_len:.1f}s.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Persist this decision
        dropped_path = output_dir / f"dropped_features_{win_len:.1f}s.csv"
        pd.Series(dropped_cols, name="dropped_feature").to_csv(
            dropped_path,
            index=False
        )
        
        fused.to_csv(out_path, index=False)

        print(f"  ✓ Wrote {len(fused)} fused windows → {out_path}")
 

if __name__ == "__main__":
    main("config/fusion.yaml")
