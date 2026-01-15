import pandas as pd
import yaml
from data.build_xy import build_xy
from models.ridge import train_ridge

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    input_path = cfg["model"]["input"]

    df = pd.read_csv(input_path)

    X_train, y_train = build_xy(df, "train")
    X_val, y_val     = build_xy(df, "val")

    model, metrics = train_ridge(
        X_train, y_train,
        X_val, y_val,
        alpha=1.0
    )

    print("Ridge validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main("config/training.yaml")
