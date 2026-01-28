#!/usr/bin/env python3
"""Show top features for each subject."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df.dropna(subset=["borg"]).copy()

def get_feature_columns(df):
    skip = {"window_id", "start_idx", "end_idx", "valid", "t_start", "t_center", 
            "t_end", "n_samples", "win_sec", "modality", "subject", "borg"}
    return [c for c in df.columns if c not in skip and not c.endswith("_r")]

def select_features(X, y, names, top_n=50, corr_thresh=0.85):
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    corrs = [abs(np.corrcoef(X[:,i], y)[0,1]) if np.std(X[:,i]) > 1e-10 else 0 for i in range(X.shape[1])]
    corrs = [c if np.isfinite(c) else 0 for c in corrs]
    
    top_idx = np.argsort(corrs)[-top_n:][::-1]
    selected = []
    for idx in top_idx:
        redundant = any(abs(np.corrcoef(X[:,idx], X[:,s])[0,1]) > corr_thresh 
                       for s in selected if np.std(X[:,idx]) > 1e-10 and np.std(X[:,s]) > 1e-10)
        if not redundant:
            selected.append(idx)
    
    return selected, [names[i] for i in selected], [corrs[i] for i in selected]

df = load_data("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv")
feature_cols = get_feature_columns(df)

for subject in sorted(df["subject"].unique()):
    df_sub = df[df["subject"] == subject]
    X = df_sub[feature_cols].values
    y = df_sub["borg"].values
    
    # Feature selection
    sel_idx, sel_names, sel_corrs = select_features(X, y, feature_cols, top_n=50)
    
    X_sel = np.nan_to_num(X[:, sel_idx], nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    
    # Train model
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, 
                             reg_lambda=2.0, min_child_weight=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y, verbose=False)
    
    # Get importance
    importance = model.feature_importances_
    
    print(f"\n{'='*70}")
    print(f"TOP FEATURES: {subject}")
    print(f"{'='*70}")
    print(f"Borg range: {y.min():.1f} - {y.max():.1f}")
    print(f"\n{'Rank':<5} {'Feature':<45} {'Corr':>8} {'Importance':>12}")
    print("-" * 70)
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:20], 1):
        fname = sel_names[idx]
        corr = sel_corrs[idx]
        imp = importance[idx]
        
        # Categorize feature
        if 'eda' in fname.lower() or 'scr' in fname.lower() or 'scl' in fname.lower():
            cat = "🔵 EDA"
        elif 'ppg' in fname.lower() or 'hr' in fname.lower():
            cat = "❤️ PPG"
        elif 'acc' in fname.lower() or 'imu' in fname.lower():
            cat = "📱 IMU"
        else:
            cat = "   "
        
        print(f"{rank:<5} {cat} {fname:<40} {corr:>8.3f} {imp:>12.4f}")

print("\n" + "="*70)
print("LEGEND")
print("="*70)
print("🔵 EDA = Electrodermal Activity (skin conductance, stress)")
print("❤️ PPG = Photoplethysmography (heart rate related)")
print("📱 IMU = Accelerometer/motion features")
print("\nCorr = Correlation with Borg score")
print("Importance = XGBoost feature importance (gain)")
