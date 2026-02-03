#!/usr/bin/env python3
"""Train XGBoost on elderly3 only - compares 70% vs 10% overlap."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Path to the aligned fused data
DATA_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")

print("="*70)
print("XGBoost Training: sim_elderly3")
print("="*70)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} labeled samples")

# Select feature columns
meta_cols = ['t_center', 'borg_cr10', 'activity', 'valid', 'n_samples', 'win_sec', 
             'valid_r', 'n_samples_r', 'win_sec_r', 't_start', 't_end', 'window_id', 
             'modality', 'start_idx', 'end_idx']
feature_cols = [c for c in df.columns if c not in meta_cols 
                and not c.startswith('Unnamed') 
                and df[c].dtype in ['float64', 'int64']]

# Remove NaN-heavy columns
valid_cols = [c for c in feature_cols if df[c].isna().mean() < 0.3]
print(f"Features: {len(valid_cols)}")

# Prepare X, y
X = df[valid_cols].fillna(0)
y = df['borg']

# Remove rows with NaN in target
mask = ~y.isna()
X = X[mask]
y = y[mask]
print(f"After removing NaN targets: {len(y)} samples")

# Remove constant columns
X = X.loc[:, X.std() > 1e-6]
print(f"Non-constant features: {len(X.columns)}")
print(f"Target range: {y.min():.1f} - {y.max():.1f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nTRAIN:  R² = {train_r2:.4f}  |  RMSE = {train_rmse:.4f}")
print(f"TEST:   R² = {test_r2:.4f}  |  RMSE = {test_rmse:.4f}")
print(f"\nGap (Train-Test R²): {train_r2 - test_r2:.4f}")

# Top features
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print("\nTop 10 Features:")
for _, row in importance.head(10).iterrows():
    print(f"  {row['feature']:45s} {row['importance']:.4f}")
