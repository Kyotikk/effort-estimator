#!/usr/bin/env python3
"""
Train XGBoost for elderly3 with 10% overlap windows.
Uses 5s windows since 10s fusion had issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3")
OUTPUT_DIR = DATA_DIR / "effort_estimation_output"

print("="*70)
print("XGBoost Training: sim_elderly3 with 10% overlap")
print("="*70)

# Load fused features (5s windows with 10% overlap)
fused_path = OUTPUT_DIR / "parsingsim3_sim_elderly3" / "fused_features_5.0s.csv"
fused = pd.read_csv(fused_path)
print(f"\nFused 5s windows: {len(fused)}")

# Load ADL labels - parse the special format
adl_raw = pd.read_csv(DATA_DIR / "scai_app" / "ADLs_1.csv", skiprows=2)
adl_raw.columns = ['Time', 'ADLs', 'Effort']

# Parse time to unix timestamp
from datetime import datetime
def parse_time(t):
    try:
        dt = datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None

adl_raw['t_center'] = adl_raw['Time'].apply(parse_time)

# Keep only rows with effort ratings (End events)
adl = adl_raw[adl_raw['Effort'].notna()].copy()
adl['borg_cr10'] = pd.to_numeric(adl['Effort'], errors='coerce')
adl['Activity'] = adl['ADLs'].str.replace(' End', '')
adl = adl.dropna(subset=['t_center', 'borg_cr10'])
print(f"ADL activities with Borg: {len(adl)}")

# Merge with labels
merged = pd.merge_asof(
    fused.sort_values('t_center'),
    adl[['t_center', 'borg_cr10', 'Activity']].sort_values('t_center'),
    on='t_center',
    direction='nearest',
    tolerance=60
)
labeled = merged.dropna(subset=['borg_cr10'])
print(f"Labeled windows: {len(labeled)}")
print(f"Borg range: {labeled['borg_cr10'].min():.1f} - {labeled['borg_cr10'].max():.1f}")

# Select feature columns (exclude metadata)
meta_cols = ['t_center', 'borg_cr10', 'Activity', 'valid', 'n_samples', 'win_sec', 
             'valid_r', 'n_samples_r', 'win_sec_r', 't_start', 't_end', 'window_id']
feature_cols = [c for c in labeled.columns if c not in meta_cols and not c.startswith('Unnamed')]

# Remove columns with too many NaNs
valid_cols = []
for c in feature_cols:
    if labeled[c].isna().sum() < len(labeled) * 0.3:  # < 30% missing
        valid_cols.append(c)

print(f"\nFeature columns: {len(valid_cols)}")

# Prepare data
X = labeled[valid_cols].fillna(0)
y = labeled['borg_cr10']

# Remove constant columns
non_constant = X.columns[X.std() > 1e-6]
X = X[non_constant]
print(f"Non-constant features: {len(non_constant)}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
print("\n" + "="*70)
print("TRAINING XGBOOST")
print("="*70)

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "="*70)
print("RESULTS (10% overlap, 5s windows)")
print("="*70)
print(f"\nTRAIN SET:")
print(f"  R²:   {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")

print(f"\nTEST SET:")
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*70)
print("TOP 15 FEATURES")
print("="*70)
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:50s} {row['importance']:.4f}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train vs Test scatter
ax1 = axes[0]
ax1.scatter(y_train, y_train_pred, alpha=0.5, label=f'Train (R²={train_r2:.3f})', c='blue')
ax1.scatter(y_test, y_test_pred, alpha=0.7, label=f'Test (R²={test_r2:.3f})', c='red')
ax1.plot([0, 8], [0, 8], 'k--', lw=2)
ax1.set_xlabel('True Borg CR10')
ax1.set_ylabel('Predicted Borg CR10')
ax1.set_title('XGBoost: 10% Overlap, 5s Windows')
ax1.legend()
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 8)

# Feature importance
ax2 = axes[1]
top_feat = importance.head(10)
ax2.barh(range(len(top_feat)), top_feat['importance'], color='steelblue')
ax2.set_yticks(range(len(top_feat)))
ax2.set_yticklabels(top_feat['feature'])
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Features')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'xgboost_10pct_overlap_results.png', dpi=150)
print(f"\n✓ Plot saved to: {OUTPUT_DIR / 'xgboost_10pct_overlap_results.png'}")

# Summary comparison
print("\n" + "="*70)
print("COMPARISON: 10% vs 70% overlap")
print("="*70)
print(f"  {'Metric':<15} {'70% overlap':<15} {'10% overlap':<15}")
print(f"  {'-'*45}")
print(f"  {'Train R²':<15} {'0.999':<15} {train_r2:.3f}")
print(f"  {'Test R²':<15} {'0.938':<15} {test_r2:.3f}")
print(f"  {'Train-Test gap':<15} {'0.061':<15} {train_r2 - test_r2:.3f}")
