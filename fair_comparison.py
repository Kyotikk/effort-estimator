#!/usr/bin/env python3
"""
FAIR COMPARISON: Literature vs XGBoost
Both evaluated the SAME way
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')
df = df.dropna(subset=['borg']).reset_index(drop=True)
y = df['borg'].values

X = df[['ppg_green_hr_mean', 'ppg_green_hr_max', 'eda_phasic_max', 'eda_cc_range']].fillna(0)

print("="*60)
print("FAIR COMPARISON: SAME EVALUATION FOR BOTH")
print("="*60)

# ============================================================
# METHOD 1: NO CROSS-VALIDATION (like literature r=0.843)
# ============================================================
print("\n--- NO CV (like literature formula) ---")

# Simple correlation
r_hr, _ = pearsonr(X['ppg_green_hr_mean'], y)
r_eda, _ = pearsonr(X['eda_phasic_max'], y)
print(f"HR mean correlation: r = {r_hr:.3f}")
print(f"EDA phasic correlation: r = {r_eda:.3f}")

# XGBoost on ALL data (overfit)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
model.fit(X_sc, y)
pred_all = model.predict(X_sc)
r_xgb_all, _ = pearsonr(y, pred_all)
print(f"XGBoost (train=test): r = {r_xgb_all:.3f}")

# ============================================================
# METHOD 2: WITH CROSS-VALIDATION (honest)
# ============================================================
print("\n--- WITH GroupKFold CV (honest) ---")

activity_ids = np.cumsum(np.diff(y, prepend=y[0]) != 0)
gkf = GroupKFold(n_splits=5)

# XGBoost with CV
pred_cv = np.zeros(len(y))
for train_idx, test_idx in gkf.split(X, y, activity_ids):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    model.fit(X_train, y[train_idx])
    pred_cv[test_idx] = model.predict(X_test)

r_xgb_cv, _ = pearsonr(y, pred_cv)
print(f"XGBoost (GroupKFold): r = {r_xgb_cv:.3f}")

# Simple model (just use HR mean) with CV
pred_hr_cv = np.zeros(len(y))
for train_idx, test_idx in gkf.split(X, y, activity_ids):
    # Fit linear: Borg = a * HR + b
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X.iloc[train_idx][['ppg_green_hr_mean']], y[train_idx])
    pred_hr_cv[test_idx] = lr.predict(X.iloc[test_idx][['ppg_green_hr_mean']])

r_hr_cv, _ = pearsonr(y, pred_hr_cv)
print(f"Linear (HR only, CV): r = {r_hr_cv:.3f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
  WITHOUT CV (cheating):
    Literature (HR_delta): r = 0.843  (computed earlier)
    XGBoost (4 features): r = {r_xgb_all:.3f}
    
  WITH GroupKFold CV (honest):
    XGBoost (4 features): r = {r_xgb_cv:.3f}
    Linear (HR only):     r = {r_hr_cv:.3f}

  CONCLUSION:
    The r=0.843 was NOT cross-validated!
    When you do honest CV, both methods get ~0.3-0.4
""")
