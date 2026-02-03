#!/usr/bin/env python3
"""Full comparison with HR, EDA, and ACC/IMU"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')
df = df.dropna(subset=['borg']).reset_index(drop=True)
y = df['borg'].values
activity_ids = np.cumsum(np.diff(y, prepend=y[0]) != 0)

# Feature sets
hr_cols = ['ppg_green_hr_mean', 'ppg_green_hr_max']
eda_cols = ['eda_phasic_max', 'eda_cc_range']
acc_cols = ['acc_x_dyn__quantile_0.6', 'acc_z_dyn__sum_of_absolute_changes', 'acc_y_dyn__sample_entropy']
all_cols = hr_cols + eda_cols + acc_cols

print("="*60)
print("COMPARISON WITH ALL MODALITIES (HR + EDA + ACC/IMU)")
print("="*60)
print(f"Data: {len(df)} windows, {len(np.unique(activity_ids))} activities")

gkf = GroupKFold(n_splits=5)

def evaluate_cv(features, name):
    X = df[features].fillna(0)
    pred = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, activity_ids):
        scaler = StandardScaler()
        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
        model.fit(scaler.fit_transform(X.iloc[train_idx]), y[train_idx])
        pred[test_idx] = model.predict(scaler.transform(X.iloc[test_idx]))
    r, _ = pearsonr(y, pred)
    print(f"  {name}: r = {r:.3f}")
    return r

print("\nXGBoost with GroupKFold CV (honest):")
evaluate_cv(hr_cols, "HR only (2 features)")
evaluate_cv(acc_cols, "ACC/IMU only (3 features)")
evaluate_cv(eda_cols, "EDA only (2 features)")
evaluate_cv(hr_cols + acc_cols, "HR + ACC (5 features)")
evaluate_cv(hr_cols + eda_cols, "HR + EDA (4 features)")
evaluate_cv(all_cols, "HR + EDA + ACC (7 features)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Features in each model:
  - HR: ppg_green_hr_mean, ppg_green_hr_max
  - ACC/IMU: acc_x_dyn__quantile_0.6, acc_z_dyn__sum_of_absolute_changes, acc_y_dyn__sample_entropy
  - EDA: eda_phasic_max, eda_cc_range

The r=0.843 "literature formula" was NOT cross-validated.
These results ARE cross-validated (GroupKFold by activity).
""")
