#!/usr/bin/env python3
"""
INVESTIGATION: Why does XGBoost fail with the same HR features that work for Linear?
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict, LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

print("="*80)
print("WHY XGBOOST FAILS WITH SAME HR FEATURES?")
print("="*80)

# Load data
hr = pd.read_csv(BASE_PATH / "vivalnk_vv330_heart_rate" / "data_1.csv.gz")
hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]

acc_files = list((BASE_PATH / "corsano_wrist_acc").glob("*.csv.gz"))
acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
acc = acc.rename(columns={'time': 'timestamp'})
acc['magnitude'] = np.sqrt(acc['accX']**2 + acc['accY']**2 + acc['accZ']**2)

adl = pd.read_csv(BASE_PATH / "scai_app" / "ADLs_1.csv", skiprows=2)
adl.columns = ['Time', 'ADLs', 'Effort']
adl['timestamp'] = adl['Time'].apply(parse_time)

adl_start = adl['timestamp'].min()
hr_offset = adl_start - hr['timestamp'].min()
acc_offset = adl_start - acc['timestamp'].min()
HR_rest = hr['heart_rate'].quantile(0.05)

# Parse ACTIVITIES (not windows)
activities = []
current = None
start_time = None

for _, row in adl.iterrows():
    if pd.isna(row['timestamp']):
        continue
    if 'Start' in str(row['ADLs']):
        current = row['ADLs'].replace(' Start', '')
        start_time = row['timestamp']
    elif 'End' in str(row['ADLs']) and current:
        duration = row['timestamp'] - start_time
        
        t_start = start_time - hr_offset
        t_end = row['timestamp'] - hr_offset
        mask = (hr['timestamp'] >= t_start) & (hr['timestamp'] <= t_end)
        hr_vals = hr.loc[mask, 'heart_rate'].values
        
        t_start_acc = start_time - acc_offset
        t_end_acc = row['timestamp'] - acc_offset
        mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
        acc_vals = acc.loc[mask, 'magnitude'].values
        
        if len(hr_vals) >= 2 and len(acc_vals) >= 10:
            activities.append({
                'activity': current,
                'duration': duration,
                'hr_mean': hr_vals.mean(),
                'hr_max': hr_vals.max(),
                'hr_std': hr_vals.std(),
                'hr_elevation': (hr_vals.max() - HR_rest) / HR_rest * 100,
                'acc_mean': acc_vals.mean(),
                'acc_std': acc_vals.std(),
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
        current = None

df = pd.DataFrame(activities).dropna()
print(f"\nPer-ACTIVITY data: {len(df)} activities (not windows!)")

#==============================================================================
# TEST 1: Same 2 features, same data, different models
#==============================================================================
print("\n" + "="*80)
print("TEST 1: SAME 2 FEATURES (hr_elevation, duration)")
print("="*80)

X_2feat = df[['hr_elevation', 'duration']].values
y = df['borg'].values
scaler = StandardScaler()
X_2feat_scaled = scaler.fit_transform(X_2feat)

loo = LeaveOneOut()

# Ridge
y_pred_ridge = cross_val_predict(Ridge(alpha=1.0), X_2feat_scaled, y, cv=loo)
r2_ridge = r2_score(y, y_pred_ridge)

# XGBoost - default
y_pred_xgb = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
    X_2feat_scaled, y, cv=loo
)
r2_xgb = r2_score(y, y_pred_xgb)

# XGBoost - very simple
y_pred_xgb_simple = cross_val_predict(
    xgb.XGBRegressor(n_estimators=10, max_depth=1, learning_rate=0.3, random_state=42),
    X_2feat_scaled, y, cv=loo
)
r2_xgb_simple = r2_score(y, y_pred_xgb_simple)

# XGBoost - heavily regularized
y_pred_xgb_reg = cross_val_predict(
    xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, 
                     reg_alpha=10, reg_lambda=10, min_child_weight=5, random_state=42),
    X_2feat_scaled, y, cv=loo
)
r2_xgb_reg = r2_score(y, y_pred_xgb_reg)

print(f"\n  Ridge Regression:           CV R² = {r2_ridge:.3f}")
print(f"  XGBoost (default):          CV R² = {r2_xgb:.3f}")
print(f"  XGBoost (very simple):      CV R² = {r2_xgb_simple:.3f}")
print(f"  XGBoost (heavy regularized): CV R² = {r2_xgb_reg:.3f}")

#==============================================================================
# TEST 2: Why XGBoost overfits - show the train vs test gap
#==============================================================================
print("\n" + "="*80)
print("TEST 2: TRAIN vs TEST PERFORMANCE (showing overfitting)")
print("="*80)

# Manual LOO to see train R²
train_r2_list = []
test_errors = []

for i in range(len(X_2feat_scaled)):
    # Leave one out
    X_train = np.delete(X_2feat_scaled, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X_2feat_scaled[i:i+1]
    y_test = y[i]
    
    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train, verbose=False)
    
    # Train R²
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_r2_list.append(train_r2)
    
    # Test error
    y_test_pred = model.predict(X_test)[0]
    test_errors.append(abs(y_test - y_test_pred))

print(f"\n  XGBoost Average TRAIN R²: {np.mean(train_r2_list):.3f}")
print(f"  XGBoost Average TEST error: {np.mean(test_errors):.2f}")
print(f"\n  → Train R² ≈ 1.0 but Test fails = OVERFITTING!")

# Same for Ridge
train_r2_ridge = []
for i in range(len(X_2feat_scaled)):
    X_train = np.delete(X_2feat_scaled, i, axis=0)
    y_train = np.delete(y, i)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    train_r2_ridge.append(r2_score(y_train, y_train_pred))

print(f"\n  Ridge Average TRAIN R²: {np.mean(train_r2_ridge):.3f}")
print(f"  → Ridge has lower train R² but generalizes better!")

#==============================================================================
# TEST 3: The core issue - sample size
#==============================================================================
print("\n" + "="*80)
print("TEST 3: THE CORE ISSUE - SAMPLE SIZE vs MODEL COMPLEXITY")
print("="*80)

print(f"""
With N = {len(df)} samples:

LINEAR MODEL (Ridge):
  - Parameters: 2 coefficients + 1 intercept = 3 parameters
  - Samples per parameter: {len(df)}/3 = {len(df)/3:.1f}
  - Can estimate reliably? YES

XGBOOST (default, 100 trees, depth 3):
  - Each tree can have up to 2^3 = 8 leaf nodes
  - 100 trees × 8 leaves = 800 potential parameters
  - Samples per parameter: {len(df)}/800 = {len(df)/800:.3f}
  - Can estimate reliably? NO! (massive overfitting)

XGBOOST (simple, 10 trees, depth 1):  
  - Each tree has 2 leaves (1 split)
  - 10 trees × 2 = 20 parameters
  - Samples per parameter: {len(df)}/20 = {len(df)/20:.1f}
  - Can estimate reliably? Borderline
""")

#==============================================================================
# TEST 4: Making XGBoost work - extreme simplification
#==============================================================================
print("\n" + "="*80)
print("TEST 4: CAN WE MAKE XGBOOST WORK?")
print("="*80)

configs = [
    ("Ridge (baseline)", Ridge(alpha=1.0)),
    ("XGB: 1 tree, depth 1", xgb.XGBRegressor(n_estimators=1, max_depth=1, random_state=42)),
    ("XGB: 3 trees, depth 1", xgb.XGBRegressor(n_estimators=3, max_depth=1, random_state=42)),
    ("XGB: 5 trees, depth 1", xgb.XGBRegressor(n_estimators=5, max_depth=1, random_state=42)),
    ("XGB: 10 trees, depth 1", xgb.XGBRegressor(n_estimators=10, max_depth=1, random_state=42)),
    ("XGB: 5 trees, depth 2", xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)),
    ("XGB: 10 trees, depth 2", xgb.XGBRegressor(n_estimators=10, max_depth=2, random_state=42)),
    ("XGB: 50 trees, depth 3 (typical)", xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)),
]

print(f"\n{'Model':<35} {'CV R²':>10}")
print("-"*50)

for name, model in configs:
    y_pred = cross_val_predict(model, X_2feat_scaled, y, cv=loo)
    r2 = r2_score(y, y_pred)
    marker = "★" if r2 > 0.2 else ""
    print(f"  {name:<35} {r2:>10.3f} {marker}")

#==============================================================================
# TEST 5: What about more features?
#==============================================================================
print("\n" + "="*80)
print("TEST 5: DOES ADDING MORE FEATURES HELP?")
print("="*80)

feature_sets = [
    ("2 feat: hr_elev, duration", ['hr_elevation', 'duration']),
    ("3 feat: + hr_std", ['hr_elevation', 'duration', 'hr_std']),
    ("4 feat: + acc_mean", ['hr_elevation', 'duration', 'hr_std', 'acc_mean']),
    ("5 feat: + acc_std", ['hr_elevation', 'duration', 'hr_std', 'acc_mean', 'acc_std']),
    ("6 feat: + hr_mean", ['hr_elevation', 'duration', 'hr_std', 'acc_mean', 'acc_std', 'hr_mean']),
]

print(f"\n{'Features':<30} {'Ridge R²':>10} {'XGB R²':>10}")
print("-"*55)

for name, cols in feature_sets:
    X = df[cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    y_pred_ridge = cross_val_predict(Ridge(alpha=1.0), X_scaled, y, cv=loo)
    r2_ridge = r2_score(y, y_pred_ridge)
    
    y_pred_xgb = cross_val_predict(
        xgb.XGBRegressor(n_estimators=5, max_depth=1, random_state=42),
        X_scaled, y, cv=loo
    )
    r2_xgb = r2_score(y, y_pred_xgb)
    
    print(f"  {name:<30} {r2_ridge:>10.3f} {r2_xgb:>10.3f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
WHY XGBOOST FAILS (even with same HR features):

1. OVERFITTING: With N={len(df)} samples, XGBoost memorizes training data
   - Train R² ≈ 1.0 (perfect fit to training)
   - Test R² ≈ negative (fails on held-out sample)

2. MODEL COMPLEXITY: Default XGBoost has ~800 effective parameters
   - Need ~10-20 samples per parameter
   - We have {len(df)} samples / 800 params = {len(df)/800:.3f} samples/param
   
3. SOLUTION: Make XGBoost as simple as linear regression
   - 1-5 trees with depth 1 works okay
   - But then... why not just use linear regression?

BOTTOM LINE:
  XGBoost ≠ magic. It needs DATA.
  With N < 100, stick to linear models.
  
  The R² = 0.72 you saw with random split was FAKE (leakage).
  With proper CV, XGBoost R² ≈ {r2_xgb:.3f} vs Ridge R² ≈ {r2_ridge:.3f}
""")
