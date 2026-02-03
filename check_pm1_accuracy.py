#!/usr/bin/env python3
"""Calculate ±1 category accuracy for all methods"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')

# Clean
meta_cols = ['t_center', 'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 
             'win_sec_r', 'borg', 'modality', 'subject', 'ppg_green_lf_power',
             'ppg_green_hf_power', 'ppg_green_total_power', 'ppg_infra_lf_power',
             'ppg_infra_hf_power', 'ppg_infra_total_power']
feature_cols = [c for c in df.columns if c not in meta_cols]
df = df.dropna(subset=['borg'])
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
valid_features = [c for c in feature_cols if df[c].isna().sum() / len(df) < 0.3]
for c in valid_features:
    df[c] = df[c].fillna(df[c].median())

X = df[valid_features].values
y = df['borg'].values
subjects = df['subject'].values

def to_cat(b):
    if b <= 2: return 0  # LOW
    elif b <= 4: return 1  # MOD
    else: return 2  # HIGH

logo = LeaveOneGroupOut()

# Method 1: Raw cross-subject
y_true1, y_pred1 = [], []
for train_idx, test_idx in logo.split(X, y, subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_true1.extend(y_test)
    y_pred1.extend(model.predict(X_test_s))

# Method 3: Calibration
y_true3, y_pred3 = [], []
for train_idx, test_idx in logo.split(X, y, subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_subjs = subjects[train_idx]
    
    X_train_norm = np.zeros_like(X_train, dtype=float)
    for subj in np.unique(train_subjs):
        mask = train_subjs == subj
        scaler = StandardScaler()
        X_train_norm[mask] = scaler.fit_transform(X_train[mask])
    
    y_train_norm = np.zeros_like(y_train, dtype=float)
    for subj in np.unique(train_subjs):
        mask = train_subjs == subj
        y_train_norm[mask] = (y_train[mask] - y_train[mask].mean()) / (y_train[mask].std() + 1e-8)
    
    test_scaler = StandardScaler()
    X_test_norm = test_scaler.fit_transform(X_test)
    test_mu, test_sigma = y_test.mean(), y_test.std() + 1e-8
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_norm, y_train_norm)
    pred_norm = model.predict(X_test_norm)
    pred = pred_norm * test_sigma + test_mu
    
    y_true3.extend(y_test)
    y_pred3.extend(pred)

# Method 4: Within-subject
y_true4, y_pred4 = [], []
for subj in np.unique(subjects):
    mask = subjects == subj
    X_subj, y_subj = X[mask], y[mask]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X_subj):
        X_train, X_test = X_subj[train_idx], X_subj[test_idx]
        y_train, y_test = y_subj[train_idx], y_subj[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        y_true4.extend(y_test)
        y_pred4.extend(model.predict(X_test_s))

# Calculate metrics
def calc_metrics(y_true, y_pred, name):
    true_cats = [to_cat(b) for b in y_true]
    pred_cats = [to_cat(b) for b in y_pred]
    exact = sum(1 for t, p in zip(true_cats, pred_cats) if t == p)
    within_1 = sum(1 for t, p in zip(true_cats, pred_cats) if abs(t - p) <= 1)
    print(f"{name}:")
    print(f"  Exact category:  {100*exact/len(true_cats):.1f}%")
    print(f"  Within ±1 cat:   {100*within_1/len(true_cats):.1f}%")
    print()

print("="*50)
print("CATEGORY ACCURACY: EXACT vs ±1")
print("="*50)
print()
calc_metrics(y_true1, y_pred1, "METHOD 1 (Raw Cross-Subject)")
calc_metrics(y_true3, y_pred3, "METHOD 3 (With Calibration)")
calc_metrics(y_true4, y_pred4, "METHOD 4 (Within-Subject)")
