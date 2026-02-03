#!/usr/bin/env python3
"""
Analysis using the 48 PRUNED/SELECTED features (not all 275)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')

# Load the 48 selected/pruned features
with open('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv') as f:
    selected_features = [line.strip() for line in f if line.strip()]

print(f"Total selected features: {len(selected_features)}")
print(f"  EDA: {len([f for f in selected_features if f.startswith('eda_')])}")
print(f"  IMU: {len([f for f in selected_features if f.startswith('acc_')])}")
print(f"  PPG: {len([f for f in selected_features if f.startswith('ppg_')])}")

# Check which features exist in the dataset
available = [f for f in selected_features if f in df.columns]
missing = [f for f in selected_features if f not in df.columns]
print(f"\nAvailable in dataset: {len(available)}")
if missing:
    print(f"Missing: {missing}")

# Use only the selected features
feature_cols = available

# Clean
df = df.dropna(subset=['borg'])
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
for c in feature_cols:
    df[c] = df[c].fillna(df[c].median())

print(f"\nFinal: {len(df)} samples × {len(feature_cols)} features")

X = df[feature_cols].values
y = df['borg'].values
subjects = df['subject'].values

def to_cat(b):
    if b <= 2: return 0  # LOW
    elif b <= 4: return 1  # MOD
    else: return 2  # HIGH

def to_cat_name(c):
    return ['LOW', 'MOD', 'HIGH'][c]

logo = LeaveOneGroupOut()

# ============================================================
# METHOD 1: Raw Cross-Subject
# ============================================================
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

# ============================================================
# METHOD 3: With Calibration
# ============================================================
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

# ============================================================
# METHOD 4: Within-Subject
# ============================================================
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

# ============================================================
# RESULTS
# ============================================================
def calc_all_metrics(y_true, y_pred, name):
    true_cats = [to_cat(b) for b in y_true]
    pred_cats = [to_cat(b) for b in y_pred]
    
    # Correlation
    r = np.corrcoef(y_true, y_pred)[0, 1]
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    # Category accuracy
    exact = sum(1 for t, p in zip(true_cats, pred_cats) if t == p)
    within_1 = sum(1 for t, p in zip(true_cats, pred_cats) if abs(t - p) <= 1)
    
    # Confusion matrix
    labels = [0, 1, 2]
    cm = confusion_matrix(true_cats, pred_cats, labels=labels)
    
    # Per-category recall
    recalls = {}
    for i, cat in enumerate(['LOW', 'MOD', 'HIGH']):
        total = cm[i].sum()
        correct = cm[i, i]
        recalls[cat] = 100 * correct / total if total > 0 else 0
    
    # Dangerous errors
    dangerous = cm[0, 2] + cm[2, 0]
    dangerous_pct = 100 * dangerous / len(true_cats)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Correlation r:     {r:.3f}")
    print(f"MAE:               {mae:.2f} Borg points")
    print(f"\nCategory Accuracy:")
    print(f"  Exact:           {100*exact/len(true_cats):.1f}%")
    print(f"  Within ±1:       {100*within_1/len(true_cats):.1f}%")
    print(f"\nPer-Category Recall:")
    print(f"  LOW:             {recalls['LOW']:.1f}%")
    print(f"  MOD:             {recalls['MOD']:.1f}%")
    print(f"  HIGH:            {recalls['HIGH']:.1f}%")
    print(f"\nDangerous (LOW↔HIGH): {dangerous_pct:.1f}%")
    
    return {
        'r': r, 'mae': mae, 
        'exact': 100*exact/len(true_cats), 
        'within_1': 100*within_1/len(true_cats),
        'recalls': recalls,
        'dangerous': dangerous_pct
    }

print("\n" + "="*60)
print("RESULTS WITH 48 SELECTED FEATURES")
print("="*60)

r1 = calc_all_metrics(y_true1, y_pred1, "METHOD 1: Cross-Subject (Raw)")
r3 = calc_all_metrics(y_true3, y_pred3, "METHOD 3: With Calibration")
r4 = calc_all_metrics(y_true4, y_pred4, "METHOD 4: Within-Subject")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"\n{'Method':<30} | {'r':>6} | {'MAE':>5} | {'Exact':>6} | {'±1':>6} | {'Danger':>7}")
print("-" * 75)
print(f"{'Random baseline':<30} |    -   |   -   |  33.3% |  66.7% |    -   ")
print(f"{'Method 1 (Raw)':<30} | {r1['r']:>5.2f}  | {r1['mae']:>4.2f}  | {r1['exact']:>5.1f}% | {r1['within_1']:>5.1f}% | {r1['dangerous']:>5.1f}% ")
print(f"{'Method 3 (Calibration)':<30} | {r3['r']:>5.2f}  | {r3['mae']:>4.2f}  | {r3['exact']:>5.1f}% | {r3['within_1']:>5.1f}% | {r3['dangerous']:>5.1f}% ")
print(f"{'Method 4 (Within-Subject)':<30} | {r4['r']:>5.2f}  | {r4['mae']:>4.2f}  | {r4['exact']:>5.1f}% | {r4['within_1']:>5.1f}% | {r4['dangerous']:>5.1f}% ")
