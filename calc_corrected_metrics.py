#!/usr/bin/env python3
"""Corrected metrics table - Borg points vs Categories"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression

df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv').dropna(subset=['borg'])
feat_cols = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv', header=None)[0].tolist()
feat_cols = [c for c in feat_cols if c in df.columns]

X = df[feat_cols].values
y = df['borg'].values
subjects = df['subject'].values

def to_cat(b):
    if b <= 2: return 0  # LOW
    elif b <= 4: return 1  # MOD
    else: return 2  # HIGH

def eval_metrics(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]
    mae = np.mean(np.abs(y_true - y_pred))
    
    # ±1 Borg POINT
    within_1pt = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    within_2pt = np.mean(np.abs(y_true - y_pred) <= 2) * 100
    
    # Category
    true_cat = np.array([to_cat(b) for b in y_true])
    pred_cat = np.array([to_cat(b) for b in y_pred])
    cat_exact = np.mean(true_cat == pred_cat) * 100
    cat_within1 = np.mean(np.abs(true_cat - pred_cat) <= 1) * 100
    
    return r, mae, within_1pt, within_2pt, cat_exact, cat_within1

print('='*90)
print('CORRECTED TABLE: Borg Points vs Categories')
print('='*90)
print()
print(f"{'Method':<35} {'r':>6} {'MAE':>6} {'±1pt':>7} {'±2pt':>7} {'CatExact':>9} {'±1Cat':>7}")
print(f"{'':35} {'':>6} {'':>6} {'(Borg)':>7} {'(Borg)':>7} {'(L/M/H)':>9} {'(L/M/H)':>7}")
print('-'*90)

# Method 1: Raw LOSO
all_p, all_t = [], []
for s in np.unique(subjects):
    m = subjects != s
    sc = StandardScaler()
    mdl = Ridge(alpha=1.0)
    mdl.fit(sc.fit_transform(X[m]), y[m])
    all_p.extend(mdl.predict(sc.transform(X[~m])))
    all_t.extend(y[~m])
r, mae, w1, w2, ce, c1 = eval_metrics(np.array(all_t), np.array(all_p))
print(f"{'1. Raw LOSO':<35} {r:>6.3f} {mae:>6.2f} {w1:>6.1f}% {w2:>6.1f}% {ce:>8.1f}% {c1:>6.1f}%")

# Method 3: Linear Calibration
all_p, all_t = [], []
for s in np.unique(subjects):
    m = subjects != s
    sc = StandardScaler()
    mdl = Ridge(alpha=1.0)
    mdl.fit(sc.fit_transform(X[m]), y[m])
    y_raw = mdl.predict(sc.transform(X[~m]))
    y_te = y[~m]
    n_cal = max(20, int(0.20 * len(y_te)))
    cal = LinearRegression()
    cal.fit(y_raw[:n_cal].reshape(-1,1), y_te[:n_cal])
    y_cal = cal.predict(y_raw.reshape(-1,1))
    all_p.extend(y_cal[n_cal:])
    all_t.extend(y_te[n_cal:])
r, mae, w1, w2, ce, c1 = eval_metrics(np.array(all_t), np.array(all_p))
print(f"{'3. LOSO + Linear Calibration':<35} {r:>6.3f} {mae:>6.2f} {w1:>6.1f}% {w2:>6.1f}% {ce:>8.1f}% {c1:>6.1f}%")

# Method 4: Within-Subject
all_p, all_t = [], []
for s in np.unique(subjects):
    sm = subjects == s
    Xs, ys = X[sm], y[sm]
    n = len(ys)
    nt = int(0.5 * n)
    sc = StandardScaler()
    mdl = Ridge(alpha=1.0)
    mdl.fit(sc.fit_transform(Xs[:nt]), ys[:nt])
    all_p.extend(mdl.predict(sc.transform(Xs[nt:])))
    all_t.extend(ys[nt:])
r, mae, w1, w2, ce, c1 = eval_metrics(np.array(all_t), np.array(all_p))
print(f"{'4. Within-Subject (50/50)':<35} {r:>6.3f} {mae:>6.2f} {w1:>6.1f}% {w2:>6.1f}% {ce:>8.1f}% {c1:>6.1f}%")

print('='*90)
print()
print('LEGEND:')
print('  ±1pt (Borg)  = Prediction within 1 actual Borg point (e.g., pred=3 when actual=4)')
print('  ±1Cat (L/M/H) = Prediction within 1 category (LOW≤2, MOD=3-4, HIGH≥5)')
print()
print('INTERPRETATION:')
print('  - ±1 Borg point accuracy is low (~33-42%)')
print('  - Category accuracy is higher (~50-55% exact, ~95%+ within ±1 category)')
print('  - For thesis: report BOTH metrics to be transparent')
