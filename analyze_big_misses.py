#!/usr/bin/env python3
"""Analyze why 13% of predictions miss by 2 categories"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')

exclude_cols = ['subject', 'borg', 't_center', 'window_start', 'window_end', 'unix_time', 'Unnamed: 0', 'index']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64'] and df[c].notna().sum() > 100]
valid_features = [c for c in feature_cols if df[c].isna().mean() < 0.5]

df_model = df.dropna(subset=['borg'])[['subject', 'borg'] + valid_features].dropna()

X = df_model[valid_features].values
y = df_model['borg'].values
groups = df_model['subject'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LOSO prediction
logo = LeaveOneGroupOut()
model = Ridge(alpha=1.0)
y_pred = cross_val_predict(model, X_scaled, y, cv=logo, groups=groups)

# Categorize
def to_cat(b):
    if b <= 2: return 0  # LOW
    elif b <= 4: return 1  # MODERATE
    else: return 2  # HIGH

y_cat = np.array([to_cat(b) for b in y])
y_pred_cat = np.array([to_cat(b) for b in y_pred])

# Find the 13% that are 2 categories off
big_miss = np.abs(y_cat - y_pred_cat) == 2
print(f'Big misses (2 categories off): {big_miss.sum()} / {len(y)} = {big_miss.mean()*100:.1f}%')
print()

# What kind of errors?
low_pred_high = (y_cat == 0) & (y_pred_cat == 2)
high_pred_low = (y_cat == 2) & (y_pred_cat == 0)

print('BIG MISS BREAKDOWN:')
print('='*60)
print(f'  LOW actual -> HIGH predicted: {low_pred_high.sum()}')
print(f'  HIGH actual -> LOW predicted: {high_pred_low.sum()}')

# By subject
print()
print('Big misses by subject:')
for subj in np.unique(groups):
    mask = groups == subj
    subj_misses = big_miss[mask].sum()
    subj_total = mask.sum()
    pct = subj_misses/subj_total*100
    print(f'  {subj.replace("sim_","")}: {subj_misses}/{subj_total} = {pct:.1f}%')

# Root cause: subject mean differences
print()
print('='*60)
print('ROOT CAUSE: Subject Borg Means')
print('='*60)
for subj in np.unique(groups):
    mask = groups == subj
    mean_borg = y[mask].mean()
    cat = 'LOW' if mean_borg <= 2 else 'MODERATE' if mean_borg <= 4 else 'HIGH'
    print(f'  {subj.replace("sim_","")}: mean Borg = {mean_borg:.2f} -> {cat}')

# The explanation
print()
print('='*60)
print('EXPLANATION')
print('='*60)
print('''
WHY 13% ARE 2 CATEGORIES OFF:

1. P5 (elderly5) has mean Borg = 1.14 (ALL LOW effort)
   - When tested on P5, model trained on P1-P4 (mean ~3.5) 
   - Model predicts MODERATE/HIGH for P5's LOW effort activities

2. P4 (elderly4) has mean Borg = 4.19 (HIGH effort tendency)
   - When tested on P4, model trained on others predicts lower
   - P4's HIGH rated as LOW/MODERATE

This is the SUBJECTIVITY problem:
- P5 rates everything as LOW (max 2.5)
- P4 rates same activities as HIGH (mean 4.2)

CONCLUSION:
The 13% "big misses" are mostly when testing on OUTLIER subjects
(P5 with very low ratings, P4 with high ratings).

With personalization, these errors would disappear because
the model learns each person's individual calibration.
''')
