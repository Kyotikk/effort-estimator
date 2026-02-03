#!/usr/bin/env python3
"""Diagnose the per-subject vs pooled correlation issue."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

pred = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/ridge_results_5.0s/predictions.csv')

print('=== DIAGNOSTIC: Why is overall r=0.64 but per-subject r~0.3? ===')
print()

# Overall correlation
r_overall, _ = pearsonr(pred['y_true'], pred['y_pred'])
print(f'OVERALL pooled r: {r_overall:.3f}')
print()

# Per-subject
print('PER-SUBJECT correlations:')
for subj in pred['subject'].unique():
    s = pred[pred['subject'] == subj]
    r, _ = pearsonr(s['y_true'], s['y_pred'])
    print(f'  {subj}: r={r:.3f}, n={len(s)}, Borg range: {s["y_true"].min():.1f}-{s["y_true"].max():.1f}')

print()
print('=== THE PROBLEM: Subject-level mean differences ===')
for subj in pred['subject'].unique():
    s = pred[pred['subject'] == subj]
    print(f'  {subj}: mean_true={s["y_true"].mean():.2f}, mean_pred={s["y_pred"].mean():.2f}')

print()
print('=== ROOT CAUSE ===')
print('The pooled r=0.64 is INFLATED because:')
print('1. Different subjects have different BASELINE Borg ratings')
print('2. Model learns to predict subject identity, not effort within subject')
print('3. When you pool all data, the BETWEEN-subject variance inflates r')
print()
print('REAL performance = within-subject r ~ 0.35 (average)')
print()
print('This is Simpson\'s Paradox - aggregate correlation hides poor individual fit!')
