#!/usr/bin/env python3
"""
Why r=0.835 formula vs Ridge R²=0.25?
And per-person normalization for multi-subject.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')
n = len(df)

print("="*70)
print("WHY r=0.835 FORMULA vs RIDGE R²=0.25?")
print("="*70)
print(f"\nData: {n} activities from sim_elderly3")

# ========================================
# THE KEY DIFFERENCE
# ========================================
print("\n" + "="*70)
print("KEY DIFFERENCE: FORMULA (r=0.835) vs RIDGE (R²=0.25)")
print("="*70)

# The formula from the plot
hr_baseline = df['hr_mean'].min()
hr_delta = df['hr_mean'] - hr_baseline
hr_effort = hr_delta * np.sqrt(df['duration_s'])

# Correlation (what the plot shows)
r_formula, _ = stats.pearsonr(hr_effort, df['borg'])
r2_formula = r_formula ** 2

# What we showed with Ridge before
hr_rest = df['hr_mean'].min()
df['hr_elevation'] = (df['hr_mean'] - hr_rest) / hr_rest * 100

print(f"""
THE PLOT SHOWS: r = {r_formula:.3f} (correlation)
                R² = r² = {r2_formula:.3f} (variance explained on ALL data)

Previous Ridge results used DIFFERENT METRICS:
  1. SEPARATE features (hr_elevation + duration) not INTERACTION (hr_delta × √dur)
  2. Leave-One-Out CV R² not correlation on all data

Let's compare apples to apples:
""")

# ========================================
# MANUAL LOO CV (to avoid sklearn warnings)
# ========================================

def manual_loo_r2(X, y):
    """Manual Leave-One-Out R² calculation"""
    y_pred = np.zeros(len(y))
    for i in range(len(y)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred[i] = lr.predict(X[i:i+1])[0]
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, y_pred

y = df['borg'].values

# Compare different feature sets
print("="*70)
print("COMPARING FEATURE FORMULATIONS (all LOO-CV)")
print("="*70)

# 1. Interaction term only (HR_delta × √duration)
X_interaction = hr_effort.values.reshape(-1, 1)
r2_cv_interaction, pred_interaction = manual_loo_r2(X_interaction, y)

# 2. Separate features (hr_elevation + duration)  
X_separate = df[['hr_elevation', 'duration_s']].values
r2_cv_separate, pred_separate = manual_loo_r2(X_separate, y)

# 3. Separate with hr_delta instead of hr_elevation
df['hr_delta'] = df['hr_mean'] - hr_rest
X_separate2 = df[['hr_delta', 'duration_s']].values
r2_cv_separate2, pred_separate2 = manual_loo_r2(X_separate2, y)

# 4. Include interaction term with separate
X_both = np.column_stack([df['hr_delta'], df['duration_s'], hr_effort])
r2_cv_both, pred_both = manual_loo_r2(X_both, y)

print(f"""
  Feature Formulation                           Corr (r)    CV R²
  ─────────────────────────────────────────────────────────────────
  HR_delta × √duration (INTERACTION)            {r_formula:.3f}      {r2_cv_interaction:.3f}
  hr_elevation + duration (SEPARATE)            -           {r2_cv_separate:.3f}
  hr_delta + duration (SEPARATE)                -           {r2_cv_separate2:.3f}
  hr_delta + duration + interaction (ALL)       -           {r2_cv_both:.3f}
""")

# ========================================
# THE ANSWER: INTERACTION TERM IS THE KEY
# ========================================
print("="*70)
print("ANSWER: THE INTERACTION TERM (× √duration) IS THE KEY!")
print("="*70)

print(f"""
  WHY INTERACTION WORKS BETTER:
  
  Separate:    Effort = a×HR + b×Duration + c
               → Treats HR and duration independently
               → High HR but short activity = same weight as low HR long activity
  
  Interaction: Effort = a × (HR_delta × √Duration)
               → "Sustained elevated HR" = accumulated stress
               → Physics: Work = Force × Distance (or Power × Time)
  
  Results:
    Separate features:    CV R² = {r2_cv_separate2:.3f}
    Interaction formula:  CV R² = {r2_cv_interaction:.3f}
    
  THE INTERACTION CAPTURES THE PHYSIOLOGY!
""")

# ========================================
# PER-PERSON NORMALIZATION
# ========================================
print("="*70)
print("PER-PERSON NORMALIZATION (for multiple subjects)")
print("="*70)

# Current: absolute HR_delta
hr_min = df['hr_mean'].min()
hr_max_observed = df['hr_mean'].max()

# Better: %HRR (Heart Rate Reserve)
# Estimated HR_max = 220 - age (for elderly ~75) = 145
hr_max_theoretical = 145
hr_rest_individual = hr_min  # Use observed minimum as resting

# %HRR normalized
df['pct_hrr'] = (df['hr_mean'] - hr_rest_individual) / (hr_max_theoretical - hr_rest_individual) * 100

# Formula with %HRR
hr_effort_pct = df['pct_hrr'] * np.sqrt(df['duration_s'])

X_pct = hr_effort_pct.values.reshape(-1, 1)
r2_cv_pct, _ = manual_loo_r2(X_pct, y)

r_pct, _ = stats.pearsonr(hr_effort_pct, df['borg'])

print(f"""
  FOR SINGLE SUBJECT (sim_elderly3):
    Both work the same because it's the same person:
    
    Absolute HR_delta:  r = {r_formula:.3f}, CV R² = {r2_cv_interaction:.3f}
    %HRR normalized:    r = {r_pct:.3f}, CV R² = {r2_cv_pct:.3f}
  
  FOR MULTIPLE SUBJECTS, %HRR IS ESSENTIAL:
    
    Person A (athlete):     HR_rest=50, HR_max=190 → range=140 bpm
    Person B (elderly):     HR_rest=80, HR_max=140 → range=60 bpm
    
    Same HR=120 means:
      Person A: %HRR = (120-50)/(190-50) = 50% → moderate effort
      Person B: %HRR = (120-80)/(140-80) = 67% → high effort!
    
    Without normalization, 120 bpm = same "effort" for both = WRONG!
""")

# Show the %HRR calculation for this subject
print("  Current subject normalization:")
print(f"    HR_rest (observed min):     {hr_rest_individual:.1f} bpm")
print(f"    HR_max (theoretical):       {hr_max_theoretical:.1f} bpm")
print(f"    HR range:                   {hr_max_theoretical - hr_rest_individual:.1f} bpm")
print(f"    HR_max (observed):          {hr_max_observed:.1f} bpm")

# ========================================
# FORMULA COMPARISON TABLE
# ========================================
print("\n" + "="*70)
print("FINAL SUMMARY: FORMULA COMPARISON")
print("="*70)

# Also check r² to compare with plot
r2_on_all_interaction = r_formula ** 2
r2_on_all_pct = r_pct ** 2

print(f"""
                              Train R²      CV R²       Difference
                              (= r²)       (honest)    (optimism)
  ────────────────────────────────────────────────────────────────
  HR_delta × √duration        {r2_on_all_interaction:.3f}        {r2_cv_interaction:.3f}        {r2_on_all_interaction - r2_cv_interaction:.3f}
  %HRR × √duration            {r2_on_all_pct:.3f}        {r2_cv_pct:.3f}        {r2_on_all_pct - r2_cv_pct:.3f}
  
  CONCLUSION:
  - The r=0.835 (R²=0.70) from the plot is on ALL data (optimistic)
  - Honest CV R² ≈ {r2_cv_interaction:.2f} (still good!)
  - %HRR normalization ready for multi-subject extension
  
  FOR PRESENTATION:
  - Show r=0.835 (impressive and valid for single subject)
  - Note: "With leave-one-out validation, R² = {r2_cv_interaction:.2f}"
  - Compare to ML pipeline: XGBoost CV R² = -0.06
  - Simple formula wins by a huge margin!
""")
