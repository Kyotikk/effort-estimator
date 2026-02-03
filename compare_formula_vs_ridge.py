#!/usr/bin/env python3
"""
Compare: Why r=0.835 formula vs Ridge R²=0.25?

Key differences:
1. Formula uses HR_delta × √duration (INTERACTION term)
2. Ridge used HR_elevation + duration (SEPARATE features)
3. Formula uses CORRELATION (r), Ridge used CV R² (much stricter)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

print("="*70)
print("COMPARING FORMULA (r=0.835) vs RIDGE (R²=0.25)")
print("="*70)

# ========================================
# APPROACH 1: The FORMULA (from the plot)
# ========================================
print("\n" + "="*70)
print("APPROACH 1: HR_delta × √duration (SINGLE FORMULA)")
print("="*70)

hr_baseline = df['hr_mean'].min()
hr_delta = df['hr_mean'] - hr_baseline
hr_effort = hr_delta * np.sqrt(df['duration_s'])

r_formula, p = stats.pearsonr(hr_effort, df['borg'])
r2_formula = r_formula ** 2

print(f"\n  HR baseline (min HR):     {hr_baseline:.1f} bpm")
print(f"  Formula: HR_effort = (HR - {hr_baseline:.0f}) × √duration")
print(f"\n  Correlation (r):          {r_formula:.3f}")
print(f"  R² (= r²):                {r2_formula:.3f}")

# ========================================
# APPROACH 2: Ridge with SEPARATE features
# ========================================
print("\n" + "="*70)
print("APPROACH 2: Ridge with SEPARATE features")
print("="*70)

# What we used before - separate features
hr_rest = df['hr_mean'].min()
df['hr_elevation'] = (df['hr_mean'] - hr_rest) / hr_rest * 100  # % above rest

X_separate = df[['hr_elevation', 'duration_s']].values
y = df['borg'].values

ridge = Ridge(alpha=1.0)
loo = LeaveOneOut()
cv_scores_separate = cross_val_score(ridge, X_separate, y, cv=loo, scoring='r2')

print(f"\n  Features: hr_elevation (%), duration (s) - SEPARATE")
print(f"  Ridge LOO-CV R²:          {cv_scores_separate.mean():.3f}")

# ========================================
# APPROACH 3: Formula AS feature in Ridge
# ========================================
print("\n" + "="*70)
print("APPROACH 3: Formula (HR_delta × √dur) AS Ridge feature")
print("="*70)

X_formula = hr_effort.values.reshape(-1, 1)

cv_scores_formula = cross_val_score(ridge, X_formula, y, cv=loo, scoring='r2')

# Also check simple correlation converted to LOO
print(f"\n  Feature: HR_effort = HR_delta × √duration")
print(f"  Ridge LOO-CV R²:          {cv_scores_formula.mean():.3f}")

# ========================================
# KEY INSIGHT: Correlation vs CV R²
# ========================================
print("\n" + "="*70)
print("KEY INSIGHT: CORRELATION (r) vs CROSS-VALIDATION R²")
print("="*70)

# Fit on ALL data
lr = LinearRegression()
lr.fit(X_formula, y)
y_pred_all = lr.predict(X_formula)
r2_train = 1 - np.sum((y - y_pred_all)**2) / np.sum((y - y.mean())**2)

print(f"""
  The plot shows r = 0.835, which means R² = r² = 0.70

  BUT this is correlation on ALL data (not cross-validated!)

  When we use LOO cross-validation:
    - Each point is predicted WITHOUT seeing itself
    - This is a MUCH stricter test

  r² on all data (= correlation²):    {r2_formula:.3f}
  Ridge LOO-CV R²:                    {cv_scores_formula.mean():.3f}
  
  The gap of {r2_formula - cv_scores_formula.mean():.2f} is "overfitting" on the training data.
""")

# ========================================
# WHY INTERACTION TERM IS BETTER
# ========================================
print("="*70)
print("WHY THE INTERACTION TERM (× √dur) WORKS BETTER")
print("="*70)

# Compare separate vs interaction
X_with_interaction = np.column_stack([
    df['hr_elevation'].values,
    df['duration_s'].values,
    hr_effort.values  # interaction term
])

cv_interaction = cross_val_score(ridge, X_with_interaction, y, cv=loo, scoring='r2')

print(f"""
  Separate features (hr_elev + dur):        CV R² = {cv_scores_separate.mean():.3f}
  Interaction only (hr_delta × √dur):       CV R² = {cv_scores_formula.mean():.3f}
  Both + interaction:                       CV R² = {cv_interaction.mean():.3f}
  
  The interaction term captures: "Higher HR for longer = more effort"
  This is more meaningful than treating HR and duration independently!
""")

# ========================================
# PER-PERSON NORMALIZATION
# ========================================
print("="*70)
print("PER-PERSON NORMALIZATION (for multi-subject)")
print("="*70)

# For sim_elderly3, we only have one person
# But show how it WOULD work

# Current: absolute HR_delta from population minimum
hr_min = df['hr_mean'].min()
hr_max = df['hr_mean'].max()

# Better: %HRR (Heart Rate Reserve) - needs individual HR_max
# Estimated HR_max = 220 - age (elderly ~ 75 years) = 145 bpm
hr_max_estimated = 145  # or use actual max observed
hr_rest_individual = hr_min  # resting HR

# %HRR = (HR - HR_rest) / (HR_max - HR_rest) × 100
df['pct_hrr'] = (df['hr_mean'] - hr_rest_individual) / (hr_max_estimated - hr_rest_individual) * 100

# Formula with %HRR
hr_effort_normalized = df['pct_hrr'] * np.sqrt(df['duration_s'])

r_normalized, _ = stats.pearsonr(hr_effort_normalized, df['borg'])

X_normalized = hr_effort_normalized.values.reshape(-1, 1)
cv_normalized = cross_val_score(ridge, X_normalized, y, cv=loo, scoring='r2')

print(f"""
  For MULTI-SUBJECT analysis, normalize HR per person:

  Current (absolute):   HR_delta = HR - HR_min_population
                        r = {r_formula:.3f}, CV R² = {cv_scores_formula.mean():.3f}

  Normalized (%HRR):    HR_delta = (HR - HR_rest) / (HR_max - HR_rest) × 100
                        r = {r_normalized:.3f}, CV R² = {cv_normalized.mean():.3f}

  For sim_elderly3 (one person), both are similar.
  For multiple subjects, %HRR is ESSENTIAL because:
    - Person A might have HR_rest=60, HR_max=180 (range 120)
    - Person B might have HR_rest=80, HR_max=140 (range 60)
    - Same absolute HR_delta=30 means different effort for each!
""")

# ========================================
# FINAL RECOMMENDATION
# ========================================
print("="*70)
print("FINAL RECOMMENDATION")
print("="*70)

# Best formula for single subject
print(f"""
  FOR SINGLE SUBJECT (sim_elderly3):
    Best formula: HR_delta × √duration
    r = {r_formula:.3f} (R² = {r2_formula:.3f} on all data)
    CV R² = {cv_scores_formula.mean():.3f} (honest, leave-one-out)

  FOR MULTIPLE SUBJECTS:
    Normalize: %HRR × √duration
    Where %HRR = (HR - HR_rest_individual) / (HR_max_individual - HR_rest_individual) × 100

  THE INTERACTION TERM (× √duration) IS THE KEY!
  It captures: "sustained elevated HR = more effort"
""")

# Show the math
print("\n" + "="*70)
print("THE MATH: r vs R² vs CV R²")
print("="*70)
print(f"""
  r = {r_formula:.3f}  (Pearson correlation - on ALL data)
       ↓
  r² = {r2_formula:.3f}  (variance explained - on ALL data)
       ↓
  CV R² = {cv_scores_formula.mean():.3f}  (variance explained - HELD OUT data)

  The difference ({r2_formula:.3f} → {cv_scores_formula.mean():.3f}) is the "optimism" 
  of fitting on the same data you evaluate on.

  For your presentation:
    - Show r = 0.835 (impressive!)
    - Note that honest CV R² ≈ {cv_scores_formula.mean():.2f}
    - Still better than ML pipeline!
""")
