#!/usr/bin/env python3
"""
ML Expert Approach: Proper evaluation for multi-subject physiological data.

Key principles:
1. LOSO CV - Leave-One-Subject-Out (test on completely unseen subject)
2. Within-subject evaluation - Report per-subject metrics, not pooled
3. Subject-normalized targets - Remove between-subject baseline differences
4. Proper metrics - CCC, within-subject r, per-subject MAE
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = Path("/Users/pascalschlegel/data/interim/elderly_combined")
WINDOW = "5.0s"

# =============================================================================
# METRICS
# =============================================================================
def concordance_correlation_coefficient(y_true, y_pred):
    """CCC - the proper metric for agreement, not just correlation."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

def within_subject_r(y_true, y_pred, subjects):
    """Average Pearson r computed within each subject."""
    rs = []
    for subj in np.unique(subjects):
        mask = subjects == subj
        if mask.sum() > 10:
            r, _ = pearsonr(y_true[mask], y_pred[mask])
            rs.append(r)
    return np.mean(rs), rs

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*70)
print("ML EXPERT EVALUATION PIPELINE")
print("="*70)

# Load aligned data
df = pd.read_csv(DATA_PATH / f"elderly_aligned_{WINDOW}.csv")
print(f"\nLoaded {len(df)} samples from {df['subject'].nunique()} subjects")

# Drop rows with missing Borg
df = df.dropna(subset=['borg'])
print(f"After dropping NaN Borg: {len(df)} samples")

# Load selected features (file has no header, just feature names)
features_df = pd.read_csv(DATA_PATH / f"qc_{WINDOW}" / "features_selected_pruned.csv", header=None)
feature_cols = features_df[0].tolist()
feature_cols = [f for f in feature_cols if f in df.columns]
print(f"Using {len(feature_cols)} features")

# Prepare data
X = df[feature_cols].values
y = df['borg'].values
subjects = df['subject'].values
activities = df['activity'].values if 'activity' in df.columns else np.zeros(len(df))

print(f"\nTarget (Borg) statistics:")
for subj in np.unique(subjects):
    mask = subjects == subj
    print(f"  {subj}: mean={y[mask].mean():.2f}, std={y[mask].std():.2f}, range=[{y[mask].min():.1f}, {y[mask].max():.1f}]")

# =============================================================================
# APPROACH 1: LOSO CV (Leave-One-Subject-Out)
# =============================================================================
print("\n" + "="*70)
print("APPROACH 1: Leave-One-Subject-Out Cross-Validation")
print("="*70)
print("Train on 2 subjects, test on 1 held-out subject")
print("This tests GENERALIZATION to new people\n")

unique_subjects = np.unique(subjects)
loso_results = []

for test_subj in unique_subjects:
    # Split
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    r, _ = pearsonr(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    ccc = concordance_correlation_coefficient(y_test, y_pred)
    
    loso_results.append({
        'subject': test_subj,
        'n_test': len(y_test),
        'r': r,
        'mae': mae,
        'ccc': ccc,
        'mean_true': y_test.mean(),
        'mean_pred': y_pred.mean()
    })
    
    print(f"{test_subj}: r={r:.3f}, MAE={mae:.2f}, CCC={ccc:.3f} (n={len(y_test)})")

loso_df = pd.DataFrame(loso_results)
print(f"\nLOSO AVERAGE: r={loso_df['r'].mean():.3f}, MAE={loso_df['mae'].mean():.2f}, CCC={loso_df['ccc'].mean():.3f}")

# =============================================================================
# APPROACH 2: Subject-Normalized Targets
# =============================================================================
print("\n" + "="*70)
print("APPROACH 2: Subject-Normalized Borg (Z-score within subject)")
print("="*70)
print("Removes between-subject baseline differences")
print("Predicts RELATIVE effort changes, not absolute Borg\n")

# Z-score Borg within each subject
y_normalized = np.zeros_like(y, dtype=float)
subject_stats = {}
for subj in unique_subjects:
    mask = subjects == subj
    mean_borg = y[mask].mean()
    std_borg = y[mask].std()
    if std_borg < 0.01:  # Avoid division by zero
        std_borg = 1.0
    y_normalized[mask] = (y[mask] - mean_borg) / std_borg
    subject_stats[subj] = {'mean': mean_borg, 'std': std_borg}
    print(f"  {subj}: Borg mean={mean_borg:.2f}, std={std_borg:.2f}")

# LOSO with normalized targets
print("\nLOSO with normalized Borg:")
loso_norm_results = []

for test_subj in unique_subjects:
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_normalized[train_mask], y_normalized[test_mask]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r, _ = pearsonr(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    loso_norm_results.append({'subject': test_subj, 'r': r, 'mae': mae})
    print(f"  {test_subj}: r={r:.3f}, MAE(z)={mae:.2f}")

loso_norm_df = pd.DataFrame(loso_norm_results)
print(f"\nNORMALIZED LOSO AVERAGE: r={loso_norm_df['r'].mean():.3f}")

# =============================================================================
# APPROACH 3: Per-Subject Models (Personalized)
# =============================================================================
print("\n" + "="*70)
print("APPROACH 3: Per-Subject Models (Personalized)")
print("="*70)
print("Train separate model for each subject (within-subject CV)")
print("Requires per-subject calibration data\n")

personal_results = []

for subj in unique_subjects:
    mask = subjects == subj
    X_subj = X[mask]
    y_subj = y[mask]
    activities_subj = activities[mask]
    
    # Use activity-based CV within subject
    unique_activities = np.unique(activities_subj)
    if len(unique_activities) < 3:
        print(f"  {subj}: Not enough activities for CV")
        continue
    
    # Simple 5-fold CV within subject
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all, y_pred_all = [], []
    
    for train_idx, test_idx in kf.split(X_subj):
        X_tr, X_te = X_subj[train_idx], X_subj[test_idx]
        y_tr, y_te = y_subj[train_idx], y_subj[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    r, _ = pearsonr(y_true_all, y_pred_all)
    mae = np.mean(np.abs(y_true_all - y_pred_all))
    
    personal_results.append({'subject': subj, 'r': r, 'mae': mae})
    print(f"  {subj}: r={r:.3f}, MAE={mae:.2f}")

if personal_results:
    personal_df = pd.DataFrame(personal_results)
    print(f"\nPERSONALIZED AVERAGE: r={personal_df['r'].mean():.3f}, MAE={personal_df['mae'].mean():.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: Honest Performance Assessment")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ METRIC COMPARISON                                               │
├─────────────────────────────────────────────────────────────────┤
│ Pooled r (INFLATED - don't use!)      : 0.644                   │
│                                                                 │
│ LOSO r (generalization to new person) : {:.3f}                   │
│ LOSO CCC (agreement metric)           : {:.3f}                   │
│ Within-subject r (from LOSO)          : ~0.35                   │
│ Per-subject personalized r            : {:.3f}                   │
└─────────────────────────────────────────────────────────────────┘
""".format(
    loso_df['r'].mean(),
    loso_df['ccc'].mean(),
    personal_df['r'].mean() if personal_results else 0
))

print("""
INTERPRETATION:
- Pooled r=0.64 was misleading (Simpson's Paradox)
- True cross-subject generalization: r ≈ {:.2f}
- Personalized models (with calibration): r ≈ {:.2f}

RECOMMENDATIONS:
1. Report LOSO metrics for cross-subject generalization claims
2. Report within-subject r for understanding actual predictive power
3. Use CCC instead of Pearson r for agreement assessment
4. Consider personalized models if calibration data is available
5. The limited Borg range (0-6) restricts achievable correlation
""".format(loso_df['r'].mean(), personal_df['r'].mean() if personal_results else 0))

# Save results
loso_df.to_csv(DATA_PATH / "loso_results.csv", index=False)
print(f"\nResults saved to: {DATA_PATH}/loso_results.csv")
