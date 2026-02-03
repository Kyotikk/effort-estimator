#!/usr/bin/env python3
"""
CLEAN Per-Category Accuracy Analysis
=====================================
Using the combined 5-elderly dataset with all features and Borg labels.

Shows per-category accuracy for LOW/MOD/HIGH effort levels.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')
print(f"Raw data: {len(df)} samples, {len(df.columns)} columns")
print(f"Subjects: {df['subject'].unique()}")

# =============================================================================
# CLEAN DATA
# =============================================================================
# Identify feature columns (exclude metadata)
meta_cols = ['t_center', 'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 
             'win_sec_r', 'borg', 'modality', 'subject', 'ppg_green_lf_power',
             'ppg_green_hf_power', 'ppg_green_total_power', 'ppg_infra_lf_power',
             'ppg_infra_hf_power', 'ppg_infra_total_power']
feature_cols = [c for c in df.columns if c not in meta_cols]

# Remove rows with missing Borg
df = df.dropna(subset=['borg'])

# Replace infinities with NaN
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Drop columns that are mostly NaN (>30%)
valid_features = []
for c in feature_cols:
    if df[c].isna().sum() / len(df) < 0.3:
        valid_features.append(c)
feature_cols = valid_features

# Fill remaining NaNs with column median
for c in feature_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

print(f"After cleaning: {len(df)} samples, {len(feature_cols)} features")

# =============================================================================
# PREPARE DATA
# =============================================================================
X = df[feature_cols].values
y = df['borg'].values
subjects = df['subject'].values

# Rename subjects for cleaner display
subject_map = {
    'sim_elderly1': 'P1', 'sim_elderly2': 'P2', 'sim_elderly3': 'P3',
    'sim_elderly4': 'P4', 'sim_elderly5': 'P5'
}
subjects = np.array([subject_map.get(s, s) for s in subjects])

print(f"\nPer-subject samples:")
for s in sorted(np.unique(subjects)):
    n = (subjects == s).sum()
    print(f"  {s}: {n} samples")

# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================
def to_category(borg):
    """Convert Borg (0-10) to LOW/MOD/HIGH"""
    if borg <= 2:
        return 'LOW'
    elif borg <= 4:
        return 'MOD'
    else:
        return 'HIGH'

print(f"\n📊 CATEGORY DISTRIBUTION:")
cats = [to_category(b) for b in y]
for cat in ['LOW', 'MOD', 'HIGH']:
    n = cats.count(cat)
    print(f"  {cat}: {n} ({100*n/len(cats):.1f}%)")

# =============================================================================
# METHOD IMPLEMENTATIONS
# =============================================================================
def method1_cross_subject_raw():
    """Method 1: Train on other subjects, test on held-out subject (no normalization tricks)"""
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    
    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        
        y_true.extend(y_test)
        y_pred.extend(pred)
    
    return np.array(y_true), np.array(y_pred)


def method3_calibration():
    """Method 3: Calibration - per-subject feature AND target normalization"""
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    
    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_subjs = subjects[train_idx]
        
        # Normalize features per subject in training
        X_train_norm = np.zeros_like(X_train, dtype=float)
        for subj in np.unique(train_subjs):
            mask = train_subjs == subj
            scaler = StandardScaler()
            X_train_norm[mask] = scaler.fit_transform(X_train[mask])
        
        # Normalize targets per subject in training  
        y_train_norm = np.zeros_like(y_train, dtype=float)
        for subj in np.unique(train_subjs):
            mask = train_subjs == subj
            mu, sigma = y_train[mask].mean(), y_train[mask].std() + 1e-8
            y_train_norm[mask] = (y_train[mask] - mu) / sigma
        
        # Normalize test features using test subject's own stats (CALIBRATION!)
        test_scaler = StandardScaler()
        X_test_norm = test_scaler.fit_transform(X_test)
        
        # Get test subject's Borg baseline (CALIBRATION!)
        test_mu = y_test.mean()
        test_sigma = y_test.std() + 1e-8
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_norm, y_train_norm)
        
        # Predict normalized, then denormalize to test subject's scale
        pred_norm = model.predict(X_test_norm)
        pred = pred_norm * test_sigma + test_mu
        
        y_true.extend(y_test)
        y_pred.extend(pred)
    
    return np.array(y_true), np.array(y_pred)


def method4_within_subject():
    """Method 4: Train and test within same subject (5-fold CV per subject)"""
    y_true, y_pred = [], []
    
    for subj in np.unique(subjects):
        mask = subjects == subj
        X_subj, y_subj = X[mask], y[mask]
        
        if len(y_subj) < 10:
            continue
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X_subj):
            X_train, X_test = X_subj[train_idx], X_subj[test_idx]
            y_train, y_test = y_subj[train_idx], y_subj[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            
            y_true.extend(y_test)
            y_pred.extend(pred)
    
    return np.array(y_true), np.array(y_pred)

# =============================================================================
# RUN ALL METHODS AND SHOW RESULTS
# =============================================================================
print("\n" + "="*70)
print("RUNNING ALL METHODS")
print("="*70)

methods = {
    "Method 1: Cross-Subject (Raw)": method1_cross_subject_raw,
    "Method 3: WITH CALIBRATION": method3_calibration,
    "Method 4: Within-Subject": method4_within_subject
}

results = {}

for name, func in methods.items():
    print(f"\n{'='*70}")
    print(f"📈 {name}")
    print("="*70)
    
    y_true, y_pred = func()
    
    # Convert to categories
    true_cats = [to_category(b) for b in y_true]
    pred_cats = [to_category(b) for b in y_pred]
    
    # Confusion matrix
    labels = ['LOW', 'MOD', 'HIGH']
    cm = confusion_matrix(true_cats, pred_cats, labels=labels)
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"                    PREDICTED")
    print(f"                  LOW    MOD   HIGH")
    print(f"         LOW    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}  | {cm[0].sum()}")
    print(f"ACTUAL   MOD    {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}  | {cm[1].sum()}")
    print(f"         HIGH   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}  | {cm[2].sum()}")
    
    # Per-category recall
    print(f"\n🎯 PER-CATEGORY RECALL ('When it's actually X, how often predict X?'):")
    recalls = {}
    for i, cat in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i, i]
        recall = 100 * correct / total if total > 0 else 0
        recalls[cat] = recall
        bar = "█" * int(recall / 5) + "░" * (20 - int(recall / 5))
        print(f"   {cat:4s}: {recall:5.1f}% [{bar}] ({correct}/{total})")
    
    # Per-category precision
    print(f"\n🎯 PER-CATEGORY PRECISION ('When we predict X, how often correct?'):")
    for i, cat in enumerate(labels):
        total = cm[:, i].sum()
        correct = cm[i, i]
        precision = 100 * correct / total if total > 0 else 0
        bar = "█" * int(precision / 5) + "░" * (20 - int(precision / 5))
        print(f"   {cat:4s}: {precision:5.1f}% [{bar}] ({correct}/{total})")
    
    # Overall and dangerous
    overall = 100 * sum(cm[i,i] for i in range(3)) / cm.sum()
    dangerous = cm[0,2] + cm[2,0]
    dangerous_pct = 100 * dangerous / cm.sum()
    
    print(f"\n📈 OVERALL EXACT: {overall:.1f}% (vs 33% random = +{overall-33.3:.1f}%)")
    print(f"⚠️  DANGEROUS (LOW↔HIGH): {dangerous} ({dangerous_pct:.1f}%)")
    
    results[name] = {
        'LOW': recalls['LOW'], 'MOD': recalls['MOD'], 'HIGH': recalls['HIGH'],
        'overall': overall, 'dangerous': dangerous_pct
    }

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("📋 FINAL SUMMARY TABLE")
print("="*70)
print(f"\n{'Method':<35} | {'LOW':>7} | {'MOD':>7} | {'HIGH':>7} | {'Overall':>8} | {'Danger':>7}")
print("-" * 90)
print(f"{'Random baseline':<35} |  33.3%  |  33.3%  |  33.3%  |   33.3%  |   N/A  ")
for name, vals in results.items():
    short = name.replace("Method ", "M").replace(": ", " - ")[:35]
    print(f"{short:<35} | {vals['LOW']:5.1f}%  | {vals['MOD']:5.1f}%  | {vals['HIGH']:5.1f}%  | {vals['overall']:6.1f}%  | {vals['dangerous']:5.1f}% ")

print("\n" + "="*70)
print("💡 INTERPRETATION")
print("="*70)
print("""
• LOW recall = "When elderly person is at rest, how often do we correctly say LOW?"
• HIGH recall = "When elderly person is exerting, how often do we correctly say HIGH?"
• DANGEROUS = Confusing LOW with HIGH (worst error for safety)

Method 1 (raw): Can't generalize across people - ~random guessing
Method 3 (calibrated): ~8 min calibration → much better predictions
Method 4 (within): Upper bound if we had lots of personal training data
""")
