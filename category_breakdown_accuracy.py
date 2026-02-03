#!/usr/bin/env python3
"""
Per-Category Accuracy Breakdown
Shows: When it's actually LOW, how often do we predict LOW?
       When it's actually MODERATE, how often do we predict MODERATE?
       When it's actually HIGH, how often do we predict HIGH?
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load all 5 elderly subjects
base = Path("/Users/pascalschlegel/data/interim")
subjects = [
    ("parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1", "P1"),
    ("parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2", "P2"),
    ("parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3", "P3"),
    ("parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4", "P4"),
    ("parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5", "P5"),
]

dfs = []
for path, name in subjects:
    fused_path = base / path / "fused_features_5.0s.csv"
    if fused_path.exists():
        df = pd.read_csv(fused_path)
        df['subject'] = name
        dfs.append(df)
        print(f"Loaded {name}: {len(df)} samples")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df)} samples")

# Clean data
meta_cols = ['subject', 'timestamp', 't_start', 't_end', 'borg', 'activity', 
             'patient', 'sim', 'condition', 'duration', 'Unnamed: 0']
feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]
df = df.dropna(subset=['borg'])
df = df.replace([np.inf, -np.inf], np.nan)

# Drop columns with too many NaNs
valid_cols = []
for c in feature_cols:
    if df[c].isna().sum() < len(df) * 0.5:  # Keep if <50% NaN
        valid_cols.append(c)
feature_cols = valid_cols

# Drop remaining NaN rows
df = df.dropna(subset=feature_cols)
print(f"After cleaning: {len(df)} samples, {len(feature_cols)} features")

X = df[feature_cols].values
y = df['borg'].values
subjects_arr = df['subject'].values

def to_category(borg):
    """Convert Borg to LOW/MOD/HIGH"""
    if borg <= 2:
        return 'LOW'
    elif borg <= 4:
        return 'MOD'
    else:
        return 'HIGH'

def run_method(method_name):
    """Run a method and return predictions"""
    logo = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []
    
    if method_name == "Method 1: Cross-Subject (Raw)":
        for train_idx, test_idx in logo.split(X, y, subjects_arr):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
    
    elif method_name == "Method 3: WITH CALIBRATION":
        for train_idx, test_idx in logo.split(X, y, subjects_arr):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Normalize features per subject for training
            train_subjects = subjects_arr[train_idx]
            X_train_norm = np.zeros_like(X_train, dtype=float)
            for subj in np.unique(train_subjects):
                mask = train_subjects == subj
                scaler = StandardScaler()
                X_train_norm[mask] = scaler.fit_transform(X_train[mask])
            
            # Normalize targets per subject for training
            y_train_norm = np.zeros_like(y_train, dtype=float)
            for subj in np.unique(train_subjects):
                mask = train_subjects == subj
                y_train_norm[mask] = (y_train[mask] - y_train[mask].mean()) / (y_train[mask].std() + 1e-8)
            
            # Normalize test features using test subject's own stats
            test_scaler = StandardScaler()
            X_test_norm = test_scaler.fit_transform(X_test)
            
            # Get test subject's calibration stats
            test_mean = y_test.mean()
            test_std = y_test.std() + 1e-8
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train_norm, y_train_norm)
            
            # Predict (normalized) and denormalize
            y_pred_norm = model.predict(X_test_norm)
            y_pred = y_pred_norm * test_std + test_mean
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
    
    elif method_name == "Method 4: Within-Subject":
        from sklearn.model_selection import KFold
        for subj in np.unique(subjects_arr):
            mask = subjects_arr == subj
            X_subj = X[mask]
            y_subj = y[mask]
            
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
                y_pred = model.predict(X_test_s)
                
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
    
    return np.array(y_true_all), np.array(y_pred_all)

print("="*70)
print("PER-CATEGORY ACCURACY BREAKDOWN")
print("="*70)
print("\nQuestion: When effort is actually LOW, how often do we predict LOW?")
print("          When effort is actually MOD, how often do we predict MOD?")
print("          When effort is actually HIGH, how often do we predict HIGH?")
print("\nRandom baseline with 3 equal categories: ~33% per category")
print("="*70)

# Show actual distribution first
print("\n📊 ACTUAL DISTRIBUTION IN DATA:")
true_cats_all = [to_category(b) for b in y]
for cat in ['LOW', 'MOD', 'HIGH']:
    count = true_cats_all.count(cat)
    print(f"   {cat}: {count} samples ({100*count/len(true_cats_all):.1f}%)")

methods = [
    "Method 1: Cross-Subject (Raw)",
    "Method 3: WITH CALIBRATION", 
    "Method 4: Within-Subject"
]

results_table = {}

for method in methods:
    print(f"\n{'='*70}")
    print(f"📈 {method}")
    print("="*70)
    
    y_true, y_pred = run_method(method)
    
    # Convert to categories
    true_cats = [to_category(b) for b in y_true]
    pred_cats = [to_category(b) for b in y_pred]
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_cats, pred_cats) if t == p)
    overall_acc = 100 * correct / len(true_cats)
    
    # Confusion matrix
    labels = ['LOW', 'MOD', 'HIGH']
    cm = confusion_matrix(true_cats, pred_cats, labels=labels)
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"                    PREDICTED")
    print(f"                  LOW    MOD   HIGH")
    print(f"         LOW    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"ACTUAL   MOD    {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"         HIGH   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    print(f"\n🎯 PER-CATEGORY RECALL (What % we get right per category):")
    print(f"   'When it's actually X, how often do we predict X?'\n")
    
    recalls = {}
    for i, cat in enumerate(labels):
        total_actual = cm[i].sum()
        correct = cm[i, i]
        recall = 100 * correct / total_actual if total_actual > 0 else 0
        recalls[cat] = recall
        improvement = recall - 33.3
        
        bar = "█" * int(recall / 5) + "░" * (20 - int(recall / 5))
        print(f"   {cat:4s}: {recall:5.1f}% [{bar}] ({correct}/{total_actual})")
        print(f"         vs random 33%: {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    results_table[method] = recalls
    results_table[method]['overall'] = overall_acc
    
    print(f"\n🎯 PER-CATEGORY PRECISION:")
    print(f"   'When we predict X, how often is it actually X?'\n")
    
    for i, cat in enumerate(labels):
        total_predicted = cm[:, i].sum()
        correct = cm[i, i]
        precision = 100 * correct / total_predicted if total_predicted > 0 else 0
        
        bar = "█" * int(precision / 5) + "░" * (20 - int(precision / 5))
        print(f"   {cat:4s}: {precision:5.1f}% [{bar}] ({correct}/{total_predicted} predicted as {cat})")
    
    # Dangerous errors
    dangerous = cm[0, 2] + cm[2, 0]  # LOW predicted as HIGH or vice versa
    dangerous_pct = 100 * dangerous / len(true_cats)
    results_table[method]['dangerous'] = dangerous_pct
    
    print(f"\n⚠️  DANGEROUS MISSES (LOW↔HIGH confusion): {dangerous} ({dangerous_pct:.1f}%)")
    
    print(f"\n📈 OVERALL: {overall_acc:.1f}% exact (vs 33% random = +{overall_acc-33.3:.1f}%)")

print("\n" + "="*70)
print("💡 FINAL SUMMARY TABLE")
print("="*70)
print(f"\n{'Method':<30} | {'LOW':>8} | {'MOD':>8} | {'HIGH':>8} | {'Overall':>8} | {'Danger':>8}")
print("-" * 90)
print(f"{'Random baseline':<30} |   33.3%  |   33.3%  |   33.3%  |   33.3%  |    N/A   ")
for method, vals in results_table.items():
    short_name = method.split(":")[0]
    print(f"{short_name:<30} | {vals['LOW']:6.1f}%  | {vals['MOD']:6.1f}%  | {vals['HIGH']:6.1f}%  | {vals['overall']:6.1f}%  | {vals['dangerous']:6.1f}%  ")
