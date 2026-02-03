#!/usr/bin/env python3
"""
Simpler 3-class ordinal: Low/Medium/High effort
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    paths = [
        '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
    ]
    
    dfs = []
    for i, p in enumerate(paths, 1):
        df = pd.read_csv(p)
        df['subject'] = f'P{i}'
        dfs.append(df)
    
    combined = pd.concat(dfs).dropna(subset=['borg'])
    imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
    imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]
    
    return combined, imu_cols


def borg_to_3class(borg):
    """3 classes: Low (0-2), Medium (3-4), High (5+)"""
    if borg <= 2:
        return 0  # Low
    elif borg <= 4:
        return 1  # Medium  
    else:
        return 2  # High


NAMES_3 = ['Low (0-2)', 'Medium (3-4)', 'High (5+)']


def run_3class_loso(df, features):
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    per_subj = []
    
    df = df.copy()
    df['cat3'] = df['borg'].apply(borg_to_3class)
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=features + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=features + ['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        X_train, y_train = train[features].values, train['cat3'].values
        X_test, y_test = test[features].values, test['cat3'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        per_subj.append((test_subj, acc))
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    return np.array(all_true), np.array(all_pred), per_subj


def main():
    print("=" * 60)
    print("3-CLASS CLASSIFICATION: Low / Medium / High")
    print("=" * 60)
    
    df, imu_cols = load_data()
    print(f"\n{len(df)} samples, {len(imu_cols)} IMU features")
    
    # Distribution
    df['cat3'] = df['borg'].apply(borg_to_3class)
    print("\nClass distribution:")
    for i, name in enumerate(NAMES_3):
        count = (df['cat3'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {name:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Run
    y_true, y_pred, per_subj = run_3class_loso(df, imu_cols)
    
    # Metrics
    pooled_acc = accuracy_score(y_true, y_pred)
    mean_acc = np.mean([a for _, a in per_subj])
    adjacent = np.mean(np.abs(y_true - y_pred) <= 1)
    
    print(f"\n" + "-" * 60)
    print("RESULTS (3-CLASS)")
    print("-" * 60)
    print(f"  Mean per-subject accuracy: {mean_acc*100:.1f}%")
    print(f"  Pooled accuracy:           {pooled_acc*100:.1f}%")
    print(f"  Adjacent (±1):             {adjacent*100:.1f}%")
    
    print("\n  Per-subject:")
    for subj, acc in per_subj:
        print(f"    {subj}: {acc*100:.1f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print("              Low  Med  High")
    for i, name in enumerate(['Low  ', 'Med  ', 'High ']):
        print(f"  True {name}: {cm[i,0]:4d} {cm[i,1]:4d} {cm[i,2]:4d}")
    
    # Per-class
    print("\n  Per-class accuracy:")
    for i, name in enumerate(NAMES_3):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"    {name:15s}: {acc*100:5.1f}%")
    
    print(f"\n" + "=" * 60)
    print(f"Random chance: 33.3%")
    print(f"Your accuracy: {mean_acc*100:.1f}% ({mean_acc*100/33.3:.1f}x better than random)")
    print("=" * 60)
    
    # Also try 2-class (binary: resting vs active)
    print("\n\n" + "=" * 60)
    print("2-CLASS CLASSIFICATION: Resting (0-2) vs Active (3+)")
    print("=" * 60)
    
    df['cat2'] = (df['borg'] >= 3).astype(int)
    
    subjects = df['subject'].unique()
    all_true_2, all_pred_2, per_subj_2 = [], [], []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=imu_cols + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=imu_cols + ['borg'])
        
        X_train, y_train = train[imu_cols].values, train['cat2'].values
        X_test, y_test = test[imu_cols].values, test['cat2'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_2 = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred_2)
        per_subj_2.append((test_subj, acc))
        
        all_true_2.extend(y_test)
        all_pred_2.extend(y_pred_2)
    
    all_true_2, all_pred_2 = np.array(all_true_2), np.array(all_pred_2)
    
    pooled_acc_2 = accuracy_score(all_true_2, all_pred_2)
    mean_acc_2 = np.mean([a for _, a in per_subj_2])
    
    print(f"\nClass distribution:")
    print(f"  Resting (0-2): {(df['cat2'] == 0).sum()} ({(df['cat2'] == 0).mean()*100:.1f}%)")
    print(f"  Active (3+):   {(df['cat2'] == 1).sum()} ({(df['cat2'] == 1).mean()*100:.1f}%)")
    
    print(f"\n  Mean per-subject accuracy: {mean_acc_2*100:.1f}%")
    print(f"  Pooled accuracy:           {pooled_acc_2*100:.1f}%")
    
    print("\n  Per-subject:")
    for subj, acc in per_subj_2:
        print(f"    {subj}: {acc*100:.1f}%")
    
    cm2 = confusion_matrix(all_true_2, all_pred_2)
    print("\n  Confusion Matrix:")
    print("              Rest  Active")
    print(f"  True Rest : {cm2[0,0]:4d} {cm2[0,1]:4d}")
    print(f"  True Active: {cm2[1,0]:4d} {cm2[1,1]:4d}")
    
    print(f"\nRandom chance: 50%")
    print(f"Your accuracy: {mean_acc_2*100:.1f}% ({mean_acc_2*100/50:.1f}x better than random)")


if __name__ == '__main__':
    main()
