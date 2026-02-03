#!/usr/bin/env python3
"""
Ordinal Classification for Effort Estimation
=============================================
Categories:
- No effort: Borg 0-1
- Light: Borg 2-3
- Moderate: Borg 4-5
- Hard: Borg 6+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load all subject data."""
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
    
    # Get IMU features
    imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
    imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]
    
    return combined, imu_cols


def borg_to_category(borg):
    """Convert Borg to ordinal category.
    - No effort: 0-1
    - Light: 2-3
    - Moderate: 4-5
    - Hard: 6+
    """
    if borg <= 1:
        return 0  # No effort
    elif borg <= 3:
        return 1  # Light
    elif borg <= 5:
        return 2  # Moderate
    else:
        return 3  # Hard


CATEGORY_NAMES = ['No effort', 'Light', 'Moderate', 'Hard']
CATEGORY_BORG_RANGES = ['0-1', '2-3', '4-5', '6+']


def run_regression_loso(df, features):
    """Standard LOSO regression for comparison."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    per_subj_r = []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=features + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=features + ['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        X_train, y_train = train[features].values, train['borg'].values
        X_test, y_test = test[features].values, test['borg'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Per-subject r
        r, _ = pearsonr(y_test, y_pred)
        per_subj_r.append(r)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    mean_r = np.mean(per_subj_r)
    return np.array(all_true), np.array(all_pred), mean_r


def run_classification_loso(df, features):
    """LOSO ordinal classification."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    per_subj_acc = []
    
    # Add category column
    df = df.copy()
    df['category'] = df['borg'].apply(borg_to_category)
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=features + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=features + ['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        X_train, y_train = train[features].values, train['category'].values
        X_test, y_test = test[features].values, test['category'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Per-subject accuracy
        acc = accuracy_score(y_test, y_pred)
        per_subj_acc.append(acc)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    mean_acc = np.mean(per_subj_acc)
    return np.array(all_true), np.array(all_pred), mean_acc, per_subj_acc


def main():
    print("=" * 60)
    print("ORDINAL CLASSIFICATION FOR EFFORT ESTIMATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df, imu_cols = load_data()
    print(f"  {len(df)} samples, {len(imu_cols)} IMU features")
    
    # Show class distribution
    df_cat = df.copy()
    df_cat['category'] = df_cat['borg'].apply(borg_to_category)
    print("\nClass distribution:")
    for cat in range(4):
        count = (df_cat['category'] == cat).sum()
        pct = count / len(df_cat) * 100
        print(f"  {CATEGORY_NAMES[cat]:12s} (Borg {CATEGORY_BORG_RANGES[cat]:4s}): {count:4d} ({pct:5.1f}%)")
    
    # Run regression baseline
    print("\n" + "-" * 60)
    print("REGRESSION BASELINE (continuous Borg)")
    print("-" * 60)
    y_true_reg, y_pred_reg, mean_r = run_regression_loso(df, imu_cols)
    pooled_r, _ = pearsonr(y_true_reg, y_pred_reg)
    mae_reg = np.mean(np.abs(y_true_reg - y_pred_reg))
    print(f"  Mean per-subject r: {mean_r:.3f}")
    print(f"  Pooled r:          {pooled_r:.3f}")
    print(f"  MAE:               {mae_reg:.2f}")
    print(f"  Pred range:        {y_pred_reg.min():.1f} - {y_pred_reg.max():.1f}")
    
    # Run ordinal classification
    print("\n" + "-" * 60)
    print("ORDINAL CLASSIFICATION (4 classes)")
    print("-" * 60)
    y_true_cat, y_pred_cat, mean_acc, per_subj_acc = run_classification_loso(df, imu_cols)
    
    # Metrics
    pooled_acc = accuracy_score(y_true_cat, y_pred_cat)
    adjacent_acc = np.mean(np.abs(y_true_cat - y_pred_cat) <= 1)
    
    print(f"\n  Mean per-subject accuracy: {mean_acc*100:.1f}%")
    print(f"  Pooled exact accuracy:     {pooled_acc*100:.1f}%")
    print(f"  Adjacent (±1 cat):         {adjacent_acc*100:.1f}%")
    
    print("\n  Per-subject accuracy:")
    subjects = df['subject'].unique()
    for i, subj in enumerate(subjects):
        print(f"    {subj}: {per_subj_acc[i]*100:.1f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_true_cat, y_pred_cat)
    print("\n  Confusion Matrix:")
    print("                Predicted")
    print("                No effort  Light  Moderate  Hard")
    for i, name in enumerate(CATEGORY_NAMES):
        row = cm[i] if i < len(cm) else [0, 0, 0, 0]
        print(f"  True {name:9s}:  {row[0]:4d}     {row[1]:4d}     {row[2]:4d}     {row[3]:4d}")
    
    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for i, name in enumerate(CATEGORY_NAMES):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            print(f"    {name:12s}: {class_acc*100:5.1f}%  ({cm[i,i]}/{cm[i].sum()})")
    
    # =====================================================
    # Create visualization
    # =====================================================
    print("\n" + "-" * 60)
    print("Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Regression scatter
    ax1 = axes[0]
    ax1.scatter(y_true_reg, y_pred_reg, alpha=0.4, s=20, c='steelblue')
    ax1.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='Perfect')
    ax1.set_xlabel('Actual Borg', fontsize=12)
    ax1.set_ylabel('Predicted Borg', fontsize=12)
    ax1.set_title(f'Regression: r = {mean_r:.2f} (mean), MAE = {mae_reg:.2f}', fontsize=12)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrix heatmap
    ax2 = axes[1]
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(['No\neffort', 'Light', 'Mod', 'Hard'])
    ax2.set_yticklabels(['No\neffort', 'Light', 'Mod', 'Hard'])
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title(f'Classification: {mean_acc*100:.0f}% (mean), {adjacent_acc*100:.0f}% ±1', fontsize=12)
    
    # Add numbers
    for i in range(4):
        for j in range(4):
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=11)
    
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # Plot 3: Per-class comparison
    ax3 = axes[2]
    x_pos = np.arange(4)
    
    # True distribution
    true_counts = [np.sum(y_true_cat == i) for i in range(4)]
    correct_counts = [cm[i, i] for i in range(4)]
    
    bars1 = ax3.bar(x_pos - 0.2, true_counts, 0.4, label='Total', color='lightsteelblue', edgecolor='steelblue')
    bars2 = ax3.bar(x_pos + 0.2, correct_counts, 0.4, label='Correct', color='steelblue', edgecolor='navy')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['No effort\n(0-1)', 'Light\n(2-3)', 'Moderate\n(4-5)', 'Hard\n(6+)'])
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Samples per Class', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy percentages on top
    for i in range(4):
        if true_counts[i] > 0:
            acc = correct_counts[i] / true_counts[i] * 100
            ax3.annotate(f'{acc:.0f}%', xy=(x_pos[i] + 0.2, correct_counts[i] + 10),
                        ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    out_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / '50_ordinal_classification.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir / '50_ordinal_classification.png'}")
    
    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
┌────────────────────────────────────────────────────────┐
│ Approach                  │ Result                         │
├───────────────────────────┼────────────────────────────────┤
│ Regression (Borg 0-10)    │ r = {mean_r:.2f} (mean), MAE = {mae_reg:.2f}     │
│ Classification (4 class)  │ {mean_acc*100:.0f}% exact (mean), {adjacent_acc*100:.0f}% ±1 class │
└───────────────────────────┴────────────────────────────────┘

Random chance for 4 classes: 25%
Your mean per-subject accuracy: {mean_acc*100:.0f}%
Your ±1 accuracy: {adjacent_acc*100:.0f}%

This means: {mean_acc*100/25:.1f}x better than random guessing
""")
    
    return mean_acc, adjacent_acc


if __name__ == '__main__':
    main()
