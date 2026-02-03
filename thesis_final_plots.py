#!/usr/bin/env python3
"""
Final Thesis Plots - Correct LOSO Results
==========================================
1. Scatter plot (Actual vs Predicted Borg)
2. Top 10 features for IMU and PPG
3. Confusion matrix (3-class: Light/Moderate/Heavy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

paths = [
    '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
    '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
]

print("Loading data...")
dfs = []
for i, p in enumerate(paths, 1):
    df = pd.read_csv(p)
    df['subject'] = f'P{i}'
    dfs.append(df)
combined = pd.concat(dfs).dropna(subset=['borg'])
print(f"  {len(combined)} samples")

# Get feature columns
imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]

ppg_cols = [c for c in combined.columns if any(x in c.lower() for x in ['ppg', 'hr', 'ibi', 'rmssd', 'sdnn', 'pnn'])]
ppg_cols = [c for c in ppg_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]

print(f"  IMU features: {len(imu_cols)}")
print(f"  PPG features: {len(ppg_cols)}")

# 3-class categories
def borg_to_3class(b):
    if b <= 2: return 0  # Light
    elif b <= 4: return 1  # Moderate
    else: return 2  # Heavy

combined['cat3'] = combined['borg'].apply(borg_to_3class)

# ============================================================================
# Run LOSO and collect results
# ============================================================================

def run_loso_full(df, feature_cols, name):
    """Run LOSO and return all predictions + feature importances."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    all_true_cls, all_pred_cls = [], []
    per_subj_r = []
    feature_imp_sum = np.zeros(len(feature_cols))
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj].dropna(subset=feature_cols + ['borg'])
        test = df[df['subject'] == test_subj].dropna(subset=feature_cols + ['borg'])
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        X_train = train[feature_cols].values
        X_test = test[feature_cols].values
        y_train, y_test = train['borg'].values, test['borg'].values
        y_train_cls, y_test_cls = train['cat3'].values, test['cat3'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train_s = scl.fit_transform(imp.fit_transform(X_train))
        X_test_s = scl.transform(imp.transform(X_test))
        
        # Regression
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train_s, y_train)
        y_pred = rf_reg.predict(X_test_s)
        
        r, _ = pearsonr(y_test, y_pred)
        per_subj_r.append(r)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        feature_imp_sum += rf_reg.feature_importances_
        
        # Classification
        rf_cls = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf_cls.fit(X_train_s, y_train_cls)
        y_pred_cls = rf_cls.predict(X_test_s)
        all_true_cls.extend(y_test_cls)
        all_pred_cls.extend(y_pred_cls)
    
    # Average feature importance
    feature_imp_avg = feature_imp_sum / len(subjects)
    
    return {
        'true': np.array(all_true),
        'pred': np.array(all_pred),
        'true_cls': np.array(all_true_cls),
        'pred_cls': np.array(all_pred_cls),
        'per_subj_r': per_subj_r,
        'feature_imp': feature_imp_avg,
        'feature_names': feature_cols
    }

print("\nRunning LOSO for IMU...")
imu_results = run_loso_full(combined, imu_cols, "IMU")
print(f"  Mean r = {np.mean(imu_results['per_subj_r']):.2f}")

print("Running LOSO for PPG...")
ppg_results = run_loso_full(combined, ppg_cols, "PPG")
print(f"  Mean r = {np.mean(ppg_results['per_subj_r']):.2f}")

# ============================================================================
# Create plots
# ============================================================================

out_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
out_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# PLOT 1: Scatter plots (IMU and PPG side by side)
# --------------------------------------------------------------------------
print("\nCreating scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, results, name, color in [(axes[0], imu_results, 'IMU', 'steelblue'),
                                   (axes[1], ppg_results, 'PPG', 'coral')]:
    y_true, y_pred = results['true'], results['pred']
    r = np.mean(results['per_subj_r'])
    mae = np.mean(np.abs(y_true - y_pred))
    
    ax.scatter(y_true, y_pred, alpha=0.4, s=25, c=color, edgecolor='none')
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, linewidth=2, label='Perfect prediction')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 10, 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8, label=f'Fit (r={r:.2f})')
    
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(f'{name}: r = {r:.2f}, MAE = {mae:.2f}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Annotate compression issue
    ax.annotate(f'Predictions: {y_pred.min():.1f} - {y_pred.max():.1f}',
                xy=(0.95, 0.05), xycoords='axes fraction', ha='right',
                fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(out_dir / '51_scatter_imu_ppg.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '51_scatter_imu_ppg.png'}")

# --------------------------------------------------------------------------
# PLOT 2: Top 10 Features (IMU and PPG)
# --------------------------------------------------------------------------
print("\nCreating feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, name, color in [(axes[0], imu_results, 'IMU', 'steelblue'),
                                   (axes[1], ppg_results, 'PPG', 'coral')]:
    # Get top 10
    imp = results['feature_imp']
    names = results['feature_names']
    
    idx = np.argsort(imp)[::-1][:10]
    top_imp = imp[idx]
    top_names = [names[i] for i in idx]
    
    # Shorten names for display
    short_names = []
    for n in top_names:
        # Clean up feature names
        n = n.replace('_', ' ').replace('acc ', 'Acc ').replace('gyro ', 'Gyro ')
        if len(n) > 25:
            n = n[:22] + '...'
        short_names.append(n)
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_imp, color=color, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top 10 {name} Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(out_dir / '52_feature_importance.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '52_feature_importance.png'}")

# --------------------------------------------------------------------------
# PLOT 3: Confusion Matrix (3-class) for IMU
# --------------------------------------------------------------------------
print("\nCreating confusion matrix...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_names = ['Light\n(0-2)', 'Moderate\n(3-4)', 'Heavy\n(5+)']

for ax, results, name, cmap in [(axes[0], imu_results, 'IMU', 'Blues'),
                                 (axes[1], ppg_results, 'PPG', 'Oranges')]:
    cm = confusion_matrix(results['true_cls'], results['pred_cls'])
    
    # Normalize for display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    # Add numbers
    for i in range(3):
        for j in range(3):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)', 
                   ha='center', va='center', color=color, fontsize=10)
    
    # Metrics
    acc = accuracy_score(results['true_cls'], results['pred_cls'])
    adj = np.mean(np.abs(results['true_cls'] - results['pred_cls']) <= 1)
    ax.set_title(f'{name}: Acc={acc*100:.0f}%, ±1={adj*100:.0f}%', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Proportion')

plt.tight_layout()
plt.savefig(out_dir / '53_confusion_matrix_3class.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '53_confusion_matrix_3class.png'}")

# --------------------------------------------------------------------------
# PLOT 4: Combined summary (all in one figure for thesis)
# --------------------------------------------------------------------------
print("\nCreating combined summary plot...")

fig = plt.figure(figsize=(16, 12))

# Top row: Scatter plots
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)

for ax, results, name, color in [(ax1, imu_results, 'IMU', 'steelblue'),
                                  (ax2, ppg_results, 'PPG', 'coral')]:
    y_true, y_pred = results['true'], results['pred']
    r = np.mean(results['per_subj_r'])
    mae = np.mean(np.abs(y_true - y_pred))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, c=color)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5)
    ax.set_xlabel('Actual Borg')
    ax.set_ylabel('Predicted Borg')
    ax.set_title(f'{name}: r={r:.2f}, MAE={mae:.2f}')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.grid(True, alpha=0.3)

# Top right: Per-subject bar chart
ax3 = fig.add_subplot(2, 3, 3)
subjects = ['P1', 'P2', 'P3', 'P4', 'P5']
x = np.arange(len(subjects))
width = 0.35

bars1 = ax3.bar(x - width/2, imu_results['per_subj_r'], width, label='IMU', color='steelblue')
bars2 = ax3.bar(x + width/2, ppg_results['per_subj_r'], width, label='PPG', color='coral')

ax3.axhline(y=np.mean(imu_results['per_subj_r']), color='steelblue', linestyle='--', alpha=0.7)
ax3.axhline(y=np.mean(ppg_results['per_subj_r']), color='coral', linestyle='--', alpha=0.7)
ax3.set_ylabel('Pearson r')
ax3.set_xlabel('Subject (Leave-One-Out)')
ax3.set_title('Per-Subject Correlation')
ax3.set_xticks(x)
ax3.set_xticklabels(subjects)
ax3.legend()
ax3.set_ylim(0, 0.8)
ax3.grid(True, alpha=0.3, axis='y')

# Bottom left: IMU feature importance
ax4 = fig.add_subplot(2, 3, 4)
imp = imu_results['feature_imp']
names = imu_results['feature_names']
idx = np.argsort(imp)[::-1][:8]
top_imp = imp[idx]
top_names = [names[i][:20] for i in idx]
ax4.barh(range(8), top_imp[::-1], color='steelblue')
ax4.set_yticks(range(8))
ax4.set_yticklabels(top_names[::-1], fontsize=9)
ax4.set_xlabel('Importance')
ax4.set_title('Top 8 IMU Features')

# Bottom middle: PPG feature importance
ax5 = fig.add_subplot(2, 3, 5)
imp = ppg_results['feature_imp']
names = ppg_results['feature_names']
idx = np.argsort(imp)[::-1][:8]
top_imp = imp[idx]
top_names = [names[i][:20] for i in idx]
ax5.barh(range(8), top_imp[::-1], color='coral')
ax5.set_yticks(range(8))
ax5.set_yticklabels(top_names[::-1], fontsize=9)
ax5.set_xlabel('Importance')
ax5.set_title('Top 8 PPG Features')

# Bottom right: Confusion matrix (IMU only)
ax6 = fig.add_subplot(2, 3, 6)
cm = confusion_matrix(imu_results['true_cls'], imu_results['pred_cls'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = ax6.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax6.set_xticks(range(3))
ax6.set_yticks(range(3))
ax6.set_xticklabels(['Light', 'Mod', 'Heavy'])
ax6.set_yticklabels(['Light', 'Mod', 'Heavy'])
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')
for i in range(3):
    for j in range(3):
        color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        ax6.text(j, i, f'{cm[i,j]}', ha='center', va='center', color=color, fontsize=11)
acc = accuracy_score(imu_results['true_cls'], imu_results['pred_cls'])
ax6.set_title(f'IMU 3-Class: {acc*100:.0f}% acc')

plt.tight_layout()
plt.savefig(out_dir / '54_combined_summary.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {out_dir / '54_combined_summary.png'}")

# ============================================================================
# Print summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
IMU Results:
  - Mean per-subject r: {np.mean(imu_results['per_subj_r']):.2f}
  - MAE: {np.mean(np.abs(imu_results['true'] - imu_results['pred'])):.2f}
  - 3-class accuracy: {accuracy_score(imu_results['true_cls'], imu_results['pred_cls'])*100:.0f}%
  - ±1 class accuracy: {np.mean(np.abs(imu_results['true_cls'] - imu_results['pred_cls']) <= 1)*100:.0f}%

PPG Results:
  - Mean per-subject r: {np.mean(ppg_results['per_subj_r']):.2f}
  - MAE: {np.mean(np.abs(ppg_results['true'] - ppg_results['pred'])):.2f}
  - 3-class accuracy: {accuracy_score(ppg_results['true_cls'], ppg_results['pred_cls'])*100:.0f}%
  - ±1 class accuracy: {np.mean(np.abs(ppg_results['true_cls'] - ppg_results['pred_cls']) <= 1)*100:.0f}%

Plots saved:
  51_scatter_imu_ppg.png      - Scatter plots for both modalities
  52_feature_importance.png   - Top 10 features for IMU and PPG
  53_confusion_matrix_3class.png - 3-class confusion matrices
  54_combined_summary.png     - All-in-one summary figure
""")
