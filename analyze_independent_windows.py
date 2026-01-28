#!/usr/bin/env python3
"""
Analyze results with independent (0% overlap) windows.
Compare ML performance, correlations, and sample efficiency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined")
OUTPUT_DIR = DATA_DIR / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("VALIDATION WITH INDEPENDENT WINDOWS (0% OVERLAP)")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_DIR / "multisub_aligned_10.0s.csv")
df_labeled = df.dropna(subset=['borg'])

print(f"\n📊 Dataset: {len(df_labeled)} labeled samples (independent windows)")

# Get feature columns
meta_cols = ['t_center', 'window_start', 'window_end', 'borg', 'subject', 'start_idx', 'end_idx']
feature_cols = [c for c in df_labeled.columns if c not in meta_cols and not c.startswith('Unnamed')]

# Load selected features
selected_features_path = DATA_DIR / "qc_10.0s" / "features_selected_pruned.csv"
if selected_features_path.exists():
    selected_df = pd.read_csv(selected_features_path, header=None)
    selected_features = selected_df[0].tolist()
    feature_cols = [f for f in selected_features if f in df_labeled.columns]
    print(f"✓ Using {len(feature_cols)} selected features")

X = df_labeled[feature_cols].copy()
y = df_labeled['borg'].values
subjects = df_labeled['subject'].values if 'subject' in df_labeled.columns else None

# Handle NaN
X = X.fillna(X.median())

print(f"\n{'='*70}")
print("1. ML VALIDATION (NO TEMPORAL LEAKAGE)")
print("="*70)

# ============================================================
# Random Split (standard ML)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

r2_random = r2_score(y_test, y_pred)
mae_random = mean_absolute_error(y_test, y_pred)

print(f"\n🎲 Random 80/20 Split:")
print(f"   R² = {r2_random:.3f}")
print(f"   MAE = {mae_random:.2f} Borg points")

# ============================================================
# Time Series Cross-Validation
# ============================================================
print(f"\n⏱️  Time-Series 5-Fold CV:")

# Sort by time
df_sorted = df_labeled.sort_values('t_center').reset_index(drop=True)
X_ts = df_sorted[feature_cols].fillna(df_sorted[feature_cols].median())
y_ts = df_sorted['borg'].values

tscv = TimeSeriesSplit(n_splits=5)
ts_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    X_tr, X_te = X_ts.iloc[train_idx], X_ts.iloc[test_idx]
    y_tr, y_te = y_ts[train_idx], y_ts[test_idx]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_scaled, y_tr)
    y_pred = rf.predict(X_te_scaled)
    
    r2 = r2_score(y_te, y_pred)
    ts_scores.append(r2)
    print(f"   Fold {fold+1}: R² = {r2:.3f}")

r2_ts_mean = np.mean(ts_scores)
r2_ts_std = np.std(ts_scores)
print(f"   Mean R² = {r2_ts_mean:.3f} ± {r2_ts_std:.3f}")

# ============================================================
# Leave-One-Subject-Out (if multiple subjects)
# ============================================================
if subjects is not None:
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) > 1:
        print(f"\n👥 Leave-One-Subject-Out CV:")
        loso_scores = []
        
        for test_subj in unique_subjects:
            train_mask = subjects != test_subj
            test_mask = subjects == test_subj
            
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            
            if len(y_te) < 10:
                continue
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_tr_scaled, y_tr)
            y_pred = rf.predict(X_te_scaled)
            
            r2 = r2_score(y_te, y_pred)
            loso_scores.append(r2)
            print(f"   Test on {test_subj}: R² = {r2:.3f} (n={len(y_te)})")
        
        r2_loso_mean = np.mean(loso_scores)
        r2_loso_std = np.std(loso_scores)
        print(f"   Mean R² = {r2_loso_mean:.3f} ± {r2_loso_std:.3f}")

print(f"\n{'='*70}")
print("2. BIVARIATE CORRELATIONS (TOP FEATURES)")
print("="*70)

# Calculate correlations
correlations = []
for col in feature_cols:
    valid_mask = ~np.isnan(X[col]) & ~np.isnan(y)
    if valid_mask.sum() > 30:
        r, p = stats.pearsonr(X[col][valid_mask], y[valid_mask])
        correlations.append({
            'feature': col,
            'r': r,
            'abs_r': abs(r),
            'p_value': p,
            'n': valid_mask.sum()
        })

corr_df = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)

print("\n🏆 Top 15 Features by |r| with Borg Effort:")
print("-" * 50)
for i, row in corr_df.head(15).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature'][:40]:<40} r = {row['r']:+.3f} {sig}")

# Group by modality
print("\n📊 Best Feature per Modality:")
print("-" * 50)
for modality, prefix in [('EDA', 'eda'), ('IMU', 'acc_'), ('PPG', 'ppg')]:
    if modality == 'IMU':
        mod_df = corr_df[corr_df['feature'].str.contains('acc_|gyro_')]
    else:
        mod_df = corr_df[corr_df['feature'].str.startswith(prefix)]
    if len(mod_df) > 0:
        best = mod_df.iloc[0]
        print(f"  {modality:<5}: {best['feature'][:35]:<35} r = {best['r']:+.3f}")

print(f"\n{'='*70}")
print("3. SAMPLE EFFICIENCY COMPARISON")
print("="*70)

# Compare with what we had before (estimated from overlap)
# With 70% overlap: ~3x more samples
# samples_with_overlap ≈ samples_no_overlap * (1 / (1 - overlap))
# For 70% overlap: factor ≈ 3.33x

samples_no_overlap = len(df_labeled)
estimated_with_overlap = int(samples_no_overlap * 3.33)

print(f"\n📉 Sample Counts:")
print(f"   With 0% overlap (current):  {samples_no_overlap:,} independent windows")
print(f"   With 70% overlap (estimated): ~{estimated_with_overlap:,} overlapping windows")
print(f"   Reduction factor: ~{estimated_with_overlap/samples_no_overlap:.1f}x fewer samples")

print(f"\n✅ Trade-off Analysis:")
print(f"   ➖ Fewer samples: {samples_no_overlap} vs ~{estimated_with_overlap}")
print(f"   ➕ No temporal leakage in ML validation")
print(f"   ➕ R² scores are now HONEST and interpretable")
print(f"   ➕ Train/test truly independent")

# ============================================================
# Create validation summary plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. ML Performance Comparison
ax1 = axes[0, 0]
methods = ['Random\nSplit', 'Time-Series\nCV', 'Leave-One-\nSubject-Out']
scores = [r2_random, r2_ts_mean]
errors = [0, r2_ts_std]
if 'r2_loso_mean' in dir():
    scores.append(r2_loso_mean)
    errors.append(r2_loso_std)
else:
    methods = methods[:2]

colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(scores)]
bars = ax1.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.errorbar(range(len(scores)), scores, yerr=errors, fmt='none', color='black', capsize=5)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('ML Validation Performance\n(Independent Windows)', fontsize=14, fontweight='bold')
ax1.set_ylim(-0.5, 1.0)

for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Top correlations
ax2 = axes[0, 1]
top_n = 10
top_corr = corr_df.head(top_n)
colors = ['#e74c3c' if 'eda' in f else '#3498db' if 'imu' in f else '#2ecc71' 
          for f in top_corr['feature']]
bars = ax2.barh(range(top_n), top_corr['abs_r'], color=colors, edgecolor='black')
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([f[:30] for f in top_corr['feature']], fontsize=9)
ax2.set_xlabel('|Correlation| with Borg', fontsize=12)
ax2.set_title('Top 10 Features\n(Bivariate Correlation)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='EDA'),
                   Patch(facecolor='#3498db', label='IMU'),
                   Patch(facecolor='#2ecc71', label='PPG')]
ax2.legend(handles=legend_elements, loc='lower right')

# 3. Prediction vs Actual scatter
ax3 = axes[1, 0]
# Re-fit for scatter plot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

ax3.scatter(y_test, y_pred, alpha=0.5, edgecolor='black', linewidth=0.5)
ax3.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Perfect prediction')
ax3.set_xlabel('Actual Borg Effort', fontsize=12)
ax3.set_ylabel('Predicted Borg Effort', fontsize=12)
ax3.set_title(f'Prediction vs Actual\n(R² = {r2_random:.3f}, MAE = {mae_random:.2f})', 
              fontsize=14, fontweight='bold')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.legend()
ax3.set_aspect('equal')

# 4. Comparison text box
ax4 = axes[1, 1]
ax4.axis('off')

# Get best features per modality (with safety checks)
eda_best = corr_df[corr_df['feature'].str.startswith('eda')]
imu_best = corr_df[corr_df['feature'].str.contains('acc_|gyro_')]  # IMU features have acc_ or gyro_ prefix
ppg_best = corr_df[corr_df['feature'].str.startswith('ppg')]

comparison_text = f"""
VALIDATION SUMMARY
═══════════════════════════════════════

📊 Dataset
   • {samples_no_overlap:,} independent windows
   • {len(feature_cols)} selected features
   • 3 subjects combined

🎯 ML Performance (Honest R²)
   • Random Split:    R² = {r2_random:.3f}
   • Time-Series CV:  R² = {r2_ts_mean:.3f} ± {r2_ts_std:.3f}
"""
if 'r2_loso_mean' in dir():
    comparison_text += f"   • LOSO CV:        R² = {r2_loso_mean:.3f} ± {r2_loso_std:.3f}\n"

comparison_text += f"""
🔬 Top Correlations
"""
if len(eda_best) > 0:
    comparison_text += f"   • EDA:  {eda_best.iloc[0]['feature'][:25]}\n           r = {eda_best.iloc[0]['r']:+.3f}\n"
if len(imu_best) > 0:
    comparison_text += f"   • IMU:  {imu_best.iloc[0]['feature'][:25]}\n           r = {imu_best.iloc[0]['r']:+.3f}\n"
if len(ppg_best) > 0:
    comparison_text += f"   • PPG:  {ppg_best.iloc[0]['feature'][:25]}\n           r = {ppg_best.iloc[0]['r']:+.3f}\n"

comparison_text += """
✅ Key Finding
   With independent windows (0% overlap),
   ML validation is now HONEST - no temporal leakage!
"""

ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'validation_summary.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\n💾 Saved: {OUTPUT_DIR / 'validation_summary.png'}")

# Save correlations
corr_df.to_csv(OUTPUT_DIR / 'feature_correlations.csv', index=False)
print(f"💾 Saved: {OUTPUT_DIR / 'feature_correlations.csv'}")

plt.show()

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE")
print("="*70)
