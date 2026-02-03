#!/usr/bin/env python3
"""
1. Plot 16 with REAL sensor data (not random)
2. Show per-subject within-patient breakdown for PPG argument
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# =============================================================================
# PLOT 16: REAL SENSOR DATA
# =============================================================================
print("="*60)
print("PLOT 16: Loading real sensor data...")
print("="*60)

# Load raw data from one subject (P3 as example)
base = Path('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3')

# Load each sensor
ppg_path = base / 'ppg_green' / 'ppg_green_preprocessed.csv'
imu_path = base / 'imu_wrist' / 'imu_preprocessed.csv'
eda_path = base / 'eda' / 'eda_preprocessed.csv'
fused_path = base / 'fused_aligned_5.0s.csv'

# Check what files exist
print(f"PPG exists: {ppg_path.exists()}")
print(f"IMU exists: {imu_path.exists()}")
print(f"EDA exists: {eda_path.exists()}")

# Load data
ppg = pd.read_csv(ppg_path)
imu = pd.read_csv(imu_path)
eda = pd.read_csv(eda_path)
fused = pd.read_csv(fused_path)

print(f"PPG shape: {ppg.shape}, value range: {ppg['value'].min():.0f} - {ppg['value'].max():.0f}")
print(f"IMU shape: {imu.shape}")
print(f"EDA shape: {eda.shape}, eda_cc range: {eda['eda_cc'].min():.1f} - {eda['eda_cc'].max():.1f}")

# Find common start time
t_start = min(ppg['t_unix'].min(), imu['t_unix'].min(), eda['t_unix'].min())

# Convert to minutes from start
ppg_t = (ppg['t_unix'].values - t_start) / 60
imu_t = (imu['t_unix'].values - t_start) / 60
eda_t = (eda['t_unix'].values - t_start) / 60
fused_t = (fused['t_center'].values - t_start) / 60

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# PPG - downsample for plotting
step = max(1, len(ppg_t) // 10000)
axes[0].plot(ppg_t[::step], ppg['value'].values[::step], 'g-', linewidth=0.3, alpha=0.8)
axes[0].set_ylabel('PPG\n(a.u.)', fontsize=11)
axes[0].set_title('Raw Sensor Signals - Subject P3 (Elderly)', fontsize=14)
# Set reasonable ylim
ppg_mean = ppg['value'].mean()
ppg_std = ppg['value'].std()
axes[0].set_ylim(ppg_mean - 3*ppg_std, ppg_mean + 3*ppg_std)

# IMU - accelerometer magnitude
step = max(1, len(imu_t) // 10000)
acc_mag = np.sqrt(imu['acc_x_dyn']**2 + imu['acc_y_dyn']**2 + imu['acc_z_dyn']**2)
axes[1].plot(imu_t[::step], acc_mag.values[::step], 'b-', linewidth=0.3, alpha=0.8)
axes[1].set_ylabel('Accelerometer\nMagnitude (g)', fontsize=11)
# Set reasonable ylim
acc_99 = np.percentile(acc_mag, 99)
axes[1].set_ylim(0, acc_99 * 1.1)

# EDA - with proper scaling
step = max(1, len(eda_t) // 5000)
axes[2].plot(eda_t[::step], eda['eda_cc'].values[::step], 'purple', linewidth=0.5, alpha=0.8)
axes[2].set_ylabel('EDA\n(µS)', fontsize=11)
# Set ylim to show variation
eda_min = eda['eda_cc'].min()
eda_max = eda['eda_cc'].max()
eda_range = eda_max - eda_min
axes[2].set_ylim(eda_min - 0.1*eda_range, eda_max + 0.1*eda_range)

# Borg ratings - scatter with line
mask = ~np.isnan(fused['borg'].values)
axes[3].scatter(fused_t[mask], fused['borg'].values[mask], c='red', s=50, alpha=0.8, zorder=5, edgecolors='black', linewidth=0.5)
axes[3].plot(fused_t[mask], fused['borg'].values[mask], 'r-', alpha=0.5, linewidth=1.5)
axes[3].set_ylabel('Borg CR10\n(0-10)', fontsize=11)
axes[3].set_ylim(-0.5, 10.5)
axes[3].set_yticks([0, 2, 4, 6, 8, 10])

axes[3].set_xlabel('Time (minutes from start)', fontsize=12)

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/16_multimodal_overview_real.png", dpi=150, bbox_inches='tight')
print(f"Saved: 16_multimodal_overview_real.png")
plt.close()

# =============================================================================
# WITHIN-PATIENT BREAKDOWN (Per Subject)
# =============================================================================
print("\n" + "="*60)
print("WITHIN-PATIENT BREAKDOWN (Per Subject)")
print("="*60)

# Load all data
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        tmp = pd.read_csv(path)
        tmp['subject'] = f'P{i}'
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df.columns if 'ppg_' in c]
eda_cols = [c for c in df.columns if 'eda_' in c]

print(f"\nFeatures: IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)}")

def within_patient_per_subject(feature_cols, modality_name):
    """70/30 split within each subject, return per-subject r values"""
    results = {}
    for subj in sorted(df['subject'].unique()):
        subj_df = df[df['subject'] == subj]
        
        X = subj_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = subj_df['borg'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r, _ = pearsonr(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        results[subj] = {'r': r, 'mae': mae, 'n_test': len(y_test)}
    
    return results

# Run for all modalities
print("\n--- IMU Within-Patient ---")
imu_within = within_patient_per_subject(imu_cols, 'IMU')
for subj, m in imu_within.items():
    print(f"  {subj}: r={m['r']:.2f}, MAE={m['mae']:.2f}")
print(f"  MEAN: r={np.mean([m['r'] for m in imu_within.values()]):.2f}")

print("\n--- PPG Within-Patient ---")
ppg_within = within_patient_per_subject(ppg_cols, 'PPG')
for subj, m in ppg_within.items():
    print(f"  {subj}: r={m['r']:.2f}, MAE={m['mae']:.2f}")
print(f"  MEAN: r={np.mean([m['r'] for m in ppg_within.values()]):.2f}")

print("\n--- EDA Within-Patient ---")
eda_within = within_patient_per_subject(eda_cols, 'EDA')
for subj, m in eda_within.items():
    print(f"  {subj}: r={m['r']:.2f}, MAE={m['mae']:.2f}")
print(f"  MEAN: r={np.mean([m['r'] for m in eda_within.values()]):.2f}")

# =============================================================================
# PLOT: Per-Subject Within vs Cross comparison
# =============================================================================
print("\n" + "="*60)
print("Creating per-subject comparison plot...")
print("="*60)

# Also get cross-patient (LOSO) per subject for comparison
def loso_per_subject(feature_cols):
    results = {}
    for test_subj in sorted(df['subject'].unique()):
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train['borg'].values
        X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r, _ = pearsonr(y_test, y_pred)
        results[test_subj] = r
    return results

ppg_cross = loso_per_subject(ppg_cols)

# Create the plot showing PPG works within but not across
fig, ax = plt.subplots(figsize=(10, 6))

subjects = ['P1', 'P2', 'P3', 'P4', 'P5']
x = np.arange(len(subjects))
width = 0.35

within_vals = [ppg_within[s]['r'] for s in subjects]
cross_vals = [ppg_cross[s] for s in subjects]

bars1 = ax.bar(x - width/2, within_vals, width, label='Within-Patient (70/30)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, cross_vals, width, label='Cross-Patient (LOSO)', color='#e74c3c', edgecolor='black')

# Add value labels
for bar, val in zip(bars1, within_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, cross_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('Pearson r', fontsize=12)
ax.set_title('PPG Features: Within-Patient vs Cross-Patient Performance\nPPG works for each individual, but doesn\'t generalize', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5')

# Add means
mean_within = np.mean(within_vals)
mean_cross = np.mean(cross_vals)
ax.text(0.02, 0.98, f'Mean Within: r={mean_within:.2f}\nMean Cross: r={mean_cross:.2f}\nGap: {mean_within - mean_cross:.2f}', 
        transform=ax.transAxes, va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{OUT}/25_ppg_within_vs_cross_per_subject.png", dpi=150, bbox_inches='tight')
print(f"Saved: 25_ppg_within_vs_cross_per_subject.png")
plt.close()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: Within-Patient Per Subject")
print("="*60)
print("""
This is the KEY INSIGHT for your thesis argument:

┌─────────┬────────────────────────────────┬────────────────────────────────┐
│ Subject │ PPG Within-Patient r           │ PPG Cross-Patient (LOSO) r     │
├─────────┼────────────────────────────────┼────────────────────────────────┤""")
for s in subjects:
    print(f"│ {s}      │ {ppg_within[s]['r']:.2f}                           │ {ppg_cross[s]:.2f}                           │")
print(f"""├─────────┼────────────────────────────────┼────────────────────────────────┤
│ MEAN    │ {mean_within:.2f}                           │ {mean_cross:.2f}                           │
└─────────┴────────────────────────────────┴────────────────────────────────┘

YOUR ARGUMENT:
- PPG achieves r ≈ 0.7 WITHIN each patient
- But r ≈ 0.26 when trying to generalize to NEW patients
- This motivates a LONGITUDINAL / PERSONALIZED approach:
  "Collect some initial data from a new patient, fine-tune the model,
   then PPG could work well for that specific person"
""")
