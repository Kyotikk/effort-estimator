#!/usr/bin/env python3
"""
Fixed visualizations:
1. Within vs Cross - 2 colors only
2. Raw signals - properly aligned x-axis
3. Verify within-patient methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gzip
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_final")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'within': '#2E86AB',  # Blue for within
    'cross': '#DC3545',   # Red for cross
}

print("="*70)
print("FIXING VISUALIZATIONS")
print("="*70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading fused data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'P{i}'
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_labeled = df_all.dropna(subset=['borg'])

imu_cols = [c for c in df_all.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in df_all.columns if 'ppg_' in c]
eda_cols = [c for c in df_all.columns if 'eda_' in c]

print(f"  {len(df_labeled)} windows, 5 subjects")

# =============================================================================
# COMPUTE WITHIN-PATIENT (PROPERLY)
# =============================================================================
print("\n" + "="*70)
print("WITHIN-PATIENT ANALYSIS (DETAILED)")
print("="*70)

def compute_within_patient_detailed(feature_cols, modality_name):
    """Compute within-patient performance with proper methodology."""
    results = []
    
    for subj in df_all['subject'].unique():
        subj_df = df_all[df_all['subject'] == subj].dropna(subset=['borg'])
        
        if len(subj_df) < 30:
            continue
        
        X = subj_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = subj_df['borg'].values
        
        # 70/30 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            results.append({
                'subject': subj,
                'r': r,
                'mae': mae,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'y_test_std': np.std(y_test),
                'y_pred_std': np.std(y_pred)
            })
            
            print(f"  {subj} ({modality_name}): r={r:.2f}, MAE={mae:.2f}, n_test={len(y_test)}, y_std={np.std(y_test):.2f}")
    
    return results

print("\nIMU Within-Patient:")
within_imu = compute_within_patient_detailed(imu_cols, "IMU")
mean_within_imu = np.mean([r['r'] for r in within_imu])

print("\nPPG Within-Patient:")
within_ppg = compute_within_patient_detailed(ppg_cols, "PPG")
mean_within_ppg = np.mean([r['r'] for r in within_ppg])

print("\nEDA Within-Patient:")
within_eda = compute_within_patient_detailed(eda_cols, "EDA")
mean_within_eda = np.mean([r['r'] for r in within_eda])

print(f"\n--- MEAN WITHIN-PATIENT ---")
print(f"  IMU: r = {mean_within_imu:.2f}")
print(f"  PPG: r = {mean_within_ppg:.2f}")
print(f"  EDA: r = {mean_within_eda:.2f}")

# =============================================================================
# COMPUTE CROSS-PATIENT (LOSO)
# =============================================================================
print("\n" + "="*70)
print("CROSS-PATIENT (LOSO)")
print("="*70)

def compute_loso(feature_cols, modality_name):
    results = []
    for test_subj in df_all['subject'].unique():
        train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
        
        if len(train_df) < 50 or len(test_df) < 20:
            continue
        
        X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            results.append({'subject': test_subj, 'r': r})
            print(f"  {test_subj} ({modality_name}): r={r:.2f}")
    
    return results

print("\nIMU LOSO:")
loso_imu = compute_loso(imu_cols, "IMU")
mean_loso_imu = np.mean([r['r'] for r in loso_imu])

print("\nPPG LOSO:")
loso_ppg = compute_loso(ppg_cols, "PPG")
mean_loso_ppg = np.mean([r['r'] for r in loso_ppg])

print("\nEDA LOSO:")
loso_eda = compute_loso(eda_cols, "EDA")
mean_loso_eda = np.mean([r['r'] for r in loso_eda])

print(f"\n--- MEAN CROSS-PATIENT (LOSO) ---")
print(f"  IMU: r = {mean_loso_imu:.2f}")
print(f"  PPG: r = {mean_loso_ppg:.2f}")
print(f"  EDA: r = {mean_loso_eda:.2f}")

# =============================================================================
# PLOT: WITHIN VS CROSS - 2 COLORS ONLY
# =============================================================================
print("\n" + "="*70)
print("CREATING FIXED PLOTS")
print("="*70)

print("\nPlot 7: Within vs Cross (2 colors)...")

fig, ax = plt.subplots(figsize=(10, 6))

modalities = ['IMU', 'PPG', 'EDA']
within_vals = [mean_within_imu, mean_within_ppg, mean_within_eda]
cross_vals = [mean_loso_imu, mean_loso_ppg, mean_loso_eda]

x = np.arange(len(modalities))
width = 0.35

bars1 = ax.bar(x - width/2, within_vals, width, label='Within-Patient', 
               color=COLORS['within'], alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, cross_vals, width, label='Cross-Patient (LOSO)', 
               color=COLORS['cross'], alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Pearson r', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(modalities, fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=11)

# Value labels - positioned to not overlap
for bar, val in zip(bars1, within_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, cross_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '7_within_vs_cross.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 7_within_vs_cross.png")

# =============================================================================
# LOAD RAW SIGNALS AND FIX ALIGNMENT
# =============================================================================
print("\nPlot 16: Multimodal Overview (fixed)...")

def load_gz_csv(path):
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)

base_path = Path('/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1')

# Load all data
ppg_path = list((base_path / 'corsano_wrist_ppg2_green_6').glob('*.csv.gz'))[0]
ppg_raw = load_gz_csv(ppg_path)

eda_path = list((base_path / 'corsano_bioz_emography').glob('*.csv.gz'))[0]
eda_raw = load_gz_csv(eda_path)

imu_path = list((base_path / 'corsano_wrist_acc').glob('*.csv.gz'))[0]
imu_raw = load_gz_csv(imu_path)

rr_path = list((base_path / 'corsano_bioz_rr_interval').glob('*.csv.gz'))[0]
rr_raw = load_gz_csv(rr_path)

# Find common time range
ppg_start = ppg_raw['time'].min()
ppg_end = ppg_raw['time'].max()
imu_start = imu_raw['time'].min()
imu_end = imu_raw['time'].max()
eda_start = eda_raw['time'].min()
eda_end = eda_raw['time'].max()
rr_start = rr_raw['time'].min()
rr_end = rr_raw['time'].max()

print(f"  PPG: {ppg_start:.0f} - {ppg_end:.0f} ({(ppg_end-ppg_start)/60:.1f} min)")
print(f"  IMU: {imu_start:.0f} - {imu_end:.0f} ({(imu_end-imu_start)/60:.1f} min)")
print(f"  EDA: {eda_start:.0f} - {eda_end:.0f} ({(eda_end-eda_start)/60:.1f} min)")
print(f"  RR:  {rr_start:.0f} - {rr_end:.0f} ({(rr_end-rr_start)/60:.1f} min)")

# Use 2 minutes of data, starting from common start
common_start = max(ppg_start, imu_start, eda_start, rr_start)
duration_sec = 120  # 2 minutes

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# PPG - convert time to relative seconds
ppg_window = ppg_raw[(ppg_raw['time'] >= common_start) & (ppg_raw['time'] < common_start + duration_sec)].copy()
ppg_window['t'] = ppg_window['time'] - common_start
axes[0].plot(ppg_window['t'], ppg_window['value'], color='#A23B72', linewidth=0.3)
axes[0].set_ylabel('PPG')
axes[0].set_title('Photoplethysmography (PPG)')

# EDA
eda_col = 'stress_skin' if 'stress_skin' in eda_raw.columns else 'cc'
eda_window = eda_raw[(eda_raw['time'] >= common_start) & (eda_raw['time'] < common_start + duration_sec)].copy()
eda_window['t'] = eda_window['time'] - common_start
axes[1].plot(eda_window['t'], eda_window[eda_col], color='#F18F01', linewidth=1, marker='o', markersize=3)
axes[1].set_ylabel('EDA')
axes[1].set_title('Electrodermal Activity (EDA)')

# IMU
x_col = 'accX' if 'accX' in imu_raw.columns else 'x'
y_col = 'accY' if 'accY' in imu_raw.columns else 'y'
z_col = 'accZ' if 'accZ' in imu_raw.columns else 'z'
imu_window = imu_raw[(imu_raw['time'] >= common_start) & (imu_raw['time'] < common_start + duration_sec)].copy()
imu_window['t'] = imu_window['time'] - common_start
imu_window['mag'] = np.sqrt(imu_window[x_col].astype(float)**2 + imu_window[y_col].astype(float)**2 + imu_window[z_col].astype(float)**2)
axes[2].plot(imu_window['t'], imu_window['mag'], color='#2E86AB', linewidth=0.3)
axes[2].set_ylabel('|Acc|')
axes[2].set_title('Accelerometer (IMU)')

# Heart Rate from RR
rr_clean = rr_raw[(rr_raw['rr'] > 400) & (rr_raw['rr'] < 1500)]
rr_window = rr_clean[(rr_clean['time'] >= common_start) & (rr_clean['time'] < common_start + duration_sec)].copy()
rr_window['t'] = rr_window['time'] - common_start
rr_window['hr'] = 60000 / rr_window['rr']
axes[3].plot(rr_window['t'], rr_window['hr'], color='#DC3545', linewidth=1, marker='o', markersize=2)
axes[3].set_ylabel('HR (bpm)')
axes[3].set_xlabel('Time (seconds)')
axes[3].set_title('Heart Rate (from RR Intervals)')

# Set common x-axis
for ax in axes:
    ax.set_xlim(0, duration_sec)

plt.tight_layout()
plt.savefig(OUT_DIR / '16_multimodal_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 16_multimodal_overview.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
METHODOLOGY:
- Within-Patient: For each subject, 70/30 train/test split, train RF, predict, compute r
                  Then AVERAGE r across all 5 subjects
- Cross-Patient (LOSO): Train on 4 subjects, test on 1 (leave-one-subject-out)
                        Then AVERAGE r across all 5 folds

RESULTS:
┌───────────┬─────────────────┬─────────────────┬───────────┐
│ Modality  │ Within-Patient  │ Cross-Patient   │ Gap (Δr)  │
├───────────┼─────────────────┼─────────────────┼───────────┤
│ IMU       │ {mean_within_imu:>6.2f}          │ {mean_loso_imu:>6.2f}          │ {mean_within_imu - mean_loso_imu:>6.2f}    │
│ PPG       │ {mean_within_ppg:>6.2f}          │ {mean_loso_ppg:>6.2f}          │ {mean_within_ppg - mean_loso_ppg:>6.2f}    │
│ EDA       │ {mean_within_eda:>6.2f}          │ {mean_loso_eda:>6.2f}          │ {mean_within_eda - mean_loso_eda:>6.2f}    │
└───────────┴─────────────────┴─────────────────┴───────────┘

INTERPRETATION:
- Within-patient r: How well can we predict Borg FOR THE SAME PERSON
  (70% training, 30% test, same person)
  
- Cross-patient r: How well can we predict Borg FOR A NEW PERSON
  (trained on other 4 people, tested on held-out person)
  
- Gap: How much performance drops when generalizing to new people
  SMALLER gap = better generalization
  
- IMU has SMALLEST gap ({mean_within_imu - mean_loso_imu:.2f}) = best generalization
- EDA has LARGEST gap ({mean_within_eda - mean_loso_eda:.2f}) = worst generalization
  (EDA patterns are highly individual, don't transfer)
""")

# Check if EDA within-patient is suspiciously high
if mean_within_eda > 0.9:
    print("""
⚠️  WARNING: EDA within-patient r > 0.9 is suspiciously high!
    Possible reasons:
    1. Small test sets (only ~80 samples per subject) → overfitting
    2. EDA features may have temporal autocorrelation artifacts
    3. The 70/30 split doesn't account for temporal ordering
    
    The cross-patient result (r={:.2f}) is the honest metric.
    Within-patient just shows that EDA is "learnable" for one person
    but doesn't generalize to others.
""".format(mean_loso_eda))
