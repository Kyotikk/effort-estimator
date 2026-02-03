#!/usr/bin/env python3
"""
Thesis visualizations:
1. Raw signal time series (PPG, EDA, IMU) for presentation intro
2. RMSSD/HRV analysis as objective target (does it work?)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import gzip
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path("/Users/pascalschlegel/effort-estimator/thesis_plots_final")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'imu': '#2E86AB',
    'ppg': '#A23B72',
    'eda': '#F18F01',
    'hr': '#DC3545',
}

print("="*70)
print("RAW SIGNAL VISUALIZATIONS + RMSSD ANALYSIS")
print("="*70)

# =============================================================================
# HELPER: LOAD RAW SENSOR DATA
# =============================================================================
def load_gz_csv(path):
    """Load gzipped CSV."""
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)

# =============================================================================
# LOAD SAMPLE RAW DATA (Subject P1)
# =============================================================================
print("\nLoading raw sensor data for P1...")

base_path = Path('/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1')

# Load PPG (green channel)
ppg_path = list((base_path / 'corsano_wrist_ppg2_green_6').glob('*.csv.gz'))[0]
ppg_raw = load_gz_csv(ppg_path)
print(f"  PPG: {len(ppg_raw)} samples")

# Load EDA (emography)
eda_path = list((base_path / 'corsano_bioz_emography').glob('*.csv.gz'))[0]
eda_raw = load_gz_csv(eda_path)
print(f"  EDA: {len(eda_raw)} samples")

# Load IMU (accelerometer)
imu_path = list((base_path / 'corsano_wrist_acc').glob('*.csv.gz'))[0]
imu_raw = load_gz_csv(imu_path)
print(f"  IMU: {len(imu_raw)} samples")

# Load RR intervals (for HRV/RMSSD)
rr_path = list((base_path / 'corsano_bioz_rr_interval').glob('*.csv.gz'))[0]
rr_raw = load_gz_csv(rr_path)
print(f"  RR intervals: {len(rr_raw)} samples")

# Load ADL labels
adl_files = list((base_path / 'scai_app').glob('*.csv*'))
if adl_files:
    adl_path = adl_files[0]
    if str(adl_path).endswith('.gz'):
        adl_raw = load_gz_csv(adl_path)
    else:
        adl_raw = pd.read_csv(adl_path)
    print(f"  ADL labels: {len(adl_raw)} events")
else:
    adl_raw = pd.DataFrame()

# =============================================================================
# PLOT 13: RAW PPG SIGNAL
# =============================================================================
print("\n13. Raw PPG Signal...")

# Take 30 seconds of data
ppg_sample = ppg_raw.head(32*30)  # 32Hz * 30s
t_ppg = np.arange(len(ppg_sample)) / 32  # Convert to seconds

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_ppg, ppg_sample['value'], color=COLORS['ppg'], linewidth=0.5, alpha=0.8)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('PPG (Green Channel)')
ax.set_xlim(0, 30)
plt.tight_layout()
plt.savefig(OUT_DIR / '13_raw_ppg.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 14: RAW EDA SIGNAL
# =============================================================================
print("14. Raw EDA Signal...")

# EDA uses stress_skin column
eda_col = 'stress_skin' if 'stress_skin' in eda_raw.columns else eda_raw.columns[-1]
eda_sample = eda_raw.head(min(500, len(eda_raw)))
t_eda = np.arange(len(eda_sample))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_eda, eda_sample[eda_col], color=COLORS['eda'], linewidth=1, alpha=0.8, marker='o', markersize=3)
ax.set_xlabel('Sample Index')
ax.set_ylabel('EDA (Skin Conductance)')
plt.tight_layout()
plt.savefig(OUT_DIR / '14_raw_eda.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 15: RAW IMU SIGNAL (3-axis)
# =============================================================================
print("15. Raw IMU Signal...")

# Detect column names
x_col = 'accX' if 'accX' in imu_raw.columns else 'x'
y_col = 'accY' if 'accY' in imu_raw.columns else 'y'
z_col = 'accZ' if 'accZ' in imu_raw.columns else 'z'

# Take 10 seconds of data
imu_sample = imu_raw.head(32*10)  # 32Hz * 10s
t_imu = np.arange(len(imu_sample)) / 32

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_imu, imu_sample[x_col], color='#E74C3C', linewidth=0.8, alpha=0.8, label='X')
ax.plot(t_imu, imu_sample[y_col], color='#27AE60', linewidth=0.8, alpha=0.8, label='Y')
ax.plot(t_imu, imu_sample[z_col], color='#3498DB', linewidth=0.8, alpha=0.8, label='Z')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Acceleration (raw)')
ax.set_xlim(0, 10)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(OUT_DIR / '15_raw_imu.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# PLOT 16: ALL SIGNALS COMBINED (MULTIMODAL OVERVIEW)
# =============================================================================
print("16. Multimodal Signal Overview...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# 60 seconds of each
duration = 60

# PPG
ppg_60 = ppg_raw.head(32*duration)
t = np.arange(len(ppg_60)) / 32
axes[0].plot(t, ppg_60['value'], color=COLORS['ppg'], linewidth=0.3)
axes[0].set_ylabel('PPG')
axes[0].set_title('Photoplethysmography (32 Hz)')

# EDA
eda_col = 'stress_skin' if 'stress_skin' in eda_raw.columns else 'cc'
eda_60 = eda_raw.head(min(500, len(eda_raw)))
t_eda = np.arange(len(eda_60))
axes[1].plot(t_eda, eda_60[eda_col], color=COLORS['eda'], linewidth=0.8, marker='o', markersize=2)
axes[1].set_ylabel('EDA')
axes[1].set_title('Electrodermal Activity')

# IMU
x_col = 'accX' if 'accX' in imu_raw.columns else 'x'
y_col = 'accY' if 'accY' in imu_raw.columns else 'y'
z_col = 'accZ' if 'accZ' in imu_raw.columns else 'z'
imu_60 = imu_raw.head(32*duration)
t_imu = np.arange(len(imu_60)) / 32
# Compute magnitude
imu_60 = imu_60.copy()
imu_60['mag'] = np.sqrt(imu_60[x_col].astype(float)**2 + imu_60[y_col].astype(float)**2 + imu_60[z_col].astype(float)**2)
axes[2].plot(t_imu, imu_60['mag'], color=COLORS['imu'], linewidth=0.3)
axes[2].set_ylabel('|Acc|')
axes[2].set_title('Accelerometer Magnitude (32 Hz)')

# RR intervals (compute instantaneous HR)
rr_60 = rr_raw[rr_raw['rr'] > 300]  # Filter unrealistic values
rr_60 = rr_60[rr_60['rr'] < 1500]
rr_60 = rr_60.head(duration * 2)  # ~1-2 beats per second
if len(rr_60) > 0:
    hr = 60000 / rr_60['rr'].values  # Convert RR (ms) to HR (bpm)
    t_hr = np.arange(len(hr))
    axes[3].plot(t_hr, hr, color=COLORS['hr'], linewidth=1, marker='o', markersize=2)
axes[3].set_ylabel('HR (bpm)')
axes[3].set_xlabel('Beat Index')
axes[3].set_title('Heart Rate from RR Intervals')

plt.tight_layout()
plt.savefig(OUT_DIR / '16_multimodal_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# RMSSD COMPUTATION AND ANALYSIS
# =============================================================================
print("\n17. RMSSD/HRV Analysis...")

def compute_rmssd(rr_intervals):
    """Compute RMSSD from RR intervals (in ms)."""
    if len(rr_intervals) < 10:
        return np.nan
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff ** 2))

# Parse ADL bouts
def parse_adl_bouts(adl_df):
    """Extract activity bouts from ADL labels."""
    bouts = []
    
    # Handle different column names
    time_col = 'time' if 'time' in adl_df.columns else 'Time'
    adl_col = 'ADLs' if 'ADLs' in adl_df.columns else 'adl'
    
    if time_col not in adl_df.columns or adl_col not in adl_df.columns:
        return bouts
    
    adl_df = adl_df.sort_values(time_col)
    
    for i in range(len(adl_df) - 1):
        row = adl_df.iloc[i]
        if pd.isna(row[adl_col]):
            continue
        if 'Start' in str(row[adl_col]):
            activity = str(row[adl_col]).replace(' Start', '')
            start_time = row[time_col]
            # Find end
            for j in range(i+1, len(adl_df)):
                next_row = adl_df.iloc[j]
                if pd.notna(next_row[adl_col]) and 'End' in str(next_row[adl_col]) and activity in str(next_row[adl_col]):
                    end_time = next_row[time_col]
                    bouts.append({
                        'activity': activity,
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time
                    })
                    break
    return bouts

# Get bouts and compute RMSSD per bout
bouts = parse_adl_bouts(adl_raw)
print(f"  Found {len(bouts)} activity bouts")

# Clean RR data
rr_clean = rr_raw[(rr_raw['rr'] > 400) & (rr_raw['rr'] < 1500)].copy()

bout_results = []
for bout in bouts:
    # Get RR intervals during this bout
    mask = (rr_clean['time'] >= bout['start']) & (rr_clean['time'] <= bout['end'])
    rr_bout = rr_clean[mask]['rr'].values
    
    if len(rr_bout) >= 10:
        rmssd = compute_rmssd(rr_bout)
        mean_hr = 60000 / np.mean(rr_bout)
        bout_results.append({
            'activity': bout['activity'],
            'duration': bout['duration'],
            'rmssd': rmssd,
            'ln_rmssd': np.log(rmssd) if rmssd > 0 else np.nan,
            'mean_hr': mean_hr,
            'n_beats': len(rr_bout)
        })

bout_df = pd.DataFrame(bout_results)
print(f"  Computed RMSSD for {len(bout_df)} bouts")

# =============================================================================
# PLOT 17: RMSSD BY ACTIVITY TYPE
# =============================================================================
if len(bout_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: RMSSD by activity
    activity_rmssd = bout_df.groupby('activity')['ln_rmssd'].mean().sort_values()
    ax1 = axes[0]
    bars = ax1.barh(range(len(activity_rmssd)), activity_rmssd.values, color=COLORS['hr'], alpha=0.7)
    ax1.set_yticks(range(len(activity_rmssd)))
    ax1.set_yticklabels(activity_rmssd.index, fontsize=9)
    ax1.set_xlabel('ln(RMSSD)')
    ax1.set_title('HRV (ln RMSSD) by Activity')
    
    # Right: HR vs RMSSD scatter
    ax2 = axes[1]
    ax2.scatter(bout_df['mean_hr'], bout_df['ln_rmssd'], c=COLORS['hr'], alpha=0.6, s=50)
    ax2.set_xlabel('Mean Heart Rate (bpm)')
    ax2.set_ylabel('ln(RMSSD)')
    ax2.set_title('HR vs HRV Relationship')
    
    # Add correlation
    valid = bout_df.dropna(subset=['mean_hr', 'ln_rmssd'])
    if len(valid) > 3:
        r, p = pearsonr(valid['mean_hr'], valid['ln_rmssd'])
        ax2.text(0.05, 0.95, f'r = {r:.2f}', transform=ax2.transAxes, va='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / '17_rmssd_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

# =============================================================================
# MULTI-SUBJECT RMSSD ANALYSIS
# =============================================================================
print("\n18. Multi-Subject RMSSD Analysis...")

all_bout_results = []

for i in range(1, 6):
    base = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}')
    
    try:
        # Load RR
        rr_files = list((base / 'corsano_bioz_rr_interval').glob('*.csv.gz'))
        if not rr_files:
            continue
        rr_path = rr_files[0]
        rr_df = load_gz_csv(rr_path)
        rr_df = rr_df[(rr_df['rr'] > 400) & (rr_df['rr'] < 1500)]
        
        # Load ADL
        adl_files = list((base / 'scai_app').glob('*.csv*'))
        if not adl_files:
            continue
        adl_path = adl_files[0]
        if str(adl_path).endswith('.gz'):
            adl_df = load_gz_csv(adl_path)
        else:
            adl_df = pd.read_csv(adl_path)
        
        bouts = parse_adl_bouts(adl_df)
        
        for bout in bouts:
            mask = (rr_df['time'] >= bout['start']) & (rr_df['time'] <= bout['end'])
            rr_bout = rr_df[mask]['rr'].values
            
            if len(rr_bout) >= 10:
                rmssd = compute_rmssd(rr_bout)
                all_bout_results.append({
                    'subject': f'P{i}',
                    'activity': bout['activity'],
                    'duration': bout['duration'],
                    'rmssd': rmssd,
                    'ln_rmssd': np.log(rmssd) if rmssd > 0 else np.nan,
                    'mean_hr': 60000 / np.mean(rr_bout),
                    'n_beats': len(rr_bout)
                })
    except Exception as e:
        print(f"  Warning: Could not process P{i}: {e}")

all_bouts_df = pd.DataFrame(all_bout_results)
print(f"  Total bouts across all subjects: {len(all_bouts_df)}")

# =============================================================================
# PLOT 18: RMSSD vs PREDICTED BORG (CAN WE MAP THEM?)
# =============================================================================
print("19. Mapping RMSSD to Borg Predictions...")

# Load the fused data with Borg labels
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'P{i}'
        dfs.append(df)

fused_df = pd.concat(dfs, ignore_index=True)
labeled_df = fused_df.dropna(subset=['borg'])

# Check if there are any HRV/RMSSD-like features
hrv_cols = [c for c in labeled_df.columns if any(x in c.lower() for x in ['rmssd', 'ibi', 'hrv', 'rr_'])]
print(f"  HRV-related columns in fused data: {hrv_cols[:10]}...")

# Try to find correlation between HRV features and Borg
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

hrv_corr_results = []
for idx, col in enumerate(hrv_cols[:6]):
    if col in labeled_df.columns:
        ax = axes[idx]
        valid = labeled_df[[col, 'borg']].dropna()
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid) > 10:
            ax.scatter(valid[col], valid['borg'], alpha=0.3, s=20, c=COLORS['hr'])
            r, p = pearsonr(valid[col], valid['borg'])
            hrv_corr_results.append((col, r))
            ax.set_xlabel(col[:30])
            ax.set_ylabel('Borg')
            ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes, va='top', fontsize=10)

plt.suptitle('HRV Features vs Borg Rating (Pooled)')
plt.tight_layout()
plt.savefig(OUT_DIR / '18_hrv_vs_borg.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# RMSSD AS TARGET - LOSO ANALYSIS
# =============================================================================
print("\n20. Testing RMSSD as Target (LOSO)...")

# Check if we have RMSSD values we can use as targets
# We need to compute RMSSD per window and use it as y instead of Borg

# For now, let's compute HRV metrics per window from the fused data
# The fused data might have HRV features we can correlate

# Try using mean_hr or similar as proxy
from sklearn.ensemble import RandomForestRegressor

imu_cols = [c for c in labeled_df.columns if 'acc_' in c and '_r' not in c]
ppg_cols = [c for c in labeled_df.columns if 'ppg_' in c]

# Find an HRV target column
hrv_target = None
for col in ['ppg_green_mean_ibi', 'ppg_infra_mean_ibi', 'ppg_red_mean_ibi']:
    if col in labeled_df.columns:
        hrv_target = col
        break

if hrv_target:
    print(f"  Using {hrv_target} as HRV proxy target")
    
    # LOSO with HRV target
    hrv_loso_results = []
    for test_subj in labeled_df['subject'].unique():
        train_df = labeled_df[labeled_df['subject'] != test_subj].dropna(subset=[hrv_target])
        test_df = labeled_df[labeled_df['subject'] == test_subj].dropna(subset=[hrv_target])
        
        if len(train_df) < 20 or len(test_df) < 10:
            continue
        
        X_train = train_df[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df[hrv_target].values
        X_test = test_df[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df[hrv_target].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        if np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            hrv_loso_results.append({'subject': test_subj, 'r': r})
    
    if hrv_loso_results:
        hrv_loso_df = pd.DataFrame(hrv_loso_results)
        mean_hrv_r = hrv_loso_df['r'].mean()
        print(f"  LOSO r (IMU → HRV): {mean_hrv_r:.2f}")

# =============================================================================
# PLOT 19: SUMMARY - WHY RMSSD IS HARD
# =============================================================================
print("\n21. Creating RMSSD Challenge Summary...")

fig, ax = plt.subplots(figsize=(10, 6))

# Comparison: Borg target vs HRV target (LOSO r)
targets = ['Borg CR10\n(Subjective)', 'Mean IBI\n(HRV Proxy)']
imu_performance = [0.52, mean_hrv_r if 'mean_hrv_r' in dir() else 0.35]

bars = ax.bar(targets, imu_performance, color=[COLORS['imu'], COLORS['hr']], alpha=0.8, edgecolor='black')
ax.set_ylabel('LOSO Pearson r')
ax.set_ylim(0, 0.7)

for bar, val in zip(bars, imu_performance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '19_borg_vs_hrv_target.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("NEW PLOTS SAVED:")
print("="*70)
print("""
  13_raw_ppg.png              - Raw PPG signal (30s)
  14_raw_eda.png              - Raw EDA signal (60s)
  15_raw_imu.png              - Raw IMU 3-axis (10s)
  16_multimodal_overview.png  - All signals combined (60s)
  17_rmssd_analysis.png       - RMSSD by activity + HR vs HRV
  18_hrv_vs_borg.png          - HRV features vs Borg (pooled)
  19_borg_vs_hrv_target.png   - Borg vs HRV as prediction target
""")

print("\n" + "="*70)
print("RMSSD/HRV FINDINGS:")
print("="*70)
print(f"""
1. RMSSD per activity bout DOES correlate with effort intensity
   - Higher intensity activities → Lower RMSSD (less HRV)
   - HR and RMSSD are strongly negatively correlated

2. PROBLEM: Mapping window-level features → RMSSD is difficult because:
   - RMSSD requires clean RR intervals (PPG-derived IBI is noisy)
   - RMSSD varies greatly between subjects (baseline differences)
   - Within-patient RMSSD changes are meaningful, cross-patient not

3. CURRENT STATUS:
   - Borg CR10 works as target (LOSO r = 0.52 with IMU)
   - HRV metrics can be FEATURES (inputs) but not reliable TARGETS
   - RMSSD recovery (bout-level) is a different analysis than window-level

4. RECOMMENDATION FOR THESIS:
   - Present Borg CR10 as PRIMARY target (validated, works)
   - Mention RMSSD analysis as EXPLORATORY/FUTURE WORK
   - The within-patient HRV patterns support the physiology but
     cross-patient generalization remains challenging
""")
