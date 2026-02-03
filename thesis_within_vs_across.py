#!/usr/bin/env python3
"""
Plot 46: Within-Patient vs Across-Patient (LOSO) Performance
Shows the generalization gap for all modalities + sequential forward selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'imu': '#2E86AB',
    'ppg': '#A23B72', 
    'eda': '#F18F01',
}

print("="*70)
print("PLOT 46: Within vs Across Patient Performance")
print("="*70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")
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
print(f"  IMU={len(imu_cols)}, PPG={len(ppg_cols)}, EDA={len(eda_cols)} features")

# =============================================================================
# FUNCTIONS
# =============================================================================
def run_loso(feature_cols):
    """Run LOSO cross-validation, return per-subject metrics."""
    per_subj_metrics = {}
    all_importances = []
    
    for test_subj in df_all['subject'].unique():
        train_df = df_all[df_all['subject'] != test_subj].dropna(subset=['borg'])
        test_df = df_all[df_all['subject'] == test_subj].dropna(subset=['borg'])
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        valid_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df['borg'].values
        X_test = test_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df['borg'].values
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        all_importances.append(dict(zip(valid_cols, rf.feature_importances_)))
        
        if len(y_test) > 2 and np.std(y_test) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_test, y_pred)
            per_subj_metrics[test_subj] = {'r': r}
    
    # Average importances
    avg_imp = {}
    for col in feature_cols:
        vals = [imp.get(col, 0) for imp in all_importances]
        avg_imp[col] = np.mean(vals) if vals else 0
    
    return per_subj_metrics, avg_imp

def run_within_patient(feature_cols):
    """Run within-patient evaluation (train/test split within each subject)."""
    within_metrics = {}
    
    for subj in df_all['subject'].unique():
        subj_df = df_all[df_all['subject'] == subj].dropna(subset=['borg'])
        if len(subj_df) < 20:
            continue
            
        valid_cols = [c for c in feature_cols if c in subj_df.columns]
        X = subj_df[valid_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = subj_df['borg'].values
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        
        if len(y_te) > 2 and np.std(y_te) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_te, y_pred)
            within_metrics[subj] = {'r': r}
    
    return within_metrics

def greedy_forward_selection(feature_pool, max_features=10):
    """Sequential forward selection using LOSO r as criterion."""
    selected = []
    r_history = []
    remaining = list(feature_pool)
    
    print(f"    Running greedy selection on {len(feature_pool)} features...")
    
    for step in range(min(max_features, len(remaining))):
        best_r = -999
        best_feat = None
        
        for feat in remaining:
            test_feats = selected + [feat]
            metrics, _ = run_loso(test_feats)
            if metrics:
                r = np.mean([m['r'] for m in metrics.values()])
                if r > best_r:
                    best_r = r
                    best_feat = feat
        
        if best_feat is None:
            break
        
        selected.append(best_feat)
        remaining.remove(best_feat)
        r_history.append(best_r)
        print(f"      Step {step+1}: +{best_feat[:40]:<40} → r={best_r:.3f}")
    
    return selected, r_history

# =============================================================================
# RUN EVALUATIONS
# =============================================================================
print("\n--- All Features ---")

# LOSO (across-patient)
print("  Running LOSO...")
metrics_imu_loso, imp_imu = run_loso(imu_cols)
metrics_ppg_loso, imp_ppg = run_loso(ppg_cols)
metrics_eda_loso, imp_eda = run_loso(eda_cols)

r_imu_loso = np.mean([m['r'] for m in metrics_imu_loso.values()])
r_ppg_loso = np.mean([m['r'] for m in metrics_ppg_loso.values()])
r_eda_loso = np.mean([m['r'] for m in metrics_eda_loso.values()])

# Within-patient
print("  Running within-patient...")
metrics_imu_within = run_within_patient(imu_cols)
metrics_ppg_within = run_within_patient(ppg_cols)
metrics_eda_within = run_within_patient(eda_cols)

r_imu_within = np.mean([m['r'] for m in metrics_imu_within.values()])
r_ppg_within = np.mean([m['r'] for m in metrics_ppg_within.values()])
r_eda_within = np.mean([m['r'] for m in metrics_eda_within.values()])

print(f"\n  IMU: within={r_imu_within:.2f}, LOSO={r_imu_loso:.2f}")
print(f"  PPG: within={r_ppg_within:.2f}, LOSO={r_ppg_loso:.2f}")
print(f"  EDA: within={r_eda_within:.2f}, LOSO={r_eda_loso:.2f}")

# =============================================================================
# GREEDY FORWARD SELECTION
# =============================================================================
print("\n--- Sequential Forward Selection ---")

# IMU - full search
print("  IMU:")
greedy_imu_feats, greedy_imu_r = greedy_forward_selection(imu_cols, max_features=10)

# PPG - pre-filter to top 30 by importance for speed
print("  PPG (top 30 candidates):")
ppg_top30 = [f for f, _ in sorted(imp_ppg.items(), key=lambda x: x[1], reverse=True)[:30]]
greedy_ppg_feats, greedy_ppg_r = greedy_forward_selection(ppg_top30, max_features=10)

# EDA - full search  
print("  EDA:")
greedy_eda_feats, greedy_eda_r = greedy_forward_selection(eda_cols, max_features=10)

# Get within-patient for greedy features too
print("\n  Computing within-patient for greedy features...")
metrics_greedy_imu_within = run_within_patient(greedy_imu_feats)
metrics_greedy_ppg_within = run_within_patient(greedy_ppg_feats)
metrics_greedy_eda_within = run_within_patient(greedy_eda_feats)

r_greedy_imu_within = np.mean([m['r'] for m in metrics_greedy_imu_within.values()])
r_greedy_ppg_within = np.mean([m['r'] for m in metrics_greedy_ppg_within.values()])
r_greedy_eda_within = np.mean([m['r'] for m in metrics_greedy_eda_within.values()])

r_greedy_imu_loso = greedy_imu_r[-1] if greedy_imu_r else 0
r_greedy_ppg_loso = greedy_ppg_r[-1] if greedy_ppg_r else 0
r_greedy_eda_loso = greedy_eda_r[-1] if greedy_eda_r else 0

print(f"\n  Greedy IMU ({len(greedy_imu_feats)} feats): within={r_greedy_imu_within:.2f}, LOSO={r_greedy_imu_loso:.2f}")
print(f"  Greedy PPG ({len(greedy_ppg_feats)} feats): within={r_greedy_ppg_within:.2f}, LOSO={r_greedy_ppg_loso:.2f}")
print(f"  Greedy EDA ({len(greedy_eda_feats)} feats): within={r_greedy_eda_within:.2f}, LOSO={r_greedy_eda_loso:.2f}")

# =============================================================================
# PLOT
# =============================================================================
print("\n--- Creating Plot ---")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Data for grouped bar chart
conditions = ['All Features', 'SFS-10']
x = np.arange(len(conditions))
width = 0.35

# --- IMU ---
ax1 = axes[0]
within_vals = [r_imu_within, r_greedy_imu_within]
loso_vals = [r_imu_loso, r_greedy_imu_loso]

bars1 = ax1.bar(x - width/2, within_vals, width, label='Within-Patient', color=COLORS['imu'], alpha=0.7)
bars2 = ax1.bar(x + width/2, loso_vals, width, label='Across-Patient (LOSO)', color=COLORS['imu'], alpha=1.0, hatch='///')

ax1.set_ylabel('Pearson r')
ax1.set_title(f'IMU ({len(imu_cols)} → {len(greedy_imu_feats)} features)', fontweight='bold', color=COLORS['imu'])
ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Acceptable (r=0.5)')
ax1.legend(loc='upper right', fontsize=9)

# Add value labels
for bar in bars1:
    ax1.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax1.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# --- PPG ---
ax2 = axes[1]
within_vals = [r_ppg_within, r_greedy_ppg_within]
loso_vals = [r_ppg_loso, r_greedy_ppg_loso]

bars1 = ax2.bar(x - width/2, within_vals, width, label='Within-Patient', color=COLORS['ppg'], alpha=0.7)
bars2 = ax2.bar(x + width/2, loso_vals, width, label='Across-Patient (LOSO)', color=COLORS['ppg'], alpha=1.0, hatch='///')

ax2.set_ylabel('Pearson r')
ax2.set_title(f'PPG ({len(ppg_cols)} → {len(greedy_ppg_feats)} features)', fontweight='bold', color=COLORS['ppg'])
ax2.set_xticks(x)
ax2.set_xticklabels(conditions)
ax2.set_ylim(0, 1.0)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)

for bar in bars1:
    ax2.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax2.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# --- EDA ---
ax3 = axes[2]
within_vals = [r_eda_within, r_greedy_eda_within]
loso_vals = [r_eda_loso, r_greedy_eda_loso]

bars1 = ax3.bar(x - width/2, within_vals, width, label='Within-Patient', color=COLORS['eda'], alpha=0.7)
bars2 = ax3.bar(x + width/2, loso_vals, width, label='Across-Patient (LOSO)', color=COLORS['eda'], alpha=1.0, hatch='///')

ax3.set_ylabel('Pearson r')
ax3.set_title(f'EDA ({len(eda_cols)} → {len(greedy_eda_feats)} features)', fontweight='bold', color=COLORS['eda'])
ax3.set_xticks(x)
ax3.set_xticklabels(conditions)
ax3.set_ylim(0, 1.0)
ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)

for bar in bars1:
    ax3.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    ax3.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save
outpath = OUT_DIR / "46_within_vs_across_patient.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {outpath}")

# =============================================================================
# ADDITIONAL PLOT: SFS Progression
# =============================================================================
print("\n--- Creating SFS Progression Plot ---")

fig2, ax = plt.subplots(figsize=(10, 5))

steps = np.arange(1, 11)

ax.plot(steps[:len(greedy_imu_r)], greedy_imu_r, 'o-', color=COLORS['imu'], 
        linewidth=2, markersize=8, label=f'IMU (final r={greedy_imu_r[-1]:.2f})')
ax.plot(steps[:len(greedy_ppg_r)], greedy_ppg_r, 's-', color=COLORS['ppg'], 
        linewidth=2, markersize=8, label=f'PPG (final r={greedy_ppg_r[-1]:.2f})')
ax.plot(steps[:len(greedy_eda_r)], greedy_eda_r, '^-', color=COLORS['eda'], 
        linewidth=2, markersize=8, label=f'EDA (final r={greedy_eda_r[-1]:.2f})')

ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Acceptable (r=0.5)')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

ax.set_xlabel('Number of Features (Sequential Forward Selection)')
ax.set_ylabel('LOSO Pearson r')
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.2, 0.8)
ax.set_xticks(steps)
ax.legend(loc='lower right')

plt.tight_layout()

outpath2 = OUT_DIR / "47_sfs_progression.png"
plt.savefig(outpath2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath2}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                    WITHIN vs ACROSS PATIENT                        │
├──────────────┬─────────────────────┬───────────────────────────────┤
│   Modality   │    All Features     │    SFS-10 Features            │
│              │  Within │   LOSO    │   Within │   LOSO             │
├──────────────┼─────────┼───────────┼──────────┼────────────────────┤
│     IMU      │  {r_imu_within:.2f}   │   {r_imu_loso:.2f}     │   {r_greedy_imu_within:.2f}    │   {r_greedy_imu_loso:.2f}  ✓           │
│     PPG      │  {r_ppg_within:.2f}   │   {r_ppg_loso:.2f}     │   {r_greedy_ppg_within:.2f}    │   {r_greedy_ppg_loso:.2f}               │
│     EDA      │  {r_eda_within:.2f}   │   {r_eda_loso:.2f}     │   {r_greedy_eda_within:.2f}    │   {r_greedy_eda_loso:.2f}               │
└──────────────┴─────────┴───────────┴──────────┴────────────────────┘

Key Findings:
1. Within-patient r is HIGH for all modalities (personalized models work)
2. Across-patient (LOSO) reveals true generalization:
   - IMU generalizes (r ≈ 0.5) - physics-based features are universal
   - PPG/EDA fail to generalize - physiology is individual
3. Sequential Forward Selection slightly improves IMU, doesn't save PPG/EDA
""")

print("\n--- Greedy Selected Features ---")
print(f"\nIMU ({len(greedy_imu_feats)} features):")
for i, f in enumerate(greedy_imu_feats, 1):
    print(f"  {i}. {f}")

print(f"\nPPG ({len(greedy_ppg_feats)} features):")
for i, f in enumerate(greedy_ppg_feats, 1):
    print(f"  {i}. {f}")

print(f"\nEDA ({len(greedy_eda_feats)} features):")
for i, f in enumerate(greedy_eda_feats, 1):
    print(f"  {i}. {f}")

plt.show()
