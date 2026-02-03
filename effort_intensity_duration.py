#!/usr/bin/env python3
"""
Effort Index: Intensity × Duration (sublinear)

Key insight:
- Resting for 60s = still easy (low intensity × any duration = low)
- Walking for 10s = moderate (high intensity × short = moderate)  
- Walking for 60s = harder (high intensity × longer = more effort)

Formula: Effort = Intensity × sqrt(duration)
- sqrt() makes duration sublinear: 4x longer = 2x more effort, not 4x
- Low intensity activities stay low regardless of duration
- High intensity activities get harder with duration
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

# =============================================================================
# INTENSITY (how hard is the activity)
# =============================================================================
# HR intensity: elevation above resting baseline
hr_baseline = df['hr_mean'].min()  # resting HR
hr_intensity = df['hr_mean'] - hr_baseline

# IMU intensity: movement vigor (RMS acceleration)
# Subtract gravity baseline (~0.49g for resting)
imu_baseline = df['rms_acc_mag'].min()
imu_intensity = df['rms_acc_mag'] - imu_baseline

# Combined intensity (z-scored)
z_hr = (hr_intensity - hr_intensity.mean()) / hr_intensity.std()
z_imu = (imu_intensity - imu_intensity.mean()) / imu_intensity.std()
intensity = 0.7 * z_hr + 0.3 * z_imu  # HR weighted more (physiological demand)

# =============================================================================
# DURATION SCALING (sublinear)
# =============================================================================
# sqrt(duration) means: 4x longer = 2x more effort
# This prevents duration from dominating
duration_factor = np.sqrt(df['duration_s'])

# Normalize duration factor to mean=1 so it's a multiplier
duration_factor_norm = duration_factor / duration_factor.mean()

# =============================================================================
# EFFORT INDEX = Intensity × Duration Factor
# =============================================================================
# Only positive intensities get amplified by duration
# Negative/low intensity activities stay low
effort_raw = intensity * duration_factor_norm

# Z-score the final effort for interpretability
df['effort_index'] = (effort_raw - effort_raw.mean()) / effort_raw.std()
df['intensity'] = intensity
df['duration_factor'] = duration_factor_norm

# =============================================================================
# COMPARE ALL METRICS
# =============================================================================
print("="*70)
print("EFFORT INDEX: Intensity × sqrt(Duration)")
print("="*70)
print()

# Correlations
metrics = {
    'HR delta only': df['hr_delta'],
    'IMU RMS only': df['rms_acc_mag'],
    'Intensity (no duration)': intensity,
    'Old TLI (load × duration)': df['TLI'],
    'NEW Effort Index': df['effort_index']
}

print("Correlations with Borg CR10:")
print("-"*50)
for name, metric in metrics.items():
    r, p = stats.pearsonr(metric.fillna(0), df['borg'])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:30s} r={r:+.3f} (p={p:.4f}){sig}")

# =============================================================================
# SHOW IT MAKES SENSE
# =============================================================================
print()
print("="*70)
print("SANITY CHECK: Does it make sense now?")
print("="*70)
print()
print(f"{'Activity':<25} {'Dur(s)':>7} {'Intens':>7} {'Dur×':>6} {'Effort':>7} {'Borg':>5}")
print("-"*65)

# Sort by effort
df_sorted = df.sort_values('effort_index', ascending=False)

# Show top activities
for _, row in df_sorted.head(8).iterrows():
    print(f"{row.activity:<25} {row.duration_s:>7.1f} {row.intensity:>+7.2f} {row.duration_factor:>6.2f} {row.effort_index:>+7.2f} {row.borg:>5.1f}")

print("...")

# Show bottom activities (should include Resting!)
for _, row in df_sorted.tail(5).iterrows():
    print(f"{row.activity:<25} {row.duration_s:>7.1f} {row.intensity:>+7.2f} {row.duration_factor:>6.2f} {row.effort_index:>+7.2f} {row.borg:>5.1f}")

# =============================================================================
# KEY COMPARISONS
# =============================================================================
print()
print("="*70)
print("KEY COMPARISONS")
print("="*70)
print()

comparisons = [
    ("Resting", "Stand to Sit", "Resting long vs quick transition"),
    ("Stand", "Level Walking", "Long standing vs walking"),
    ("(Un)button Shirt", "Level Walking", "Short fiddly vs walking"),
]

for act1, act2, desc in comparisons:
    r1 = df[df['activity']==act1].iloc[0] if act1 in df['activity'].values else None
    r2 = df[df['activity']==act2].iloc[0] if act2 in df['activity'].values else None
    if r1 is not None and r2 is not None:
        print(f"{desc}:")
        print(f"  {act1:<20}: {r1.duration_s:5.0f}s, intensity={r1.intensity:+.2f}, effort={r1.effort_index:+.2f}, borg={r1.borg}")
        print(f"  {act2:<20}: {r2.duration_s:5.0f}s, intensity={r2.intensity:+.2f}, effort={r2.effort_index:+.2f}, borg={r2.borg}")
        print()

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Effort vs Borg
ax = axes[0, 0]
ax.scatter(df['effort_index'], df['borg'], alpha=0.7, s=80)
r, p = stats.pearsonr(df['effort_index'], df['borg'])
ax.set_xlabel('Effort Index (Intensity × √Duration)', fontsize=11)
ax.set_ylabel('Borg CR10', fontsize=11)
ax.set_title(f'Effort Index vs Borg\nr = {r:.3f}***', fontsize=12)

# Add regression line
z = np.polyfit(df['effort_index'], df['borg'], 1)
x_line = np.linspace(df['effort_index'].min(), df['effort_index'].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.7)
ax.grid(True, alpha=0.3)

# 2. Intensity vs Duration colored by Borg
ax = axes[0, 1]
sc = ax.scatter(df['duration_s'], df['intensity'], c=df['borg'], cmap='RdYlGn_r', s=80, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Borg CR10')
ax.set_xlabel('Duration (seconds)', fontsize=11)
ax.set_ylabel('Intensity', fontsize=11)
ax.set_title('Intensity vs Duration\n(colored by Borg)', fontsize=12)
ax.grid(True, alpha=0.3)

# Annotate key points
for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking', 'Stand']:
        ax.annotate(row.activity, (row.duration_s, row.intensity), fontsize=8)

# 3. Compare metrics
ax = axes[1, 0]
old_tli_norm = (df['TLI'] - df['TLI'].mean()) / df['TLI'].std()
ax.scatter(old_tli_norm, df['effort_index'], alpha=0.7, s=80)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Old TLI (normalized)', fontsize=11)
ax.set_ylabel('New Effort Index', fontsize=11)
ax.set_title('Old TLI vs New Effort Index', fontsize=12)
ax.grid(True, alpha=0.3)

# Annotate Resting
resting = df[df['activity']=='Resting'].iloc[0]
resting_tli_norm = (resting.TLI - df['TLI'].mean()) / df['TLI'].std()
ax.annotate('Resting', (resting_tli_norm, resting.effort_index), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red'), color='red')

# 4. Activity ranking
ax = axes[1, 1]
top_n = 12
df_top = df_sorted.head(top_n)
colors = plt.cm.RdYlGn_r(df_top['borg'] / df_top['borg'].max())
bars = ax.barh(range(top_n), df_top['effort_index'], color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(df_top['activity'], fontsize=9)
ax.set_xlabel('Effort Index', fontsize=11)
ax.set_title('Top Activities by Effort Index', fontsize=12)
ax.invert_yaxis()

# Add Borg labels
for i, (_, row) in enumerate(df_top.iterrows()):
    ax.text(row.effort_index + 0.1, i, f'Borg={row.borg}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('slides/effort_index_results.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to slides/effort_index_results.png")

# Save results
df_sorted.to_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/effort_index_results.csv', index=False)
print(f"Saved results to effort_index_results.csv")
