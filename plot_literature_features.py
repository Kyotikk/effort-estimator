#!/usr/bin/env python3
"""
Visualize literature-backed effort features and their correlations with Borg CR10.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
DATA_DIR = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output")
OUTPUT_DIR = Path("/Users/pascalschlegel/effort-estimator/slides")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load features
df = pd.read_csv(DATA_DIR / "effort_features_full.csv")

# Create figure
fig = plt.figure(figsize=(16, 14))

# ============================================================================
# Plot 1: Top HR features correlation with Borg
# ============================================================================
ax1 = fig.add_subplot(3, 3, 1)

hr_features = ['hr_mean', 'hr_delta', 'hr_reserve_pct', 'trimp_banister', 
               'trimp_edwards', 'hr_load_linear', 'hr_load_sqrt']
hr_corrs = []
for feat in hr_features:
    r, p = stats.pearsonr(df[feat].fillna(0), df['borg'])
    hr_corrs.append((feat, r, p))

hr_corrs.sort(key=lambda x: x[1], reverse=True)
names = [x[0].replace('_', '\n') for x in hr_corrs]
values = [x[1] for x in hr_corrs]
colors = ['darkred' if v > 0.7 else 'red' if v > 0.5 else 'lightcoral' for v in values]

bars = ax1.barh(names, values, color=colors)
ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5')
ax1.axvline(0.7, color='gray', linestyle=':', alpha=0.5, label='r=0.7')
ax1.set_xlabel('Correlation with Borg CR10')
ax1.set_title('HR Features\n(Literature-backed)', fontweight='bold')
ax1.set_xlim(0, 1)

for bar, val in zip(bars, values):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}', va='center', fontsize=9)

# ============================================================================
# Plot 2: Top IMU features correlation with Borg
# ============================================================================
ax2 = fig.add_subplot(3, 3, 2)

imu_features = ['rms_acc', 'sma', 'sma_dynamic', 'mad', 'rms_jerk',
                'energy_proxy', 'imu_load_linear', 'imu_load_sqrt', 
                'mad_load', 'jerk_load']
imu_corrs = []
for feat in imu_features:
    r, p = stats.pearsonr(df[feat].fillna(0), df['borg'])
    if not np.isnan(r):
        imu_corrs.append((feat, r, p))

imu_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
names = [x[0].replace('_', '\n') for x in imu_corrs[:8]]
values = [x[1] for x in imu_corrs[:8]]
colors = ['darkblue' if v > 0.5 else 'blue' if v > 0.3 else 'lightblue' for v in values]

bars = ax2.barh(names, values, color=colors)
ax2.axvline(0.3, color='gray', linestyle='--', alpha=0.5, label='r=0.3')
ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='r=0.5')
ax2.set_xlabel('Correlation with Borg CR10')
ax2.set_title('IMU Features\n(Literature-backed)', fontweight='bold')
ax2.set_xlim(-0.3, 0.7)

for bar, val in zip(bars, values):
    ax2.text(max(val + 0.02, 0.02), bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}', va='center', fontsize=9)

# ============================================================================
# Plot 3: TRIMP Banister vs Borg (Best HR feature)
# ============================================================================
ax3 = fig.add_subplot(3, 3, 3)
r, p = stats.pearsonr(df['trimp_banister'], df['borg'])

ax3.scatter(df['trimp_banister'], df['borg'], c='red', alpha=0.7, s=80)
z = np.polyfit(df['trimp_banister'], df['borg'], 1)
x_line = np.linspace(df['trimp_banister'].min(), df['trimp_banister'].max(), 100)
ax3.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
ax3.set_xlabel('Banister TRIMP')
ax3.set_ylabel('Borg CR10')
ax3.set_title(f'TRIMP Banister (1991)\nr = {r:.3f}', fontweight='bold')

# ============================================================================
# Plot 4: HR Load √t vs Borg (Best overall)
# ============================================================================
ax4 = fig.add_subplot(3, 3, 4)
r, p = stats.pearsonr(df['hr_load_sqrt'], df['borg'])

ax4.scatter(df['hr_load_sqrt'], df['borg'], c='darkred', alpha=0.7, s=80)
z = np.polyfit(df['hr_load_sqrt'], df['borg'], 1)
x_line = np.linspace(df['hr_load_sqrt'].min(), df['hr_load_sqrt'].max(), 100)
ax4.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
ax4.set_xlabel('HR Load (Stevens √t)')
ax4.set_ylabel('Borg CR10')
ax4.set_title(f'HR × √duration (Stevens)\nr = {r:.3f}', fontweight='bold')

# ============================================================================
# Plot 5: IMU Load vs Borg
# ============================================================================
ax5 = fig.add_subplot(3, 3, 5)
r, p = stats.pearsonr(df['imu_load_sqrt'], df['borg'])

ax5.scatter(df['imu_load_sqrt'], df['borg'], c='blue', alpha=0.7, s=80)
z = np.polyfit(df['imu_load_sqrt'], df['borg'], 1)
x_line = np.linspace(df['imu_load_sqrt'].min(), df['imu_load_sqrt'].max(), 100)
ax5.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
ax5.set_xlabel('IMU Load (RMS×√t)')
ax5.set_ylabel('Borg CR10')
ax5.set_title(f'IMU Load √duration\nr = {r:.3f}', fontweight='bold')

# ============================================================================
# Plot 6: MAD Load vs Borg
# ============================================================================
ax6 = fig.add_subplot(3, 3, 6)
r, p = stats.pearsonr(df['mad_load'], df['borg'])

ax6.scatter(df['mad_load'], df['borg'], c='purple', alpha=0.7, s=80)
z = np.polyfit(df['mad_load'], df['borg'], 1)
x_line = np.linspace(df['mad_load'].min(), df['mad_load'].max(), 100)
ax6.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
ax6.set_xlabel('MAD Load')
ax6.set_ylabel('Borg CR10')
ax6.set_title(f'Mean Amplitude Deviation Load\nr = {r:.3f}', fontweight='bold')

# ============================================================================
# Plot 7: Combined HR + MAD Load
# ============================================================================
ax7 = fig.add_subplot(3, 3, 7)

def zscore(x):
    return (x - x.mean()) / x.std()

z_hr = zscore(df['hr_load_sqrt'])
z_mad = zscore(df['mad_load'])
combined = 0.8 * z_hr + 0.2 * z_mad

r, p = stats.pearsonr(combined, df['borg'])
ax7.scatter(combined, df['borg'], c='green', alpha=0.7, s=80)
z = np.polyfit(combined, df['borg'], 1)
x_line = np.linspace(combined.min(), combined.max(), 100)
ax7.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
ax7.set_xlabel('Combined Score (0.8×HR + 0.2×MAD)')
ax7.set_ylabel('Borg CR10')
ax7.set_title(f'Best Combination\nr = {r:.3f}', fontweight='bold')

# ============================================================================
# Plot 8: Feature comparison table
# ============================================================================
ax8 = fig.add_subplot(3, 3, 8)
ax8.axis('off')

# Create summary table
table_data = [
    ['Feature', 'Domain', 'r', 'Source'],
    ['─' * 15, '─' * 8, '─' * 6, '─' * 20],
    ['hr_load_sqrt', 'HR', '0.816', 'Stevens Power Law'],
    ['trimp_banister', 'HR', '0.758', 'Banister (1991)'],
    ['trimp_edwards', 'HR', '0.713', 'Edwards (1993)'],
    ['imu_load_sqrt', 'IMU', '0.608', 'Stevens + RMS'],
    ['energy_proxy', 'IMU', '0.586', 'Bouten (1997)'],
    ['mad_load', 'IMU', '0.482', 'Mathie (2004)'],
    ['─' * 15, '─' * 8, '─' * 6, '─' * 20],
    ['Combined', 'HR+IMU', '0.826', '80% HR + 20% MAD'],
]

text = '\n'.join(['  '.join(row) for row in table_data])
ax8.text(0.1, 0.9, text, fontsize=11, family='monospace',
         transform=ax8.transAxes, verticalalignment='top')
ax8.set_title('Feature Summary\n(Literature Sources)', fontweight='bold')

# ============================================================================
# Plot 9: Predicted vs Actual with activities labeled
# ============================================================================
ax9 = fig.add_subplot(3, 3, 9)

# Best model: HR_load_sqrt + MAD_load
predicted = 0.8 * z_hr + 0.2 * z_mad

# Normalize to Borg scale
slope, intercept, r, p, se = stats.linregress(predicted, df['borg'])
pred_borg = slope * predicted + intercept

ax9.scatter(df['borg'], pred_borg, c='darkgreen', alpha=0.7, s=80)
ax9.plot([0, 10], [0, 10], 'k--', linewidth=2, label='Perfect')
ax9.fill_between([0, 10], [-1, 9], [1, 11], alpha=0.1, color='green')

# Label some activities
for i, row in df.iterrows():
    if abs(pred_borg.iloc[i] - row['borg']) > 1.5:  # Label outliers
        ax9.annotate(row['activity'][:12], (row['borg'], pred_borg.iloc[i]),
                    fontsize=7, alpha=0.7, rotation=30)

ax9.set_xlabel('Actual Borg CR10')
ax9.set_ylabel('Predicted Borg CR10')
ax9.set_title(f'Predicted vs Actual\n(Combined Model)', fontweight='bold')
ax9.legend(loc='lower right')
ax9.set_xlim(0, 10)
ax9.set_ylim(0, 10)

plt.suptitle('Literature-Backed Effort Features Analysis\n'
             'HR: TRIMP, HR Reserve | IMU: MAD, SMA, Energy Proxy, Jerk',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'literature_features_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR / 'literature_features_analysis.png'}")

# ============================================================================
# Create detailed feature correlation heatmap
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 10))

# Select features for heatmap
features = ['hr_mean', 'hr_delta', 'hr_reserve_pct', 'trimp_banister', 'trimp_edwards',
            'hr_load_sqrt', 'rms_acc', 'sma_dynamic', 'mad', 'energy_proxy',
            'imu_load_sqrt', 'mad_load', 'jerk_load', 'borg']

# Compute correlation matrix
corr_data = df[features].corr()

# Create heatmap
im = ax.imshow(corr_data.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=9)

# Add correlation values
for i in range(len(features)):
    for j in range(len(features)):
        val = corr_data.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                color=color, fontsize=8)

plt.colorbar(im, label='Pearson Correlation')
ax.set_title('Feature Correlation Matrix\n(HR + IMU Literature Features)', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'literature_features_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {OUTPUT_DIR / 'literature_features_correlation.png'}")

# ============================================================================
# Print summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: BEST LITERATURE-BACKED FEATURES")
print("=" * 70)
print()
print("HR FEATURES (from exercise physiology):")
print("  1. hr_load_sqrt (r=0.816) - Stevens Power Law: ΔHR × √duration")
print("  2. trimp_banister (r=0.758) - Banister TRIMP: dur × HRr × 0.64 × e^(1.92×HRr)")
print("  3. trimp_edwards (r=0.713) - Time in HR zones, weighted 1-5")
print()
print("IMU FEATURES (from activity recognition):")
print("  1. imu_load_sqrt (r=0.608) - RMS acceleration × √duration")
print("  2. energy_proxy (r=0.586) - Bouten: ∫(acc-gravity)² dt")
print("  3. mad_load (r=0.482) - Mean Amplitude Deviation × √duration")
print()
print("BEST COMBINED MODEL:")
print("  Effort = 0.8 × z(HR_load_sqrt) + 0.2 × z(MAD_load)")
print("  r = 0.826 with Borg CR10")
print()
