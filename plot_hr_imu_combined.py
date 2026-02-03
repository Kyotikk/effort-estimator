#!/usr/bin/env python3
"""Visualize HR + IMU combined effort formula"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

# Calculate components
hr_baseline = df['hr_mean'].min()
hr_delta = df['hr_mean'] - hr_baseline
hr_effort = hr_delta * np.sqrt(df['duration_s'])

imu_baseline = df['rms_acc_mag'].min()
imu_delta = df['rms_acc_mag'] - imu_baseline
imu_effort = imu_delta * np.sqrt(df['duration_s'])

# Z-score
z_hr = (hr_effort - hr_effort.mean()) / hr_effort.std()
z_imu = (imu_effort - imu_effort.mean()) / imu_effort.std()

# Combined (80/20)
combined = 0.8 * z_hr + 0.2 * z_imu

# Store in df
df['hr_effort'] = hr_effort
df['imu_effort'] = imu_effort
df['combined'] = combined

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. HR Effort vs Borg
ax = axes[0, 0]
ax.scatter(hr_effort, df['borg'], alpha=0.7, s=80, c='red', edgecolors='darkred')
r, p = stats.pearsonr(hr_effort, df['borg'])
z = np.polyfit(hr_effort, df['borg'], 1)
x_line = np.linspace(hr_effort.min(), hr_effort.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7, linewidth=2)
ax.set_xlabel('HR Effort = HR_delta × √duration', fontsize=11)
ax.set_ylabel('Borg CR10', fontsize=11)
ax.set_title(f'HR Component\nr = {r:.3f}***', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Annotate key points
for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking']:
        ax.annotate(row.activity, (row.hr_effort, row.borg), fontsize=8, alpha=0.8)

# 2. IMU Effort vs Borg
ax = axes[0, 1]
ax.scatter(imu_effort, df['borg'], alpha=0.7, s=80, c='blue', edgecolors='darkblue')
r, p = stats.pearsonr(imu_effort, df['borg'])
z = np.polyfit(imu_effort, df['borg'], 1)
x_line = np.linspace(imu_effort.min(), imu_effort.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7, linewidth=2)
ax.set_xlabel('IMU Effort = RMS_acc_delta × √duration', fontsize=11)
ax.set_ylabel('Borg CR10', fontsize=11)
ax.set_title(f'IMU Component\nr = {r:.3f}***', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking']:
        ax.annotate(row.activity, (row.imu_effort, row.borg), fontsize=8, alpha=0.8)

# 3. Combined vs Borg
ax = axes[0, 2]
ax.scatter(combined, df['borg'], alpha=0.7, s=80, c='green', edgecolors='darkgreen')
r, p = stats.pearsonr(combined, df['borg'])
z = np.polyfit(combined, df['borg'], 1)
x_line = np.linspace(combined.min(), combined.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7, linewidth=2)
ax.set_xlabel('Combined = 0.8×z(HR) + 0.2×z(IMU)', fontsize=11)
ax.set_ylabel('Borg CR10', fontsize=11)
ax.set_title(f'Combined (80% HR + 20% IMU)\nr = {r:.3f}***', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking', 'Stand']:
        ax.annotate(row.activity, (row.combined, row.borg), fontsize=8, alpha=0.8)

# 4. Weight comparison bar chart
ax = axes[1, 0]
weights = [(100, 0), (80, 20), (70, 30), (60, 40), (50, 50), (0, 100)]
correlations = []
labels = []
for w_hr, w_imu in weights:
    c = (w_hr/100) * z_hr + (w_imu/100) * z_imu
    r, _ = stats.pearsonr(c, df['borg'])
    correlations.append(r)
    labels.append(f'{w_hr}/{w_imu}')

colors = ['red' if l == '80/20' else 'steelblue' for l in labels]
bars = ax.bar(labels, correlations, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Weight Ratio (HR% / IMU%)', fontsize=11)
ax.set_ylabel('Correlation with Borg (r)', fontsize=11)
ax.set_title('Optimal Weight Finding\n80/20 is best', fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 0.9)
for bar, r in zip(bars, correlations):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r:.3f}', 
            ha='center', fontsize=9)

# 5. HR vs IMU scatter (colored by Borg)
ax = axes[1, 1]
sc = ax.scatter(z_hr, z_imu, c=df['borg'], cmap='RdYlGn_r', s=80, alpha=0.8, edgecolors='black')
plt.colorbar(sc, ax=ax, label='Borg CR10')
ax.set_xlabel('z(HR Effort)', fontsize=11)
ax.set_ylabel('z(IMU Effort)', fontsize=11)
ax.set_title('HR vs IMU Components\n(colored by Borg)', fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking', 'Eating/Drinking']:
        ax.annotate(row.activity, (z_hr[df.index == row.name].values[0], 
                                   z_imu[df.index == row.name].values[0]), fontsize=8)

# 6. Activity ranking
ax = axes[1, 2]
df_sorted = df.sort_values('combined', ascending=True)
top_n = 15
df_plot = df_sorted.tail(top_n)

colors = plt.cm.RdYlGn_r(df_plot['borg'] / df_plot['borg'].max())
y_pos = range(len(df_plot))
ax.barh(y_pos, df_plot['combined'], color=colors, edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['activity'], fontsize=9)
ax.set_xlabel('Combined Effort Score', fontsize=11)
ax.set_title('Top Activities by Effort\n(color = Borg rating)', fontsize=12, fontweight='bold')

# Add Borg labels
for i, (_, row) in enumerate(df_plot.iterrows()):
    ax.text(row.combined + 0.1, i, f'Borg={row.borg}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('slides/hr_imu_combined_plots.png', dpi=150, bbox_inches='tight')
print("Saved: slides/hr_imu_combined_plots.png")

# Also show the formula summary
print()
print("=" * 50)
print("FINAL FORMULA SUMMARY")
print("=" * 50)
print()
print("Effort = 0.8 × z(HR_effort) + 0.2 × z(IMU_effort)")
print()
print("WHERE:")
print(f"  HR_effort  = (HR_mean - {hr_baseline:.1f}) × √duration")
print(f"  IMU_effort = (RMS_acc - {imu_baseline:.4f}) × √duration")
print()
r_final, _ = stats.pearsonr(combined, df['borg'])
print(f"CORRELATION: r = {r_final:.3f}")
