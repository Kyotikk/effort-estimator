#!/usr/bin/env python3
"""Visualize HR load and IMU load across activities for both subjects."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load both datasets
elderly = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/effort_features_full.csv')
healthy = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_healthy3/effort_estimation_output/effort_features_full.csv')

# Sort by start time
elderly = elderly.sort_values('t_start').reset_index(drop=True)
healthy = healthy.sort_values('t_start').reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# === sim_elderly3 ===
ax1 = axes[0]
x = range(len(elderly))
width = 0.35

# Normalize for comparison
hr_norm = elderly['hr_load_sqrt'] / elderly['hr_load_sqrt'].max()
imu_norm = elderly['mad_load'] / elderly['mad_load'].max()
borg_norm = elderly['borg'] / elderly['borg'].max()

bars1 = ax1.bar([i - width/2 for i in x], hr_norm, width, label='HR Load (normalized)', color='crimson', alpha=0.7)
bars2 = ax1.bar([i + width/2 for i in x], imu_norm, width, label='IMU Load (normalized)', color='steelblue', alpha=0.7)
ax1.plot(x, borg_norm, 'ko-', markersize=6, linewidth=2, label='Borg (normalized)')

ax1.set_ylabel('Normalized Value')
ax1.set_title('sim_elderly3: HR Load vs IMU Load vs Borg (sorted by time)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(elderly['activity'], rotation=45, ha='right', fontsize=8)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 1.2)
ax1.grid(axis='y', alpha=0.3)

# === sim_healthy3 ===
ax2 = axes[1]
x = range(len(healthy))

hr_norm = healthy['hr_load_sqrt'] / healthy['hr_load_sqrt'].max()
imu_norm = healthy['mad_load'] / healthy['mad_load'].max()
borg_norm = healthy['borg'] / max(healthy['borg'].max(), 0.1)  # Avoid div by zero

bars1 = ax2.bar([i - width/2 for i in x], hr_norm, width, label='HR Load (normalized)', color='crimson', alpha=0.7)
bars2 = ax2.bar([i + width/2 for i in x], imu_norm, width, label='IMU Load (normalized)', color='steelblue', alpha=0.7)
ax2.plot(x, borg_norm, 'ko-', markersize=6, linewidth=2, label='Borg (normalized)')

ax2.set_ylabel('Normalized Value')
ax2.set_title('sim_healthy3: HR Load vs IMU Load vs Borg (sorted by time)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(healthy['activity'], rotation=45, ha='right', fontsize=8)
ax2.legend(loc='upper left')
ax2.set_ylim(0, 1.2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('slides/hr_imu_load_timeseries.png', dpi=150, bbox_inches='tight')
print("Saved: slides/hr_imu_load_timeseries.png")

# Also create a version with actual values (not normalized)
fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10))

# === sim_elderly3 - actual values ===
ax1 = axes2[0]
x = range(len(elderly))

ax1_twin = ax1.twinx()
bars1 = ax1.bar(x, elderly['hr_load_sqrt'], width=0.6, label='HR Load (sqrt)', color='crimson', alpha=0.7)
line1 = ax1_twin.plot(x, elderly['borg'], 'go-', markersize=8, linewidth=2, label='Borg CR10')

ax1.set_ylabel('HR Load (HR_delta × √duration)', color='crimson')
ax1_twin.set_ylabel('Borg CR10', color='green')
ax1.set_title('sim_elderly3: HR Load vs Borg Rating', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(elderly['activity'], rotation=45, ha='right', fontsize=8)
ax1.tick_params(axis='y', labelcolor='crimson')
ax1_twin.tick_params(axis='y', labelcolor='green')
ax1_twin.set_ylim(0, 7)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# === sim_healthy3 - actual values ===
ax2 = axes2[1]
x = range(len(healthy))

ax2_twin = ax2.twinx()
bars2 = ax2.bar(x, healthy['hr_load_sqrt'], width=0.6, label='HR Load (sqrt)', color='crimson', alpha=0.7)
line2 = ax2_twin.plot(x, healthy['borg'], 'go-', markersize=8, linewidth=2, label='Borg CR10')

ax2.set_ylabel('HR Load (HR_delta × √duration)', color='crimson')
ax2_twin.set_ylabel('Borg CR10', color='green')
ax2.set_title('sim_healthy3: HR Load vs Borg Rating', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(healthy['activity'], rotation=45, ha='right', fontsize=8)
ax2.tick_params(axis='y', labelcolor='crimson')
ax2_twin.tick_params(axis='y', labelcolor='green')
ax2_twin.set_ylim(0, 7)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('slides/hr_load_vs_borg_timeseries.png', dpi=150, bbox_inches='tight')
print("Saved: slides/hr_load_vs_borg_timeseries.png")

plt.show()
