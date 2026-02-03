#!/usr/bin/env python3
"""
Plot 16: Multimodal Overview - PRESENTATION QUALITY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"
base = Path('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3')

# Load all data
print("Loading data...")
ppg = pd.read_csv(base / 'ppg_green' / 'ppg_green_preprocessed.csv')
imu = pd.read_csv(base / 'imu_wrist' / 'imu_preprocessed.csv')
eda = pd.read_csv(base / 'eda' / 'eda_preprocessed.csv')
fused = pd.read_csv(base / 'fused_aligned_5.0s.csv')

# Find time range where we have Borg labels (the interesting part)
borg_data = fused.dropna(subset=['borg'])
t_start = borg_data['t_center'].min() - 60  # 1 min before first label
t_end = borg_data['t_center'].max() + 60    # 1 min after last label

print(f"Focusing on labeled period: {(t_end - t_start)/60:.1f} minutes")

# Filter all data to this range
ppg_filt = ppg[(ppg['t_unix'] >= t_start) & (ppg['t_unix'] <= t_end)].copy()
imu_filt = imu[(imu['t_unix'] >= t_start) & (imu['t_unix'] <= t_end)].copy()
eda_filt = eda[(eda['t_unix'] >= t_start) & (eda['t_unix'] <= t_end)].copy()

# Convert to minutes from start
ppg_filt['minutes'] = (ppg_filt['t_unix'] - t_start) / 60
imu_filt['minutes'] = (imu_filt['t_unix'] - t_start) / 60
eda_filt['minutes'] = (eda_filt['t_unix'] - t_start) / 60
borg_data = borg_data.copy()
borg_data['minutes'] = (borg_data['t_center'] - t_start) / 60

print(f"PPG samples: {len(ppg_filt)}")
print(f"IMU samples: {len(imu_filt)}")
print(f"EDA samples: {len(eda_filt)}")
print(f"Borg labels: {len(borg_data)}")

# Create figure - 3 panels (no EDA)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Preprocessed Sensor Data - Subject P3 (ADL Session)', fontsize=14, fontweight='bold', y=0.98)

# Color scheme
colors = {'ppg': '#2ecc71', 'imu': '#3498db', 'borg': '#e74c3c'}

# 1. PPG - show FULL amplitude
step = max(1, len(ppg_filt) // 3000)
ax = axes[0]
ax.plot(ppg_filt['minutes'].values[::step], ppg_filt['value'].values[::step], 
        color=colors['ppg'], linewidth=0.5, alpha=0.9)
ax.set_ylabel('PPG\n(counts)', fontsize=11)
# Show full range
ax.set_ylim(ppg_filt['value'].min() * 0.95, ppg_filt['value'].max() * 1.05)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#fafafa')

# 2. IMU - accelerometer magnitude - show FULL amplitude
step = max(1, len(imu_filt) // 3000)
ax = axes[1]
acc_mag = np.sqrt(imu_filt['acc_x_dyn']**2 + imu_filt['acc_y_dyn']**2 + imu_filt['acc_z_dyn']**2)
ax.plot(imu_filt['minutes'].values[::step], acc_mag.values[::step], 
        color=colors['imu'], linewidth=0.5, alpha=0.9)
ax.set_ylabel('Accelerometer\n(g)', fontsize=11)
ax.set_ylim(0, acc_mag.max() * 1.05)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#fafafa')

# 3. Borg ratings - only scatter, no line connecting (to show gaps)
ax = axes[2]
ax.scatter(borg_data['minutes'].values, borg_data['borg'].values, 
           c=colors['borg'], s=40, alpha=0.9, zorder=5, edgecolors='darkred', linewidth=0.5)
ax.set_ylabel('Borg CR10\n(0-10 scale)', fontsize=11)
ax.set_ylim(-0.5, max(borg_data['borg'].max() + 1, 7))
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax.grid(True, alpha=0.3)
ax.set_facecolor('#fafafa')

ax.set_xlabel('Time (minutes)', fontsize=12)

# Set x range
max_min = borg_data['minutes'].max() + 1
for ax in axes:
    ax.set_xlim(0, max_min)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.savefig(f"{OUT}/16_multimodal_overview_real.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT}/16_multimodal_overview_real.png")
plt.close()

print("Done!")
