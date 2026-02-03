#!/usr/bin/env python3
"""Visualize TLI results."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. TLI vs Borg
ax1 = axes[0, 0]
ax1.scatter(df['TLI'], df['borg'], alpha=0.7, s=100, c='steelblue')
z = np.polyfit(df['TLI'], df['borg'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['TLI'].min(), df['TLI'].max(), 100)
ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='r = 0.721')
ax1.set_xlabel('TLI (z-scored)', fontsize=12)
ax1.set_ylabel('Borg CR10', fontsize=12)
ax1.set_title('TLI vs Perceived Exertion', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. HR load vs Borg
ax2 = axes[0, 1]
ax2.scatter(df['hr_load'], df['borg'], alpha=0.7, s=100, c='crimson')
z = np.polyfit(df['hr_load'], df['borg'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['hr_load'].min(), df['hr_load'].max(), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label='r = 0.771')
ax2.set_xlabel('HR Load (delta HR x duration)', fontsize=12)
ax2.set_ylabel('Borg CR10', fontsize=12)
ax2.set_title('HR Load vs Perceived Exertion', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. IMU load vs Borg
ax3 = axes[1, 0]
ax3.scatter(df['imu_load'], df['borg'], alpha=0.7, s=100, c='forestgreen')
z = np.polyfit(df['imu_load'], df['borg'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['imu_load'].min(), df['imu_load'].max(), 100)
ax3.plot(x_line, p(x_line), 'r--', linewidth=2, label='r = 0.625')
ax3.set_xlabel('IMU Load (active time x intensity)', fontsize=12)
ax3.set_ylabel('Borg CR10', fontsize=12)
ax3.set_title('IMU Load vs Perceived Exertion', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Activity breakdown
ax4 = axes[1, 1]
df_sorted = df.sort_values('TLI')
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df_sorted)))
bars = ax4.barh(range(len(df_sorted)), df_sorted['TLI'], color=colors)
ax4.set_yticks(range(len(df_sorted)))
ax4.set_yticklabels(df_sorted['activity'], fontsize=8)
ax4.set_xlabel('TLI', fontsize=12)
ax4.set_title('TLI by Activity', fontsize=14, fontweight='bold')
ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax4.grid(True, alpha=0.3, axis='x')

plt.suptitle('Training Load Index (TLI) - sim_elderly3', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/slides/tli_results.png', dpi=150, bbox_inches='tight')
print('Saved to slides/tli_results.png')
plt.close()
