#!/usr/bin/env python3
"""Visualize HR load and IMU load across activities for all 3 subjects."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load all 3 datasets
elderly = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/effort_features_full.csv')
healthy = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_healthy3/effort_estimation_output/effort_features_full.csv')
severe = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_severe3/effort_estimation_output/effort_features_full.csv')

# Sort by start time
elderly = elderly.sort_values('t_start').reset_index(drop=True)
healthy = healthy.sort_values('t_start').reset_index(drop=True)
severe = severe.sort_values('t_start').reset_index(drop=True)

datasets = [
    (elderly, 'sim_elderly3', 'Borg: 0.5-6.0'),
    (healthy, 'sim_healthy3', 'Borg: 0.0-1.5'),
    (severe, 'sim_severe3', 'Borg: 0.0-7.0')
]

# === FIGURE 1: All 3 subjects - HR Load, IMU Load, Borg ===
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

for idx, (df, name, borg_range) in enumerate(datasets):
    ax = axes[idx]
    x = range(len(df))
    width = 0.35
    
    # Normalize for comparison
    hr_norm = df['hr_load_sqrt'] / df['hr_load_sqrt'].max() if df['hr_load_sqrt'].max() > 0 else df['hr_load_sqrt']
    imu_norm = df['mad_load'] / df['mad_load'].max() if df['mad_load'].max() > 0 else df['mad_load']
    borg_max = max(df['borg'].max(), 0.1)
    borg_norm = df['borg'] / borg_max
    
    bars1 = ax.bar([i - width/2 for i in x], hr_norm, width, label='HR Load', color='crimson', alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], imu_norm, width, label='IMU Load (MAD)', color='steelblue', alpha=0.7)
    ax.plot(x, borg_norm, 'go-', markersize=6, linewidth=2, label='Borg')
    
    ax.set_ylabel('Normalized (0-1)')
    ax.set_title(f'{name}: HR Load vs IMU Load vs Borg ({borg_range})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['activity'], rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.3)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('slides/all_subjects_hr_imu_borg.png', dpi=150, bbox_inches='tight')
print("Saved: slides/all_subjects_hr_imu_borg.png")

# === FIGURE 2: Summary comparison ===
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

from scipy import stats

for idx, (df, name, _) in enumerate(datasets):
    ax = axes2[idx]
    
    # Scatter: HR load vs Borg
    ax.scatter(df['hr_load_sqrt'], df['borg'], c='crimson', alpha=0.6, s=60, label='HR Load')
    ax.scatter(df['mad_load'] * df['hr_load_sqrt'].max() / max(df['mad_load'].max(), 0.1), 
               df['borg'], c='steelblue', alpha=0.6, s=60, marker='s', label='IMU Load (scaled)')
    
    # Correlation
    r_hr = stats.pearsonr(df['hr_load_sqrt'], df['borg'])[0]
    r_imu = stats.pearsonr(df['mad_load'], df['borg'])[0]
    
    ax.set_xlabel('Load Value')
    ax.set_ylabel('Borg CR10')
    ax.set_title(f'{name}\nHR r={r_hr:.2f}, IMU r={r_imu:.2f}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('slides/all_subjects_correlation_scatter.png', dpi=150, bbox_inches='tight')
print("Saved: slides/all_subjects_correlation_scatter.png")

# === Print summary table ===
print("\n" + "="*70)
print("SUMMARY: All 3 Subjects")
print("="*70)
print(f"\n{'Subject':<15} {'Activities':<12} {'Borg Range':<12} {'r(HR)':<10} {'r(IMU)':<10} {'r(Combined)':<12}")
print("-"*70)

for df, name, _ in datasets:
    r_hr = stats.pearsonr(df['hr_load_sqrt'], df['borg'])[0]
    r_imu = stats.pearsonr(df['mad_load'], df['borg'])[0]
    
    # Combined
    z_hr = (df['hr_load_sqrt'] - df['hr_load_sqrt'].mean()) / df['hr_load_sqrt'].std()
    z_imu = (df['mad_load'] - df['mad_load'].mean()) / df['mad_load'].std()
    combined = 0.8 * z_hr + 0.2 * z_imu
    r_comb = stats.pearsonr(combined, df['borg'])[0]
    
    borg_range = f"{df['borg'].min():.1f}-{df['borg'].max():.1f}"
    print(f"{name:<15} {len(df):<12} {borg_range:<12} {r_hr:<10.3f} {r_imu:<10.3f} {r_comb:<12.3f}")

plt.show()
