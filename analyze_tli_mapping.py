#!/usr/bin/env python3
"""Analyze how Training Load Index maps to Borg CR10 ratings."""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load all 3 datasets
elderly = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/effort_features_full.csv')
healthy = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_healthy3/effort_estimation_output/effort_features_full.csv')
severe = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_severe3/effort_estimation_output/effort_features_full.csv')

# Add subject column
elderly['subject'] = 'elderly'
healthy['subject'] = 'healthy'
severe['subject'] = 'severe'

# Combine
all_data = pd.concat([elderly, healthy, severe], ignore_index=True)

print("="*70)
print("TRAINING LOAD INDEX (TLI) vs BORG CR10 MAPPING")
print("="*70)

# Compute TLI for each subject (z-score within subject, then combine)
def compute_tli(df):
    """Compute TLI = 0.8 * z(HR_load) + 0.2 * z(IMU_load)"""
    z_hr = (df['hr_load_sqrt'] - df['hr_load_sqrt'].mean()) / df['hr_load_sqrt'].std()
    z_imu = (df['mad_load'] - df['mad_load'].mean()) / df['mad_load'].std()
    return 0.8 * z_hr + 0.2 * z_imu

elderly['tli'] = compute_tli(elderly)
healthy['tli'] = compute_tli(healthy)
severe['tli'] = compute_tli(severe)

all_data = pd.concat([elderly, healthy, severe], ignore_index=True)

print("\n1. RAW HR_LOAD VALUES BY BORG CATEGORY")
print("-"*70)
print(f"{'Borg':<8} {'n':<6} {'HR_load_sqrt':<20} {'MAD_load':<15}")
print(f"{'Range':<8} {'':<6} {'mean ± std':<20} {'mean ± std':<15}")
print("-"*70)

for borg_low in [0, 1, 2, 3, 4, 5, 6, 7]:
    borg_high = borg_low + 1
    subset = all_data[(all_data['borg'] >= borg_low) & (all_data['borg'] < borg_high)]
    if len(subset) >= 2:
        hr_mean = subset['hr_load_sqrt'].mean()
        hr_std = subset['hr_load_sqrt'].std()
        imu_mean = subset['mad_load'].mean()
        imu_std = subset['mad_load'].std()
        print(f"{borg_low}-{borg_high:<6} {len(subset):<6} {hr_mean:.1f} ± {hr_std:.1f}        {imu_mean:.2f} ± {imu_std:.2f}")

print("\n2. TLI (z-score) BY BORG CATEGORY")
print("-"*70)
print(f"{'Borg':<8} {'n':<6} {'TLI':<20} {'Interpretation':<25}")
print("-"*70)

borg_to_tli = []
for borg_low in [0, 1, 2, 3, 4, 5, 6, 7]:
    borg_high = borg_low + 1
    subset = all_data[(all_data['borg'] >= borg_low) & (all_data['borg'] < borg_high)]
    if len(subset) >= 2:
        tli_mean = subset['tli'].mean()
        tli_std = subset['tli'].std()
        borg_mid = (borg_low + borg_high) / 2
        borg_to_tli.append((borg_mid, tli_mean))
        
        # Interpretation
        if tli_mean < -1:
            interp = "Very Low Effort"
        elif tli_mean < -0.5:
            interp = "Low Effort"
        elif tli_mean < 0.5:
            interp = "Moderate Effort"
        elif tli_mean < 1:
            interp = "High Effort"
        else:
            interp = "Very High Effort"
        
        print(f"{borg_low}-{borg_high:<6} {len(subset):<6} {tli_mean:+.2f} ± {tli_std:.2f}      {interp}")

print("\n3. EXAMPLE ACTIVITIES BY EFFORT LEVEL")
print("-"*70)

# Low effort (Borg 0-2)
low = all_data[all_data['borg'] <= 2].nsmallest(5, 'tli')[['activity', 'subject', 'borg', 'hr_load_sqrt', 'tli']]
print("\nLOW EFFORT (Borg ≤ 2):")
for _, row in low.iterrows():
    print(f"  {row['activity']:<25} ({row['subject']}) Borg={row['borg']:.1f}, HR_load={row['hr_load_sqrt']:.0f}, TLI={row['tli']:+.2f}")

# High effort (Borg ≥ 4)
high = all_data[all_data['borg'] >= 4].nlargest(5, 'tli')[['activity', 'subject', 'borg', 'hr_load_sqrt', 'tli']]
print("\nHIGH EFFORT (Borg ≥ 4):")
for _, row in high.iterrows():
    print(f"  {row['activity']:<25} ({row['subject']}) Borg={row['borg']:.1f}, HR_load={row['hr_load_sqrt']:.0f}, TLI={row['tli']:+.2f}")

print("\n4. LINEAR REGRESSION: TLI → Borg")
print("-"*70)

# Fit regression
slope, intercept, r, p, se = stats.linregress(all_data['tli'], all_data['borg'])
print(f"Borg = {slope:.2f} × TLI + {intercept:.2f}")
print(f"R² = {r**2:.3f}, p < 0.001")
print(f"\nTo convert TLI to Borg estimate:")
print(f"  TLI = -2  →  Borg ≈ {slope*-2 + intercept:.1f}")
print(f"  TLI = -1  →  Borg ≈ {slope*-1 + intercept:.1f}")
print(f"  TLI =  0  →  Borg ≈ {slope*0 + intercept:.1f}")
print(f"  TLI = +1  →  Borg ≈ {slope*1 + intercept:.1f}")
print(f"  TLI = +2  →  Borg ≈ {slope*2 + intercept:.1f}")

print("\n5. PROBLEM: TLI IS RELATIVE (z-score), NOT ABSOLUTE")
print("-"*70)
print("""
The TLI is computed as z-scores WITHIN each subject, which means:
- TLI = 0 means "average effort FOR THIS PERSON"
- TLI = +1 means "1 SD above their average"

This is RELATIVE, not ABSOLUTE. A healthy person's TLI=+1 might be 
an elderly person's TLI=-1 in terms of actual physiological load.

For ABSOLUTE training load, we need raw values:
""")

print("\n6. ABSOLUTE HR LOAD → BORG MAPPING")
print("-"*70)

# Use raw HR_load_sqrt values
slope_abs, intercept_abs, r_abs, _, _ = stats.linregress(all_data['hr_load_sqrt'], all_data['borg'])
print(f"Borg ≈ {slope_abs:.4f} × HR_load + {intercept_abs:.2f}")
print(f"R² = {r_abs**2:.3f}")
print(f"\nTo estimate Borg from raw HR_load_sqrt:")
print(f"  HR_load = 50   →  Borg ≈ {slope_abs*50 + intercept_abs:.1f}")
print(f"  HR_load = 100  →  Borg ≈ {slope_abs*100 + intercept_abs:.1f}")
print(f"  HR_load = 150  →  Borg ≈ {slope_abs*150 + intercept_abs:.1f}")
print(f"  HR_load = 200  →  Borg ≈ {slope_abs*200 + intercept_abs:.1f}")
print(f"  HR_load = 300  →  Borg ≈ {slope_abs*300 + intercept_abs:.1f}")

# === VISUALIZATION ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: TLI vs Borg scatter
ax1 = axes[0, 0]
colors = {'elderly': 'red', 'healthy': 'blue', 'severe': 'green'}
for subj in ['elderly', 'healthy', 'severe']:
    subset = all_data[all_data['subject'] == subj]
    ax1.scatter(subset['tli'], subset['borg'], c=colors[subj], alpha=0.6, s=50, label=subj)
x_line = np.linspace(-3, 3, 100)
ax1.plot(x_line, slope * x_line + intercept, 'k--', label=f'Fit: Borg = {slope:.2f}×TLI + {intercept:.2f}')
ax1.set_xlabel('TLI (z-score)')
ax1.set_ylabel('Borg CR10')
ax1.set_title('TLI vs Borg (Relative)')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Raw HR_load vs Borg
ax2 = axes[0, 1]
for subj in ['elderly', 'healthy', 'severe']:
    subset = all_data[all_data['subject'] == subj]
    ax2.scatter(subset['hr_load_sqrt'], subset['borg'], c=colors[subj], alpha=0.6, s=50, label=subj)
x_line = np.linspace(0, 350, 100)
ax2.plot(x_line, slope_abs * x_line + intercept_abs, 'k--', label=f'Fit')
ax2.set_xlabel('HR Load (HR_delta × √duration)')
ax2.set_ylabel('Borg CR10')
ax2.set_title('HR Load vs Borg (Absolute)')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Borg distribution by TLI bins
ax3 = axes[1, 0]
tli_bins = [-3, -1.5, -0.5, 0.5, 1.5, 3]
tli_labels = ['Very Low\n(<-1.5)', 'Low\n(-1.5 to -0.5)', 'Moderate\n(-0.5 to 0.5)', 'High\n(0.5 to 1.5)', 'Very High\n(>1.5)']
all_data['tli_bin'] = pd.cut(all_data['tli'], bins=tli_bins, labels=tli_labels)
all_data.boxplot(column='borg', by='tli_bin', ax=ax3)
ax3.set_xlabel('TLI Category')
ax3.set_ylabel('Borg CR10')
ax3.set_title('Borg Distribution by TLI Category')
plt.suptitle('')

# Plot 4: HR_load distribution by Borg category
ax4 = axes[1, 1]
borg_bins = [0, 2, 4, 6, 10]
borg_labels = ['Light\n(0-2)', 'Moderate\n(2-4)', 'Hard\n(4-6)', 'Very Hard\n(6+)']
all_data['borg_cat'] = pd.cut(all_data['borg'], bins=borg_bins, labels=borg_labels)
all_data.boxplot(column='hr_load_sqrt', by='borg_cat', ax=ax4)
ax4.set_xlabel('Borg Category')
ax4.set_ylabel('HR Load (HR_delta × √duration)')
ax4.set_title('HR Load by Borg Category')
plt.suptitle('')

plt.tight_layout()
plt.savefig('slides/tli_borg_mapping.png', dpi=150, bbox_inches='tight')
print("\nSaved: slides/tli_borg_mapping.png")

plt.show()
