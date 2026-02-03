#!/usr/bin/env python3
"""
Scientifically-grounded Effort Index based on established methods:

1. **Banister's TRIMP** (Training Impulse, 1991):
   TRIMP = duration × HR_fraction × e^(b × HR_fraction)
   Where HR_fraction = (HR - HR_rest) / (HR_max - HR_rest)
   The exponential term captures that high HR zones are disproportionately harder

2. **Session RPE** (Foster et al., 2001):
   Session_Load = RPE × duration (minutes)
   
3. **Edwards' TRIMP** (1993):
   Weighted time in HR zones (Zone 1-5 weighted 1-5)

Key scientific insight:
- The relationship between HR and metabolic cost is EXPONENTIAL at higher intensities
- This is why sqrt(duration) wasn't quite right
- Banister's method: uses e^(b × HR_fraction) where b ≈ 1.92 (men) or 1.67 (women)

For ADL effort estimation, we adapt:
- Use HR elevation fraction (not absolute HR)
- Apply exponential weighting to intensity
- Duration has linear effect on TOTAL load, but we want PERCEIVED effort
- Borg reflects perceived difficulty which has complex duration relationship
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/tli_results.csv')

# =============================================================================
# METHOD 1: Banister TRIMP (adapted for ADL)
# =============================================================================
# Estimate HR parameters (for elderly participant)
hr_rest = df['hr_mean'].min()  # ~75.5 BPM (resting baseline)
hr_max = 220 - 70  # Estimated max HR for ~70 year old = 150 BPM

# HR reserve fraction: how much of their HR capacity are they using?
df['hr_fraction'] = (df['hr_mean'] - hr_rest) / (hr_max - hr_rest)
df['hr_fraction'] = df['hr_fraction'].clip(0, 1)  # Bound to [0, 1]

# Banister weighting factor (exponential): higher HR zones are harder
# Using b = 1.92 (male default from literature)
b = 1.92
df['banister_weight'] = df['hr_fraction'] * np.exp(b * df['hr_fraction'])

# TRIMP = duration × weighting
df['TRIMP'] = df['duration_s'] * df['banister_weight']

# =============================================================================
# METHOD 2: Intensity-only (no duration) - for moment-to-moment perception
# =============================================================================
# Pure physiological intensity (exponentially weighted HR)
df['intensity_exp'] = df['banister_weight']

# =============================================================================
# METHOD 3: Power Law Duration (Stevens' Law)
# =============================================================================
# Stevens' Power Law: Perceived magnitude = k × Stimulus^n
# For duration perception, n is typically 0.8-1.0 for pain/effort
# Using n = 0.5 (sqrt) is too aggressive; try 0.7
power = 0.7
df['effort_stevens'] = df['intensity_exp'] * (df['duration_s'] ** power)

# =============================================================================
# METHOD 4: Logarithmic Duration (Weber-Fechner)
# =============================================================================
# Weber-Fechner: Perceived = k × log(Stimulus)
# Longer durations feel progressively less different
df['effort_weber'] = df['intensity_exp'] * (1 + np.log(df['duration_s'] / 10))  # normalize to ~10s baseline

# =============================================================================
# METHOD 5: Threshold-based (IMU adds if moving)
# =============================================================================
# Only count duration if actually moving (IMU intensity above threshold)
imu_threshold = df['rms_acc_mag'].quantile(0.25)  # 25th percentile as "not moving"
df['is_active'] = (df['rms_acc_mag'] > imu_threshold).astype(float)
df['effort_active'] = df['intensity_exp'] * df['duration_s'] * (0.3 + 0.7 * df['is_active'])

# =============================================================================
# COMPARE ALL METHODS
# =============================================================================
print("="*70)
print("SCIENTIFIC COMPARISON: Effort Estimation Methods")
print("="*70)
print()

methods = {
    'HR delta (linear)': df['hr_delta'],
    'HR fraction (0-1)': df['hr_fraction'],
    'Banister weight (exp)': df['banister_weight'],
    'Intensity × duration': df['hr_delta'] * df['duration_s'],
    'Intensity × sqrt(dur)': df['hr_delta'] * np.sqrt(df['duration_s']),
    'TRIMP (Banister)': df['TRIMP'],
    'Intensity only (exp)': df['intensity_exp'],
    'Stevens (dur^0.7)': df['effort_stevens'],
    'Weber-Fechner (log)': df['effort_weber'],
    'Active-weighted': df['effort_active'],
}

print(f"{'Method':<30} {'r':<8} {'p-value':<12} {'Interpretation'}")
print("-"*70)

results = []
for name, metric in methods.items():
    r, p = stats.pearsonr(metric.fillna(0), df['borg'])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    interp = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.5 else "weak"
    print(f"{name:<30} {r:+.3f}    {p:.6f}     {interp} ({sig})")
    results.append({'method': name, 'r': r, 'p': p})

# =============================================================================
# BEST METHOD ANALYSIS
# =============================================================================
print()
print("="*70)
print("ANALYSIS: Which method is scientifically best?")
print("="*70)
print()
print("1. TRIMP (Banister) r=+{:.3f}".format(stats.pearsonr(df['TRIMP'].fillna(0), df['borg'])[0]))
print("   - Gold standard in sports science for training load")
print("   - Uses exponential HR weighting (physiologically valid)")
print("   - Duration is LINEAR (total accumulated stress)")
print("   - Problem: Makes resting accumulate load over time")
print()
print("2. Intensity only (exp) r=+{:.3f}".format(stats.pearsonr(df['intensity_exp'].fillna(0), df['borg'])[0]))
print("   - Pure intensity, no duration")
print("   - Best for 'how hard does this feel RIGHT NOW'")
print("   - Doesn't capture that 30min standing > 1min standing")
print()
print("3. Stevens Power Law r=+{:.3f}".format(stats.pearsonr(df['effort_stevens'].fillna(0), df['borg'])[0]))
print("   - Duration^0.7 (sublinear)")
print("   - Psychophysical basis (perception research)")
print("   - Balances intensity and duration")
print()

# Best overall
best_r = max(results, key=lambda x: abs(x['r']))
print(f"BEST CORRELATION: {best_r['method']} (r={best_r['r']:.3f})")

# =============================================================================
# SANITY CHECK WITH BEST METHOD
# =============================================================================
print()
print("="*70)
print("SANITY CHECK: Stevens Power Law (dur^0.7)")
print("="*70)
print()

df['effort_best'] = df['effort_stevens']
# Z-score for interpretability
df['effort_z'] = (df['effort_best'] - df['effort_best'].mean()) / df['effort_best'].std()

df_sorted = df.sort_values('effort_z', ascending=False)

print(f"{'Activity':<25} {'Dur':>6} {'HR%':>6} {'Effort':>8} {'Borg':>5}")
print("-"*55)
for _, row in df_sorted.head(8).iterrows():
    print(f"{row.activity:<25} {row.duration_s:>6.0f} {row.hr_fraction*100:>5.1f}% {row.effort_z:>+8.2f} {row.borg:>5.1f}")
print("...")
for _, row in df_sorted.tail(5).iterrows():
    print(f"{row.activity:<25} {row.duration_s:>6.0f} {row.hr_fraction*100:>5.1f}% {row.effort_z:>+8.2f} {row.borg:>5.1f}")

# =============================================================================
# CONCLUSION
# =============================================================================
print()
print("="*70)
print("SCIENTIFIC RECOMMENDATION")
print("="*70)
print("""
For ADL effort estimation (predicting Borg CR10):

1. **If predicting perceived INTENSITY** (how hard right now):
   → Use exponential HR weighting: HR_frac × e^(1.92 × HR_frac)
   → This is physiologically grounded (Banister TRIMP coefficient)
   
2. **If predicting perceived TOTAL EFFORT** (overall task difficulty):
   → Use Stevens' Power Law: Intensity × Duration^0.7
   → This captures that longer tasks feel harder, but sublinearly
   
3. **Key insight**: The exponential HR weighting is critical!
   → Linear HR delta treats 5 BPM increase = 5 BPM increase
   → Exponential weighting: going from 80% to 90% HR_max is MUCH harder
     than going from 40% to 50%

The Borg scale itself was designed with ratio properties, so Stevens' 
Power Law is theoretically appropriate.
""")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Compare methods
ax = axes[0, 0]
method_names = ['HR delta', 'TRIMP', 'Intensity (exp)', 'Stevens (d^0.7)']
correlations = [
    stats.pearsonr(df['hr_delta'], df['borg'])[0],
    stats.pearsonr(df['TRIMP'], df['borg'])[0],
    stats.pearsonr(df['intensity_exp'], df['borg'])[0],
    stats.pearsonr(df['effort_stevens'], df['borg'])[0],
]
colors = ['gray', 'blue', 'orange', 'green']
bars = ax.bar(method_names, correlations, color=colors, alpha=0.7)
ax.axhline(0.5, color='red', linestyle='--', label='r=0.5 threshold')
ax.set_ylabel('Correlation with Borg CR10 (r)')
ax.set_title('Method Comparison')
ax.set_ylim(0, 1)
for bar, r in zip(bars, correlations):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{r:.2f}', 
            ha='center', fontsize=10)

# 2. Exponential HR weighting visualization
ax = axes[0, 1]
hr_frac = np.linspace(0, 1, 100)
linear_weight = hr_frac
exp_weight = hr_frac * np.exp(1.92 * hr_frac)
ax.plot(hr_frac * 100, linear_weight, 'b-', label='Linear', linewidth=2)
ax.plot(hr_frac * 100, exp_weight, 'r-', label='Banister (exp)', linewidth=2)
ax.set_xlabel('HR Reserve Used (%)')
ax.set_ylabel('Weighting Factor')
ax.set_title('Why Exponential Weighting Matters')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Best method scatter
ax = axes[1, 0]
ax.scatter(df['effort_z'], df['borg'], alpha=0.7, s=80)
r, p = stats.pearsonr(df['effort_z'], df['borg'])
z = np.polyfit(df['effort_z'], df['borg'], 1)
x_line = np.linspace(df['effort_z'].min(), df['effort_z'].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.7)
ax.set_xlabel('Stevens Effort Index (z-scored)')
ax.set_ylabel('Borg CR10')
ax.set_title(f'Stevens Power Law vs Borg\nr = {r:.3f}***')
ax.grid(True, alpha=0.3)

# Annotate outliers
for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking', 'Stand', 'Eating/Drinking']:
        ax.annotate(row.activity, (row.effort_z, row.borg), fontsize=8, alpha=0.7)

# 4. Duration effect
ax = axes[1, 1]
scatter = ax.scatter(df['duration_s'], df['effort_z'], c=df['borg'], cmap='RdYlGn_r', 
                     s=80, alpha=0.8)
plt.colorbar(scatter, ax=ax, label='Borg CR10')
ax.set_xlabel('Duration (seconds)')
ax.set_ylabel('Effort Index (Stevens)')
ax.set_title('Duration vs Effort\n(colored by Borg)')
ax.grid(True, alpha=0.3)

# Annotate
for _, row in df.iterrows():
    if row.activity in ['Resting', 'Level Walking']:
        ax.annotate(row.activity, (row.duration_s, row.effort_z), fontsize=8)

plt.tight_layout()
plt.savefig('slides/scientific_effort_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: slides/scientific_effort_comparison.png")

# Save results
df.to_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/scientific_effort_results.csv', index=False)
print(f"Saved: scientific_effort_results.csv")
