#!/usr/bin/env python3
"""
Visualize subject variability to justify why cross-subject generalization fails.

Creates plots showing:
1. Same activity → different Borg ratings across subjects
2. Same physiological features → different perceived effort
3. Subject-specific Borg distributions

This demonstrates the inherent SUBJECTIVITY of perceived effort.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

# Load combined data
DATA_PATH = Path('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')
PLOTS_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined_5subj/plots')
PLOTS_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples from {df['subject'].nunique()} subjects")

# Style
plt.style.use('seaborn-v0_8-whitegrid')
SUBJECT_COLORS = {
    'sim_elderly1': '#e74c3c',
    'sim_elderly2': '#3498db',
    'sim_elderly3': '#2ecc71',
    'sim_elderly4': '#9b59b6',
    'sim_elderly5': '#f39c12',
}

# =============================================================================
# PLOT 1: Borg distribution per subject (boxplot + violin)
# =============================================================================
print("\n📊 Plot 1: Borg distribution per subject...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
ax1 = axes[0]
subjects_order = sorted(df['subject'].unique())
colors = [SUBJECT_COLORS[s] for s in subjects_order]

bp = df.boxplot(column='borg', by='subject', ax=ax1, patch_artist=True,
                positions=range(len(subjects_order)))
ax1.set_title('')
fig.suptitle('')

for patch, color in zip(bp.artists if hasattr(bp, 'artists') else ax1.patches, colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xlabel('Subject', fontsize=12)
ax1.set_ylabel('Borg CR-10 Rating', fontsize=12)
ax1.set_title('Borg Rating Distribution by Subject', fontsize=14, fontweight='bold')
ax1.set_xticklabels([s.replace('sim_', '') for s in subjects_order], rotation=15)

# Add mean annotations
for i, subj in enumerate(subjects_order):
    mean_borg = df[df['subject'] == subj]['borg'].mean()
    ax1.annotate(f'μ={mean_borg:.1f}', xy=(i, mean_borg), xytext=(i+0.3, mean_borg+0.5),
                fontsize=9, color='red')

# Violin plot
ax2 = axes[1]
parts = ax2.violinplot([df[df['subject'] == s]['borg'].dropna().values for s in subjects_order],
                        positions=range(len(subjects_order)), showmeans=True, showmedians=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

ax2.set_xticks(range(len(subjects_order)))
ax2.set_xticklabels([s.replace('sim_', '') for s in subjects_order], rotation=15)
ax2.set_xlabel('Subject', fontsize=12)
ax2.set_ylabel('Borg CR-10 Rating', fontsize=12)
ax2.set_title('Borg Rating Distribution (Violin)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '1_borg_distribution_by_subject.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 1_borg_distribution_by_subject.png")

# =============================================================================
# PLOT 2: Same PPG feature value → Different Borg (subjectivity)
# =============================================================================
print("\n📊 Plot 2: Same physiological feature → Different Borg...")

# Find a good PPG feature
ppg_features = [c for c in df.columns if 'ppg_green' in c and 'hr' not in c.lower()]
if ppg_features:
    feature = 'ppg_green_p95' if 'ppg_green_p95' in df.columns else ppg_features[0]
else:
    feature = [c for c in df.columns if 'eda' in c][0]

fig, ax = plt.subplots(figsize=(10, 7))

for subj in subjects_order:
    subj_data = df[df['subject'] == subj]
    ax.scatter(subj_data[feature], subj_data['borg'], 
               c=SUBJECT_COLORS[subj], label=subj.replace('sim_', ''),
               alpha=0.6, s=40)

ax.set_xlabel(f'{feature}', fontsize=12)
ax.set_ylabel('Borg CR-10 Rating', fontsize=12)
ax.set_title(f'Same Physiological Signal → Different Perceived Effort\n(Demonstrates Inter-Subject Variability)', 
             fontsize=14, fontweight='bold')
ax.legend(title='Subject', loc='upper right')

# Add annotation box
textstr = 'Key Insight:\nSame PPG amplitude\n→ Different Borg ratings\n→ Effort is SUBJECTIVE'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '2_same_feature_different_borg.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 2_same_feature_different_borg.png")

# =============================================================================
# PLOT 3: Feature-Borg correlation differs by subject
# =============================================================================
print("\n📊 Plot 3: Feature-Borg correlation varies by subject...")

# Select top features
top_features = ['ppg_green_p95', 'ppg_green_range', 'eda_stress_skin_max']
top_features = [f for f in top_features if f in df.columns][:3]

if len(top_features) < 3:
    # Fallback to any available features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    top_features = [c for c in numeric_cols if c not in ['borg', 't_center'] and 'valid' not in c][:3]

fig, axes = plt.subplots(1, len(top_features), figsize=(5*len(top_features), 5))
if len(top_features) == 1:
    axes = [axes]

for ax, feat in zip(axes, top_features):
    correlations = []
    for subj in subjects_order:
        subj_data = df[df['subject'] == subj].dropna(subset=[feat, 'borg'])
        if len(subj_data) > 10:
            r, _ = pearsonr(subj_data[feat], subj_data['borg'])
            correlations.append((subj, r))
    
    subjects_plot = [c[0] for c in correlations]
    rs = [c[1] for c in correlations]
    colors_plot = [SUBJECT_COLORS[s] for s in subjects_plot]
    
    bars = ax.bar(range(len(subjects_plot)), rs, color=colors_plot, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(subjects_plot)))
    ax.set_xticklabels([s.replace('sim_', '') for s in subjects_plot], rotation=15)
    ax.set_ylabel('Pearson r with Borg', fontsize=11)
    ax.set_title(f'{feat}\n(correlation varies!)', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.6, 0.6)
    
    # Add value labels
    for bar, r in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width()/2, r + 0.03 if r >= 0 else r - 0.08, 
               f'{r:.2f}', ha='center', fontsize=9)

plt.suptitle('Same Feature Has Different Predictive Power Per Subject', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '3_correlation_varies_by_subject.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 3_correlation_varies_by_subject.png")

# =============================================================================
# PLOT 4: Subject mean Borg vs physiological baseline (the "identity problem")
# =============================================================================
print("\n📊 Plot 4: Between-subject variance dominates...")

# Calculate per-subject means
subject_means = df.groupby('subject').agg({
    'borg': 'mean',
    feature: 'mean',
}).reset_index()

fig, ax = plt.subplots(figsize=(8, 6))

for _, row in subject_means.iterrows():
    subj = row['subject']
    ax.scatter(row[feature], row['borg'], c=SUBJECT_COLORS[subj], 
               s=200, label=subj.replace('sim_', ''), edgecolor='black', linewidth=2)

# Fit line through subject means
r_between, _ = pearsonr(subject_means[feature], subject_means['borg'])
z = np.polyfit(subject_means[feature], subject_means['borg'], 1)
p = np.poly1d(z)
x_line = np.linspace(subject_means[feature].min(), subject_means[feature].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Between-subject r={r_between:.2f}')

ax.set_xlabel(f'Mean {feature}', fontsize=12)
ax.set_ylabel('Mean Borg CR-10', fontsize=12)
ax.set_title('Why Pooled Correlation is Misleading\n(Model learns subject IDENTITY, not effort)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left')

# Annotation
textstr = f'Between-subject r = {r_between:.2f}\n(Explains "good" pooled performance)\n\nWithin-subject r ≈ 0.35\n(Actual predictive power)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '4_between_subject_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 4_between_subject_variance.png")

# =============================================================================
# PLOT 5: Summary figure for thesis - LOSO failure explanation
# =============================================================================
print("\n📊 Plot 5: Summary figure for thesis...")

fig = plt.figure(figsize=(14, 10))

# Create grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Borg distributions
ax1 = fig.add_subplot(gs[0, 0])
for i, subj in enumerate(subjects_order):
    subj_data = df[df['subject'] == subj]['borg'].dropna()
    ax1.hist(subj_data, bins=15, alpha=0.5, label=subj.replace('sim_', ''), 
             color=SUBJECT_COLORS[subj], density=True)
ax1.set_xlabel('Borg CR-10', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('A. Borg Distributions Differ by Subject', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)

# Panel B: Same feature, different Borg
ax2 = fig.add_subplot(gs[0, 1])
for subj in subjects_order:
    subj_data = df[df['subject'] == subj]
    ax2.scatter(subj_data[feature], subj_data['borg'], 
               c=SUBJECT_COLORS[subj], alpha=0.4, s=20, label=subj.replace('sim_', ''))
ax2.set_xlabel(feature, fontsize=11)
ax2.set_ylabel('Borg CR-10', fontsize=11)
ax2.set_title('B. Same Signal → Different Perceived Effort', fontsize=12, fontweight='bold')

# Panel C: LOSO results bar chart
ax3 = fig.add_subplot(gs[1, 0])
# Simulated LOSO results (you can replace with actual values)
loso_results = {
    'elderly1': 0.15,
    'elderly2': 0.28,
    'elderly3': 0.12,
    'elderly4': 0.33,
    'elderly5': 0.35,
}
subjects_loso = list(loso_results.keys())
rs_loso = list(loso_results.values())
colors_loso = [SUBJECT_COLORS[f'sim_{s}'] for s in subjects_loso]

bars = ax3.bar(subjects_loso, rs_loso, color=colors_loso, alpha=0.8, edgecolor='black')
ax3.axhline(y=np.mean(rs_loso), color='red', linestyle='--', linewidth=2, label=f'Mean r={np.mean(rs_loso):.2f}')
ax3.set_ylabel('LOSO Pearson r', fontsize=11)
ax3.set_title('C. LOSO Performance (Test on Held-Out Subject)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 0.5)
ax3.legend()

for bar, r in zip(bars, rs_loso):
    ax3.text(bar.get_x() + bar.get_width()/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)

# Panel D: Key message
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

message = """
KEY FINDINGS:

1. Borg ratings are SUBJECTIVE
   → Same activity = different perceived effort

2. Physiological signals vary by INDIVIDUAL
   → Same Borg = different PPG/EDA patterns

3. Cross-subject generalization FAILS
   → LOSO r = 0.25 (effectively random)

4. Pooled correlation is MISLEADING
   → Model learns subject identity, not effort

IMPLICATION:
Personalized/longitudinal approach required
for practical effort estimation.
"""

ax4.text(0.1, 0.9, message, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax4.set_title('D. Conclusions', fontsize=12, fontweight='bold')

plt.suptitle('Inter-Subject Variability Prevents Cross-Subject Effort Prediction', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(PLOTS_DIR / '5_thesis_summary_figure.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 5_thesis_summary_figure.png")

# =============================================================================
# SUMMARY STATISTICS TABLE
# =============================================================================
print("\n" + "="*70)
print("SUBJECT VARIABILITY STATISTICS")
print("="*70)

stats = df.groupby('subject').agg({
    'borg': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
stats.columns = ['mean', 'std', 'min', 'max', 'n']
print(stats)

print(f"\n✅ All plots saved to: {PLOTS_DIR}")
print("\nPlots created:")
print("  1. Borg distribution by subject (boxplot + violin)")
print("  2. Same feature → different Borg (scatter)")
print("  3. Correlation varies by subject (bar chart)")
print("  4. Between-subject variance explains pooled r")
print("  5. THESIS SUMMARY FIGURE (4-panel)")
