#!/usr/bin/env python3
"""
Feature Extraction Pipeline Visualization for Thesis
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# =============================================================================
# FIGURE: FEATURE EXTRACTION PIPELINE FLOW
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
C_SIGNAL = '#3498db'
C_PREPROCESS = '#9b59b6'  
C_WINDOW = '#2ecc71'
C_FEATURE = '#e67e22'
C_IMU = '#e74c3c'
C_PPG = '#1abc9c'
C_EDA = '#f39c12'

def draw_box(ax, x, y, w, h, text, color, fontsize=9, alpha=0.85):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Title
ax.text(8, 11.5, 'Feature Extraction Pipeline', ha='center', fontsize=16, fontweight='bold')

# =============================================================================
# ROW 1: RAW SIGNALS
# =============================================================================
y1 = 10
ax.text(0.5, y1+0.5, 'RAW SIGNALS', fontsize=10, fontweight='bold', color='gray')

draw_box(ax, 1, y1-0.4, 2.5, 0.8, 'IMU\n3-axis accel @ 32Hz', C_IMU, 9)
draw_box(ax, 4.5, y1-0.4, 2.5, 0.8, 'PPG Green\n@ 32Hz', C_PPG, 9)
draw_box(ax, 8, y1-0.4, 2.5, 0.8, 'PPG Infra/Red\n@ 32Hz', C_PPG, 9)
draw_box(ax, 11.5, y1-0.4, 2.5, 0.8, 'EDA\n@ 32Hz', C_EDA, 9)

# =============================================================================
# ROW 2: PREPROCESSING
# =============================================================================
y2 = 8.2
ax.text(0.5, y2+0.5, 'PREPROCESSING', fontsize=10, fontweight='bold', color='gray')

# IMU preprocessing
draw_box(ax, 0.8, y2-0.6, 3, 1.0, 'Gravity Removal\n(HPF 0.3Hz)\nNoise Filter (LPF 5Hz)', C_IMU, 8)

# PPG Green - NO HPF
draw_box(ax, 4.3, y2-0.6, 3, 1.0, 'Resampling only\n(No HPF applied)\nDC offset preserved', C_PPG, 8)

# PPG Infra/Red - WITH HPF
draw_box(ax, 7.8, y2-0.6, 3, 1.0, 'HPF 0.5Hz\n(Baseline removal)\nCentered at 0', C_PPG, 8)

# EDA
draw_box(ax, 11.3, y2-0.6, 3, 1.0, 'Tonic/Phasic\nDecomposition\n(cvxEDA)', C_EDA, 8)

# Arrows
for x in [2.25, 5.75, 9.25, 12.75]:
    draw_arrow(ax, x, y1-0.4, x, y2+0.4)

# =============================================================================
# ROW 3: WINDOWING
# =============================================================================
y3 = 6.2
ax.text(0.5, y3+0.5, 'WINDOWING', fontsize=10, fontweight='bold', color='gray')

draw_box(ax, 3, y3-0.3, 10, 0.8, 'Sliding Window: 5 seconds, 10% overlap → 160 samples/window @ 32Hz', C_WINDOW, 10)

# Arrows converging
for x in [2.3, 5.8, 9.3, 12.8]:
    draw_arrow(ax, x, y2-0.6, 8, y3+0.5)

# =============================================================================
# ROW 4: FEATURE EXTRACTION (DETAILED)
# =============================================================================
y4 = 4.0
ax.text(0.5, y4+1.0, 'FEATURE\nEXTRACTION', fontsize=10, fontweight='bold', color='gray')

# IMU Features box
imu_box_x = 0.8
draw_box(ax, imu_box_x, y4-1.5, 3.5, 2.5, '', C_IMU, alpha=0.3)
ax.text(imu_box_x+1.75, y4+0.8, 'IMU Features (30)', fontsize=9, fontweight='bold', ha='center')

imu_features = [
    'Quantiles (0.3, 0.4, 0.6, 0.9)',
    'Sample Entropy',
    'Approximate Entropy', 
    'Katz Fractal Dim.',
    'Tsallis Entropy',
    'Variance of Δ|x|',
    'Sum of |changes|',
]
for i, feat in enumerate(imu_features):
    ax.text(imu_box_x+0.2, y4+0.4-i*0.25, f'• {feat}', fontsize=7, va='top')

# PPG Features box
ppg_box_x = 4.8
draw_box(ax, ppg_box_x, y4-1.5, 3.5, 2.5, '', C_PPG, alpha=0.3)
ax.text(ppg_box_x+1.75, y4+0.8, 'PPG Features (183)', fontsize=9, fontweight='bold', ha='center')

ppg_features = [
    'Mean, Std, Min, Max',
    'Percentiles (p1-p99)',
    'Skewness, Kurtosis',
    'HR mean/std/min/max',
    'SDNN, RMSSD, pNN50',
    'Zero-crossing rate',
    'Crest/Shape factors',
]
for i, feat in enumerate(ppg_features):
    ax.text(ppg_box_x+0.2, y4+0.4-i*0.25, f'• {feat}', fontsize=7, va='top')

# EDA Features box
eda_box_x = 8.8
draw_box(ax, eda_box_x, y4-1.5, 3.5, 2.5, '', C_EDA, alpha=0.3)
ax.text(eda_box_x+1.75, y4+0.8, 'EDA Features (47)', fontsize=9, fontweight='bold', ha='center')

eda_features = [
    'SCL mean/std/slope',
    'SCR count',
    'SCR amplitude stats',
    'Phasic AUC/energy',
    'Tonic AUC',
    'Stress skin stats',
    'IQR, MAD, Kurtosis',
]
for i, feat in enumerate(eda_features):
    ax.text(eda_box_x+0.2, y4+0.4-i*0.25, f'• {feat}', fontsize=7, va='top')

# Total box
draw_box(ax, 12.8, y4-0.5, 2.5, 1.3, 'TOTAL\n260\nfeatures', '#34495e', 10)

# Arrows from windowing
draw_arrow(ax, 8, y3-0.3, 2.5, y4+1)
draw_arrow(ax, 8, y3-0.3, 6.5, y4+1)
draw_arrow(ax, 8, y3-0.3, 10.5, y4+1)

# Arrows to total
draw_arrow(ax, 4.3, y4, 12.8, y4)
draw_arrow(ax, 8.3, y4, 12.8, y4)
draw_arrow(ax, 12.3, y4, 12.8, y4)

# =============================================================================
# ROW 5: LOSO RESULTS
# =============================================================================
y5 = 0.8
ax.text(0.5, y5+0.8, 'LOSO\nRESULTS', fontsize=10, fontweight='bold', color='gray')

# Result boxes
draw_box(ax, 1.5, y5-0.3, 3, 0.9, 'IMU: r = 0.52 ✓', C_IMU, 11)
draw_box(ax, 5.5, y5-0.3, 3, 0.9, 'PPG: r = 0.28 ✗', C_PPG, 11)
draw_box(ax, 9.5, y5-0.3, 3, 0.9, 'EDA: r = 0.02 ✗', C_EDA, 11)

# Final arrow
draw_box(ax, 13, y5-0.3, 2.5, 0.9, 'Best:\nIMU only', '#27ae60', 10)
draw_arrow(ax, 4.5, y5+0.15, 13, y5+0.15)

# =============================================================================
# NOTES
# =============================================================================
notes = [
    "Note: PPG green has no HPF (tested both - no improvement with HPF)",
    "HPF = High-pass filter, LPF = Low-pass filter",
]
for i, note in enumerate(notes):
    ax.text(0.5, 0.1-i*0.25, note, fontsize=7, style='italic', color='gray')

plt.tight_layout()
plt.savefig(f"{OUT}/42_feature_extraction_pipeline.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/42_feature_extraction_pipeline.png")
plt.close()

# =============================================================================
# FIGURE 2: IMU FEATURE DETAILS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sample entropy visualization
ax = axes[0]
ax.set_title('Sample Entropy (SampEn)', fontsize=12, fontweight='bold')
ax.text(0.5, 0.85, 'Measures signal regularity', ha='center', transform=ax.transAxes, fontsize=10)
ax.text(0.5, 0.70, 'SampEn = -ln(A/B)', ha='center', transform=ax.transAxes, fontsize=11, 
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.50, 'A = matches at m+1 dimensions\nB = matches at m dimensions\nm = 2, r = 0.2×std', 
        ha='center', transform=ax.transAxes, fontsize=9)
ax.text(0.5, 0.25, 'Low SampEn → Regular (rest)\nHigh SampEn → Irregular (activity)', 
        ha='center', transform=ax.transAxes, fontsize=10, color='#e74c3c')
ax.axis('off')

# Katz fractal dimension
ax = axes[1]
ax.set_title('Katz Fractal Dimension', fontsize=12, fontweight='bold')
ax.text(0.5, 0.85, 'Measures signal complexity', ha='center', transform=ax.transAxes, fontsize=10)
ax.text(0.5, 0.70, 'KFD = log₁₀(n) / (log₁₀(d/L) + log₁₀(n))', ha='center', transform=ax.transAxes, 
        fontsize=10, family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.50, 'n = number of samples\nL = total path length\nd = max deviation from start', 
        ha='center', transform=ax.transAxes, fontsize=9)
ax.text(0.5, 0.25, 'Higher KFD → More complex trajectory\n(e.g., vigorous movement)', 
        ha='center', transform=ax.transAxes, fontsize=10, color='#e74c3c')
ax.axis('off')

# Variance of absolute differences
ax = axes[2]
ax.set_title('Variance of |Δx|', fontsize=12, fontweight='bold')
ax.text(0.5, 0.85, 'Measures movement jerkiness', ha='center', transform=ax.transAxes, fontsize=10)
ax.text(0.5, 0.70, 'Var(|x[i+1] - x[i]|)', ha='center', transform=ax.transAxes, 
        fontsize=11, family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(0.5, 0.50, 'Computes variance of\nabsolute sample-to-sample changes', 
        ha='center', transform=ax.transAxes, fontsize=9)
ax.text(0.5, 0.25, 'High variance → Jerky, effortful motion\nLow variance → Smooth movement', 
        ha='center', transform=ax.transAxes, fontsize=10, color='#e74c3c')
ax.axis('off')

plt.suptitle('Key IMU Features Explained', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/43_imu_features_explained.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/43_imu_features_explained.png")
plt.close()

print("\nDone!")
