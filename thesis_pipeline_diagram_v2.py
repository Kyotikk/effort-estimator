#!/usr/bin/env python3
"""
Pipeline Diagram - matching user's style with detailed feature extraction
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_facecolor('white')

# Color scheme matching user's diagram
dark_teal = '#4A7C8A'
light_teal = '#7BA3AD'
white = 'white'

def draw_box(ax, x, y, w, h, color, text, fontsize=10, textcolor='white'):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='none',
        linewidth=0
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, 
            ha='center', va='center',
            fontsize=fontsize, 
            color=textcolor, wrap=True)
    return box

def draw_arrow(ax, start, end):
    """Draw arrow"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=dark_teal, lw=1.5))

# ============================================
# ROW 1: Raw Signals
# ============================================
y_row1 = 10.2
ax.text(0.3, y_row1 + 0.15, "Raw Signals", fontsize=10, fontweight='bold', color='#333', va='center')

draw_box(ax, 1.8, y_row1 - 0.25, 3.0, 0.7, dark_teal, "IMU\n(3-axis accel.)", fontsize=9)
draw_box(ax, 5.5, y_row1 - 0.25, 3.0, 0.7, dark_teal, "PPG\n(3 wavelengths)", fontsize=9)
draw_box(ax, 9.2, y_row1 - 0.25, 3.0, 0.7, dark_teal, "EDA\n(skin cond.)", fontsize=9)

# ============================================
# ROW 2: Preprocessing
# ============================================
y_row2 = 9.0
ax.text(0.3, y_row2 + 0.15, "Preprocessing", fontsize=10, fontweight='bold', color='#333', va='center')

# Arrows down
for x in [3.3, 7.0, 10.7]:
    draw_arrow(ax, (x, y_row1 - 0.25), (x, y_row2 + 0.45))

draw_box(ax, 1.8, y_row2 - 0.25, 3.0, 0.7, light_teal, "Resampling\nLow-pass filtering", fontsize=9)
draw_box(ax, 5.5, y_row2 - 0.25, 3.0, 0.7, light_teal, "Resampling\nHigh-pass filtering", fontsize=9)
draw_box(ax, 9.2, y_row2 - 0.25, 3.0, 0.7, light_teal, "Resampling\nTonic/Phasic split", fontsize=9)

# ============================================
# ROW 3: Temporal Segmentation
# ============================================
y_row3 = 7.8
ax.text(0.3, y_row3 + 0.1, "Temporal\nSegmentation", fontsize=10, fontweight='bold', color='#333', va='center')

# Arrows converging
for x in [3.3, 7.0, 10.7]:
    draw_arrow(ax, (x, y_row2 - 0.25), (x, y_row3 + 0.4))

draw_box(ax, 1.8, y_row3 - 0.2, 10.4, 0.6, dark_teal, 
         "5-second windows  •  10% overlap  •  1421 labeled segments", fontsize=10)

# ============================================
# ROW 4: Feature Extraction - DETAILED
# ============================================
y_row4 = 6.3
ax.text(0.3, y_row4 + 0.5, "Feature\nExtraction", fontsize=10, fontweight='bold', color='#333', va='center')

# Arrows down
draw_arrow(ax, (7.0, y_row3 - 0.2), (7.0, y_row4 + 1.1))

# Feature count labels
ax.text(3.3, y_row4 + 0.95, "IMU (30)", fontsize=9, fontweight='bold', color='#333', ha='center')
ax.text(7.0, y_row4 + 0.95, "PPG (183)", fontsize=9, fontweight='bold', color='#333', ha='center')
ax.text(10.7, y_row4 + 0.95, "EDA (47)", fontsize=9, fontweight='bold', color='#333', ha='center')

# IMU features box
draw_box(ax, 1.8, y_row4 - 0.6, 3.0, 1.4, light_teal, 
         "Quantiles, entropy\nKatz fractal dimension\nSum of abs. changes\nVariance of differences", fontsize=8)

# PPG features box - more detailed
draw_box(ax, 5.5, y_row4 - 0.6, 3.0, 1.4, light_teal,
         "Statistical moments\nRMSSD, SDNN, pNN50\nLF/HF power ratio\nHeart rate variability", fontsize=8)

# EDA features box
draw_box(ax, 9.2, y_row4 - 0.6, 3.0, 1.4, light_teal,
         "Tonic level (SCL)\nSlope, IQR, MAD\nMean absolute diff.\nSkewness, kurtosis", fontsize=8)

# ============================================
# ROW 5: Fusion & Alignment
# ============================================
y_row5 = 4.6
ax.text(0.3, y_row5 + 0.1, "Fusion &\nAlignment", fontsize=10, fontweight='bold', color='#333', va='center')

# Arrows converging
for x in [3.3, 7.0, 10.7]:
    draw_arrow(ax, (x, y_row4 - 0.6), (x, y_row5 + 0.35))

draw_box(ax, 1.8, y_row5 - 0.15, 10.4, 0.5, dark_teal,
         "Time alignment  →  Modality fusion  →  Label matching", fontsize=10)

# ============================================
# ROW 6: Feature Selection
# ============================================
y_row6 = 3.5
ax.text(0.3, y_row6 + 0.1, "Feature\nSelection", fontsize=10, fontweight='bold', color='#333', va='center')

draw_arrow(ax, (7.0, y_row5 - 0.15), (7.0, y_row6 + 0.35))

draw_box(ax, 1.8, y_row6 - 0.15, 10.4, 0.5, dark_teal,
         "Correlation ranking  →  Redundancy pruning (r > 0.90)  →  LOSO-consistent filtering", fontsize=10)

# ============================================
# ROW 7: Model Training
# ============================================
y_row7 = 2.4
ax.text(0.3, y_row7 + 0.1, "Model\nTraining", fontsize=10, fontweight='bold', color='#333', va='center')

draw_arrow(ax, (7.0, y_row6 - 0.15), (7.0, y_row7 + 0.35))

draw_box(ax, 1.8, y_row7 - 0.15, 10.4, 0.5, dark_teal,
         "Random Forest (n=100, depth=6)  •  LOSO Cross-Validation  •  Per-modality models", fontsize=10)

# ============================================
# ROW 8: LOSO Results
# ============================================
y_row8 = 1.0
ax.text(0.3, y_row8 + 0.15, "LOSO\nResults", fontsize=10, fontweight='bold', color='#333', va='center')

# Arrows down
for x in [3.3, 7.0, 10.7]:
    draw_arrow(ax, (x, y_row7 - 0.15), (x, y_row8 + 0.45))

draw_box(ax, 1.8, y_row8 - 0.25, 3.0, 0.7, dark_teal, "IMU\nr = 0.52", fontsize=11)
draw_box(ax, 5.5, y_row8 - 0.25, 3.0, 0.7, light_teal, "PPG\nr = 0.26", fontsize=11)
draw_box(ax, 9.2, y_row8 - 0.25, 3.0, 0.7, light_teal, "EDA\nr = 0.02", fontsize=11)

# Key finding text
ax.text(7.0, 0.15, "Key finding: Only motion-based features (IMU) generalize across elderly patients",
        fontsize=9, style='italic', ha='center', color='#555')

# Save
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/62_pipeline_detailed.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/62_pipeline_detailed.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 62_pipeline_detailed.png/pdf")
plt.show()
