#!/usr/bin/env python3
"""
Thesis-Level Pipeline Overview - Clean, unified color scheme
Arrows only between boxes, titles on left side
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Unified color scheme - teal/blue-gray from screenshot
C_PRIMARY = '#3d7a8c'      # Main teal
C_PRIMARY_LIGHT = '#5a9aad'  # Lighter teal
C_PRIMARY_DARK = '#2c5a68'   # Darker teal
C_NEUTRAL = '#8fa4ab'      # Neutral gray-teal
C_TEXT = '#2c3e50'         # Dark text
C_LIGHT_TEXT = '#7f8c8d'   # Light gray text
C_ARROW = '#a8c4cc'        # Light teal for arrows

def draw_box(ax, x, y, w, h, text, color, fontsize=10, textcolor='white', alpha=0.95):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=textcolor, linespacing=1.3)

def draw_arrow_down(ax, x, y1, y2, color=None):
    """Draw vertical arrow from y1 down to y2"""
    if color is None:
        color = C_ARROW
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Box positions
box_left = 2.5    # left edge of content area
box_width = 3.0   # width of each modality box
box_gap = 0.5     # gap between boxes
title_x = 0.8     # x position for left-side titles

# Calculate box centers
x_imu = box_left + box_width/2                           # 4.0
x_ppg = box_left + box_width + box_gap + box_width/2     # 7.5
x_eda = box_left + 2*box_width + 2*box_gap + box_width/2 # 11.0

# =============================================================================
# ROW 1: RAW SIGNALS
# =============================================================================
y1 = 10.0
h1 = 0.55
ax.text(title_x, y1+h1/2, 'Raw Signals\n(32 Hz)', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y1, box_width, h1, 'IMU\n(3-axis accel)', C_PRIMARY, 8)
draw_box(ax, box_left+box_width+box_gap, y1, box_width, h1, 'PPG\n(3 wavelengths)', C_PRIMARY, 8)
draw_box(ax, box_left+2*(box_width+box_gap), y1, box_width, h1, 'EDA\n(skin cond.)', C_PRIMARY, 8)

# =============================================================================
# ROW 2: PREPROCESSING
# =============================================================================
y2 = 8.9
h2 = 0.55
ax.text(title_x, y2+h2/2, 'Preprocessing', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y2, box_width, h2, '32 Hz resample\nLPF 5 Hz', C_PRIMARY_LIGHT, 8)
draw_box(ax, box_left+box_width+box_gap, y2, box_width, h2, '32 Hz resample\nHPF 0.5 Hz', C_PRIMARY_LIGHT, 8)
draw_box(ax, box_left+2*(box_width+box_gap), y2, box_width, h2, '32 Hz resample\nTonic/Phasic split', C_PRIMARY_LIGHT, 8)

# Arrows: Row 1 → Row 2 (from each box center)
draw_arrow_down(ax, x_imu, y1, y2+h2)
draw_arrow_down(ax, x_ppg, y1, y2+h2)
draw_arrow_down(ax, x_eda, y1, y2+h2)

# =============================================================================
# ROW 3: WINDOWING
# =============================================================================
y3 = 7.85
h3 = 0.5
ax.text(title_x, y3+h3/2, 'Temporal\nSegmentation', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y3, 3*box_width+2*box_gap, h3, 
         '5-second windows  •  10% overlap  •  1421 labeled segments', C_PRIMARY, 9)

# Arrows: Row 2 → Row 3 (three arrows converging)
draw_arrow_down(ax, x_imu, y2, y3+h3)
draw_arrow_down(ax, x_ppg, y2, y3+h3)
draw_arrow_down(ax, x_eda, y2, y3+h3)

# =============================================================================
# ROW 4: FEATURE EXTRACTION
# =============================================================================
y4 = 6.0
h4 = 0.85
ax.text(title_x, y4+h4/2, 'Feature\nExtraction', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y4, box_width, h4, 'Movement intensity\nMovement variability\nMovement complexity', 
         C_PRIMARY_LIGHT, 7)
ax.text(x_imu, y4+h4+0.12, 'IMU (30)', fontsize=8, ha='center', fontweight='bold', color=C_PRIMARY_DARK)

draw_box(ax, box_left+box_width+box_gap, y4, box_width, h4, 'Cardiac timing (HR)\nPulse morphology\nHRV statistics', 
         C_PRIMARY_LIGHT, 7)
ax.text(x_ppg, y4+h4+0.12, 'PPG (183)', fontsize=8, ha='center', fontweight='bold', color=C_PRIMARY_DARK)

draw_box(ax, box_left+2*(box_width+box_gap), y4, box_width, h4, 'Tonic arousal level\nPhasic dynamics\nSCR characteristics', 
         C_PRIMARY_LIGHT, 7)
ax.text(x_eda, y4+h4+0.12, 'EDA (47)', fontsize=8, ha='center', fontweight='bold', color=C_PRIMARY_DARK)

# Arrows: Row 3 → Row 4 (diverging from windowing to three feature boxes)
draw_arrow_down(ax, x_imu, y3, y4+h4)
draw_arrow_down(ax, x_ppg, y3, y4+h4)
draw_arrow_down(ax, x_eda, y3, y4+h4)

# =============================================================================
# ROW 5: FUSION & ALIGNMENT
# =============================================================================
y5 = 4.7
h5 = 0.5
ax.text(title_x, y5+h5/2, 'Fusion &\nAlignment', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y5, 3*box_width+2*box_gap, h5, 
         'Time alignment  →  Modality fusion  →  Label matching', C_PRIMARY, 9)

# Arrows: Row 4 → Row 5 (three arrows converging)
draw_arrow_down(ax, x_imu, y4, y5+h5)
draw_arrow_down(ax, x_ppg, y4, y5+h5)
draw_arrow_down(ax, x_eda, y4, y5+h5)

# =============================================================================
# ROW 6: FEATURE SELECTION
# =============================================================================
y6 = 3.7
h6 = 0.5
ax.text(title_x, y6+h6/2, 'Feature\nSelection', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y6, 3*box_width+2*box_gap, h6, 
         'Correlation ranking  →  Redundancy pruning  →  LOSO-consistent filtering', C_PRIMARY, 9)

# Arrow: Row 5 → Row 6
draw_arrow_down(ax, x_ppg, y5, y6+h6)

# =============================================================================
# ROW 7: MODEL TRAINING
# =============================================================================
y7 = 2.7
h7 = 0.5
ax.text(title_x, y7+h7/2, 'Model\nTraining', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

draw_box(ax, box_left, y7, 3*box_width+2*box_gap, h7, 
         'Random Forest Regressor  •  LOSO Cross-Validation  •  Per-modality models', C_PRIMARY, 9)

# Arrow: Row 6 → Row 7
draw_arrow_down(ax, x_ppg, y6, y7+h7)

# =============================================================================
# ROW 8: OUTCOME
# =============================================================================
y8 = 1.5
h8 = 0.65
ax.text(title_x, y8+h8/2, 'LOSO\nResults', fontsize=8, ha='center', va='center',
        fontweight='bold', color=C_TEXT)

# Results - all in teal color scheme
draw_box(ax, box_left, y8, box_width, h8, 'IMU\nr = 0.52', C_PRIMARY_DARK, 10)
draw_box(ax, box_left+box_width+box_gap, y8, box_width, h8, 'PPG\nr = 0.26', C_NEUTRAL, 10, textcolor=C_TEXT)
draw_box(ax, box_left+2*(box_width+box_gap), y8, box_width, h8, 'EDA\nr = 0.02', C_NEUTRAL, 10, textcolor=C_TEXT)

# Arrows: Row 7 → Row 8 (diverging)
draw_arrow_down(ax, x_imu, y7, y8+h8)
draw_arrow_down(ax, x_ppg, y7, y8+h8)
draw_arrow_down(ax, x_eda, y7, y8+h8)

# =============================================================================
# BOTTOM NOTE
# =============================================================================
ax.text(7, 0.9, 'Key finding: Only motion-based features (IMU) generalize across elderly patients', 
        ha='center', fontsize=10, style='italic', color=C_LIGHT_TEXT)

plt.tight_layout()
plt.savefig(f"{OUT}/44_framework_overview.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/44_framework_overview.png")
plt.close()

print("Done!")
