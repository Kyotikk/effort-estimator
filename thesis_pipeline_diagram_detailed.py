#!/usr/bin/env python3
"""
Clean Pipeline Diagram for Thesis Presentation
Matches original style - simple and readable
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('white')

# Color scheme - matching original teal/slate style
main_color = '#4A7C8A'      # Teal-slate
light_color = '#6B9AA8'     # Lighter teal
result_color = '#5A8A96'    # Results boxes
label_color = '#2D4A52'     # Dark text

def draw_box(x, y, w, h, text, fontsize=10, color=main_color):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='none',
        alpha=0.95
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, color='white',
            linespacing=1.3)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=main_color, lw=1.5))

def draw_label(x, y, text):
    ax.text(x, y, text, fontsize=10, fontweight='bold', color=label_color, va='center')

# ============================================
# ROW 1: Raw Signals
# ============================================
draw_label(0.3, 9.3, "Raw Signals")

draw_box(2.0, 9.0, 3.0, 0.7, "IMU\n(3-axis accel.)", fontsize=9)
draw_box(5.5, 9.0, 3.0, 0.7, "PPG\n(3 wavelengths)", fontsize=9)
draw_box(9.0, 9.0, 3.0, 0.7, "EDA\n(skin cond.)", fontsize=9)

# Arrows down
draw_arrow(3.5, 9.0, 3.5, 8.5)
draw_arrow(7.0, 9.0, 7.0, 8.5)
draw_arrow(10.5, 9.0, 10.5, 8.5)

# ============================================
# ROW 2: Preprocessing
# ============================================
draw_label(0.3, 8.1, "Preprocessing")

draw_box(2.0, 7.7, 3.0, 0.7, "Resampling\nLow-pass filtering", fontsize=9, color=light_color)
draw_box(5.5, 7.7, 3.0, 0.7, "Resampling\nHigh-pass filtering", fontsize=9, color=light_color)
draw_box(9.0, 7.7, 3.0, 0.7, "Resampling\nTonic/Phasic split", fontsize=9, color=light_color)

# Arrows down
draw_arrow(3.5, 7.7, 3.5, 7.2)
draw_arrow(7.0, 7.7, 7.0, 7.2)
draw_arrow(10.5, 7.7, 10.5, 7.2)

# ============================================
# ROW 3: Temporal Segmentation
# ============================================
draw_label(0.3, 6.8, "Temporal\nSegmentation")

draw_box(2.0, 6.4, 10.0, 0.7, "5-second windows  •  10% overlap  •  1421 labeled segments", fontsize=10)

# Arrow down
draw_arrow(7.0, 6.4, 7.0, 5.9)

# ============================================
# ROW 4: Feature Extraction
# ============================================
draw_label(0.3, 5.3, "Feature\nExtraction")

# IMU label
ax.text(3.5, 5.65, "IMU (30)", fontsize=9, ha='center', color=label_color, fontweight='bold')
draw_box(2.0, 4.6, 3.0, 0.95,
         "Quantiles, entropy\nKatz fractal dimension\nSum of abs. changes\nVariance of differences", 
         fontsize=8, color=light_color)

# PPG label
ax.text(7.0, 5.65, "PPG (183)", fontsize=9, ha='center', color=label_color, fontweight='bold')
draw_box(5.5, 4.6, 3.0, 0.95,
         "Statistical moments\nRMSSD, SDNN, pNN50\nLF/HF power ratio\nHeart rate variability",
         fontsize=8, color=light_color)

# EDA label
ax.text(10.5, 5.65, "EDA (47)", fontsize=9, ha='center', color=label_color, fontweight='bold')
draw_box(9.0, 4.6, 3.0, 0.95,
         "Tonic level (SCL)\nSlope, IQR, MAD\nMean absolute diff.\nSkewness, kurtosis",
         fontsize=8, color=light_color)

# Arrows down (converging)
draw_arrow(3.5, 4.6, 5.5, 4.1)
draw_arrow(7.0, 4.6, 7.0, 4.1)
draw_arrow(10.5, 4.6, 8.5, 4.1)

# ============================================
# ROW 5: Fusion & Alignment
# ============================================
draw_label(0.3, 3.7, "Fusion &\nAlignment")

draw_box(2.0, 3.4, 10.0, 0.6, "Time alignment  →  Modality fusion  →  Label matching", fontsize=10)

# Arrow down
draw_arrow(7.0, 3.4, 7.0, 2.9)

# ============================================
# ROW 6: Feature Selection
# ============================================
draw_label(0.3, 2.5, "Feature\nSelection")

draw_box(2.0, 2.2, 10.0, 0.6, 
         "Correlation ranking  →  Redundancy pruning (r > 0.90)  →  LOSO-consistent filtering", 
         fontsize=10)

# Arrow down
draw_arrow(7.0, 2.2, 7.0, 1.7)

# ============================================
# ROW 7: Model Training
# ============================================
draw_label(0.3, 1.3, "Model\nTraining")

draw_box(2.0, 1.0, 10.0, 0.6,
         "Random Forest (n=100, depth=6)  •  LOSO Cross-Validation  •  Per-modality models",
         fontsize=10)

# Arrows down to results
draw_arrow(3.5, 1.0, 3.5, 0.5)
draw_arrow(7.0, 1.0, 7.0, 0.5)
draw_arrow(10.5, 1.0, 10.5, 0.5)

# ============================================
# ROW 8: LOSO Results
# ============================================
draw_label(0.3, 0.15, "LOSO\nResults")

draw_box(2.0, -0.15, 3.0, 0.6, "IMU\nr = 0.52", fontsize=10, color=result_color)
draw_box(5.5, -0.15, 3.0, 0.6, "PPG\nr = 0.26", fontsize=10, color=result_color)
draw_box(9.0, -0.15, 3.0, 0.6, "EDA\nr = 0.02", fontsize=10, color=result_color)

# Save
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/62_pipeline_clean.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/pascalschlegel/effort-estimator/data/feature_extraction/analysis/62_pipeline_clean.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 62_pipeline_clean.png/pdf")
plt.show()
