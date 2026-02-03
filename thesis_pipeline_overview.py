#!/usr/bin/env python3
"""
Thesis Pipeline Architecture Visualization
Creates overview diagram showing the complete data processing pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Output directory
OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# =============================================================================
# FIGURE 1: PIPELINE ARCHITECTURE OVERVIEW
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
COLOR_RAW = '#3498db'       # Blue - raw data
COLOR_PREPROCESS = '#9b59b6'  # Purple - preprocessing
COLOR_WINDOW = '#2ecc71'    # Green - windowing
COLOR_FEATURE = '#e67e22'   # Orange - feature extraction
COLOR_ALIGN = '#1abc9c'     # Teal - alignment
COLOR_ML = '#e74c3c'        # Red - machine learning
COLOR_OUTPUT = '#34495e'    # Dark - output

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color='white' if color != '#f1c40f' else 'black',
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """Draw an arrow between points"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(7, 9.5, 'Effort Estimation Pipeline Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold')

# ============= ROW 1: Raw Data Sources =============
y1 = 7.8
draw_box(ax, 0.5, y1, 2.5, 0.9, 'IMU Sensors\n(Accelerometer)', COLOR_RAW, 9)
draw_box(ax, 3.5, y1, 2.5, 0.9, 'PPG Sensor\n(3 wavelengths)', COLOR_RAW, 9)
draw_box(ax, 6.5, y1, 2.5, 0.9, 'EDA Sensor\n(Skin conductance)', COLOR_RAW, 9)
draw_box(ax, 9.5, y1, 2.5, 0.9, 'Borg CR10\n(Effort labels)', COLOR_RAW, 9)

# Sampling rates annotation
ax.text(1.75, y1-0.15, '32 Hz', ha='center', fontsize=8, style='italic')
ax.text(4.75, y1-0.15, '32 Hz', ha='center', fontsize=8, style='italic')
ax.text(7.75, y1-0.15, '32 Hz', ha='center', fontsize=8, style='italic')
ax.text(10.75, y1-0.15, '~1 per task', ha='center', fontsize=8, style='italic')

# ============= ROW 2: Preprocessing =============
y2 = 6.2
draw_box(ax, 0.5, y2, 2.5, 0.9, 'Gravity Removal\nNoise Filtering', COLOR_PREPROCESS, 9)
draw_box(ax, 3.5, y2, 2.5, 0.9, 'HPF (0.5Hz)\nBaseline Removal', COLOR_PREPROCESS, 9)
draw_box(ax, 6.5, y2, 2.5, 0.9, 'Lowpass Filter\nArtifact Removal', COLOR_PREPROCESS, 9)

# Arrows from raw to preprocessing
for x in [1.75, 4.75, 7.75]:
    draw_arrow(ax, x, y1, x, y2+0.9, 'gray')

# ============= ROW 3: Windowing =============
y3 = 4.6
draw_box(ax, 2, y3, 7, 1.0, 'Sliding Window Segmentation\n5s windows, 10% overlap → 1421 windows', COLOR_WINDOW, 11, bold=True)

# Arrows from preprocessing to windowing
draw_arrow(ax, 1.75, y2, 5.5, y3+1.0, 'gray')
draw_arrow(ax, 4.75, y2, 5.5, y3+1.0, 'gray')
draw_arrow(ax, 7.75, y2, 5.5, y3+1.0, 'gray')

# ============= ROW 4: Feature Extraction =============
y4 = 3.0
draw_box(ax, 0.5, y4, 3, 1.0, 'IMU Features (30)\nMean, Std, Energy\nPeak freq, Jerk', COLOR_FEATURE, 9)
draw_box(ax, 4, y4, 3, 1.0, 'PPG Features (183)\nHRV, Pulse metrics\n3 wavelengths', COLOR_FEATURE, 9)
draw_box(ax, 7.5, y4, 3, 1.0, 'EDA Features (47)\nSCL, SCR peaks\nPhasic/Tonic', COLOR_FEATURE, 9)

# Arrows from windowing to feature extraction
draw_arrow(ax, 4, y3, 2, y4+1.0, 'gray')
draw_arrow(ax, 5.5, y3, 5.5, y4+1.0, 'gray')
draw_arrow(ax, 7, y3, 9, y4+1.0, 'gray')

# ============= ROW 5: Alignment & Fusion =============
y5 = 1.5
draw_box(ax, 2, y5, 7, 1.0, 'Temporal Alignment & Borg Label Fusion\nMatch windows to effort ratings (±2s tolerance)', COLOR_ALIGN, 11, bold=True)

# Arrow from Borg labels to alignment
draw_arrow(ax, 10.75, y1, 10.75, y5+0.5, 'gray')
ax.plot([10.75, 9], [y5+0.5, y5+0.5], 'gray', lw=2)

# Arrows from feature extraction to alignment
draw_arrow(ax, 2, y4, 4, y5+1.0, 'gray')
draw_arrow(ax, 5.5, y4, 5.5, y5+1.0, 'gray')
draw_arrow(ax, 9, y4, 7, y5+1.0, 'gray')

# ============= ROW 6: ML Training =============
y6 = 0.2
draw_box(ax, 2.5, y6, 6, 0.9, 'Random Forest (LOSO Cross-Validation)\nLeave-One-Subject-Out for honest evaluation', COLOR_ML, 11, bold=True)

draw_arrow(ax, 5.5, y5, 5.5, y6+0.9, 'gray')

# ============= Results box =============
draw_box(ax, 10, y6-0.1, 3.5, 1.1, 'Results\nr = 0.52\nMAE = 1.64', COLOR_OUTPUT, 11, bold=True)
draw_arrow(ax, 8.5, y6+0.45, 10, y6+0.45, COLOR_ML)

# ============= Legend =============
legend_y = 9.3
legend_items = [
    (COLOR_RAW, 'Raw Data'),
    (COLOR_PREPROCESS, 'Preprocessing'),
    (COLOR_WINDOW, 'Windowing'),
    (COLOR_FEATURE, 'Feature Extraction'),
    (COLOR_ALIGN, 'Alignment'),
    (COLOR_ML, 'ML Training'),
]

for i, (color, label) in enumerate(legend_items):
    x = 0.5 + i * 2.2
    rect = mpatches.Rectangle((x, legend_y), 0.3, 0.2, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + 0.4, legend_y + 0.1, label, fontsize=8, va='center')

# Subjects annotation
ax.text(12.5, 3.5, 'N = 5 subjects\n(elderly)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

plt.tight_layout()
plt.savefig(f"{OUT}/40_pipeline_architecture.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/40_pipeline_architecture.png")
plt.close()


# =============================================================================
# FIGURE 2: DATA FLOW WITH NUMBERS
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(6, 5.6, 'Data Flow: From Raw Signals to Predictions', ha='center', fontsize=14, fontweight='bold')

# Stages
stages = [
    ('Raw\nSignals', '~2h recording\nper subject', COLOR_RAW),
    ('Windows', '1421 total\n(5s, 10% overlap)', COLOR_WINDOW),
    ('Features', '260 features\n(IMU+PPG+EDA)', COLOR_FEATURE),
    ('Aligned', '1421 labeled\nwith Borg', COLOR_ALIGN),
    ('Model', 'RandomForest\nLOSO CV', COLOR_ML),
    ('Output', 'r = 0.52\nMAE = 1.64', COLOR_OUTPUT),
]

x_positions = [0.5, 2.3, 4.1, 5.9, 7.7, 9.8]

for i, (x, (name, detail, color)) in enumerate(zip(x_positions, stages)):
    # Main box
    draw_box(ax, x, 2.5, 1.6, 1.2, name, color, 11, bold=True)
    # Detail below
    ax.text(x + 0.8, 2.2, detail, ha='center', va='top', fontsize=9, style='italic')
    
    # Arrow to next
    if i < len(stages) - 1:
        draw_arrow(ax, x + 1.6, 3.1, x_positions[i+1], 3.1, 'gray')

# Dropout annotation
ax.annotate('Only IMU used\n(PPG/EDA failed LOSO)', xy=(4.9, 2.5), xytext=(4.9, 1.2),
            fontsize=9, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.tight_layout()
plt.savefig(f"{OUT}/41_data_flow.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT}/41_data_flow.png")
plt.close()

print("\nDone! Created pipeline architecture plots.")
