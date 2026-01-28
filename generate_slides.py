#!/usr/bin/env python3
"""Generate presentation slides as PNG images."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Output directory
OUT_DIR = Path('/Users/pascalschlegel/effort-estimator/slides')
OUT_DIR.mkdir(exist_ok=True)

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

def create_slide(title, content_func, filename, subtitle=None):
    """Create a slide with title and custom content."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Background
    ax.add_patch(mpatches.Rectangle((0, 0), 16, 9, facecolor='white', edgecolor='none'))
    
    # Title bar
    ax.add_patch(mpatches.Rectangle((0, 7.8), 16, 1.2, facecolor='#2C3E50', edgecolor='none'))
    ax.text(0.5, 8.4, title, fontsize=28, fontweight='bold', color='white', va='center')
    
    if subtitle:
        ax.text(0.5, 7.95, subtitle, fontsize=16, color='#BDC3C7', va='center')
    
    # Content
    content_func(ax)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created {filename}")

def slide1_objective(ax):
    """Slide 1: Objective"""
    y = 7.0
    ax.text(0.5, y, "Goal:", fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(0.5, y-0.5, "Predict perceived exertion (Borg CR10, 0-10) from wearable sensors", fontsize=18, color='#34495E')
    
    y = 5.5
    ax.text(0.5, y, "Input Signals:", fontsize=20, fontweight='bold', color='#2C3E50')
    signals = [
        ("PPG", "Photoplethysmography → Heart rate, HRV"),
        ("EDA", "Electrodermal activity → Sympathetic arousal"),
        ("IMU", "Accelerometer + Gyroscope → Movement intensity")
    ]
    for i, (sig, desc) in enumerate(signals):
        ax.text(1.0, y-0.6-i*0.5, f"• {sig}:", fontsize=16, fontweight='bold', color='#3498DB')
        ax.text(2.5, y-0.6-i*0.5, desc, fontsize=16, color='#34495E')
    
    y = 3.0
    ax.text(0.5, y, "Output:", fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(0.5, y-0.5, "Continuous effort score (0-10 Borg CR10 scale)", fontsize=18, color='#34495E')
    
    # Borg scale visualization
    ax.add_patch(mpatches.Rectangle((10, 1.5), 5, 4, facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=2))
    ax.text(12.5, 5.2, "Borg CR10 Scale", fontsize=14, fontweight='bold', ha='center', color='#2C3E50')
    scale = [(0, "Nothing"), (2, "Weak"), (4, "Moderate"), (6, "Strong"), (8, "Very Strong"), (10, "Maximal")]
    for val, label in scale:
        color = plt.cm.RdYlGn_r(val/10)
        ax.add_patch(mpatches.Rectangle((10.2, 1.7+val*0.32), 0.5, 0.28, facecolor=color))
        ax.text(10.9, 1.84+val*0.32, f"{val}", fontsize=12, va='center', color='#2C3E50')
        ax.text(11.4, 1.84+val*0.32, label, fontsize=12, va='center', color='#34495E')

def slide2_data(ax):
    """Slide 2: Data Collection"""
    # Table header
    headers = ["Sensor", "Signal", "Rate", "Device"]
    col_x = [0.5, 3.5, 7.5, 10.0]
    y = 7.0
    ax.add_patch(mpatches.Rectangle((0.3, y-0.3), 14, 0.6, facecolor='#3498DB'))
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=16, fontweight='bold', color='white', va='center')
    
    # Table rows
    data = [
        ["PPG", "Green, Infrared, Red", "64 Hz", "Empatica E4"],
        ["EDA", "Skin conductance", "4 Hz", "Empatica E4"],
        ["IMU", "Accel + Gyro (6-axis)", "100 Hz", "Chest sensor"],
    ]
    for i, row in enumerate(data):
        bg = '#ECF0F1' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle((0.3, y-0.9-i*0.6), 14, 0.6, facecolor=bg))
        for j, val in enumerate(row):
            ax.text(col_x[j], y-0.6-i*0.6, val, fontsize=14, color='#2C3E50', va='center')
    
    # Ground truth
    y = 4.0
    ax.text(0.5, y, "Ground Truth:", fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(0.5, y-0.5, "• Borg CR10 ratings collected during ADL tasks", fontsize=16, color='#34495E')
    ax.text(0.5, y-1.0, "• Activities: walking, stair climbing, sitting, household tasks", fontsize=16, color='#34495E')
    
    y = 2.0
    ax.text(0.5, y, "Subjects:", fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(0.5, y-0.5, "• 3 participants (elderly, healthy, clinical condition)", fontsize=16, color='#34495E')
    ax.text(0.5, y-1.0, "• 1199 labeled 10-second windows", fontsize=16, color='#34495E')

def slide3_pipeline(ax):
    """Slide 3: Pipeline Overview"""
    # Pipeline boxes
    steps = ["Raw\nSignals", "Preprocess", "Window", "Extract\nFeatures", "Fuse", "Align\nLabels", "Output"]
    colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = 0.5 + i * 2.2
        ax.add_patch(mpatches.FancyBboxPatch((x, 5.5), 1.8, 1.2, boxstyle="round,pad=0.05", 
                                              facecolor=color, edgecolor='white', linewidth=2))
        ax.text(x+0.9, 6.1, step, fontsize=12, fontweight='bold', color='white', ha='center', va='center')
        if i < len(steps)-1:
            ax.annotate('', xy=(x+2.1, 6.1), xytext=(x+1.9, 6.1),
                       arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))
    
    # Details
    details = [
        ("Preprocess", "Filter noise, normalize signals"),
        ("Window", "10-second segments, 0% overlap"),
        ("Extract", "~300 features per window"),
        ("Fuse", "Combine all modalities on t_center"),
        ("Align", "Match Borg labels to windows"),
    ]
    y = 4.5
    for i, (step, desc) in enumerate(details):
        ax.text(0.5, y-i*0.6, f"• {step}:", fontsize=14, fontweight='bold', color='#3498DB')
        ax.text(3.5, y-i*0.6, desc, fontsize=14, color='#34495E')

def slide4_preprocessing(ax):
    """Slide 4: Preprocessing"""
    # PPG
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 5.0), 4.5, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
    ax.text(2.55, 7.2, "PPG", fontsize=18, fontweight='bold', color='#E74C3C', ha='center')
    ax.text(0.5, 6.6, "• Bandpass: 0.5–4.0 Hz", fontsize=13, color='#2C3E50')
    ax.text(0.5, 6.1, "• Preserves 30-240 BPM", fontsize=13, color='#2C3E50')
    ax.text(0.5, 5.6, "• Z-score normalization", fontsize=13, color='#2C3E50')
    
    # EDA
    ax.add_patch(mpatches.FancyBboxPatch((5.3, 5.0), 4.5, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(7.55, 7.2, "EDA", fontsize=18, fontweight='bold', color='#27AE60', ha='center')
    ax.text(5.5, 6.6, "• Lowpass: 1.0 Hz", fontsize=13, color='#2C3E50')
    ax.text(5.5, 6.1, "• Decompose: tonic + phasic", fontsize=13, color='#2C3E50')
    ax.text(5.5, 5.6, "• Remove motion artifacts", fontsize=13, color='#2C3E50')
    
    # IMU
    ax.add_patch(mpatches.FancyBboxPatch((10.3, 5.0), 4.5, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#D6EAF8', edgecolor='#3498DB', linewidth=2))
    ax.text(12.55, 7.2, "IMU", fontsize=18, fontweight='bold', color='#3498DB', ha='center')
    ax.text(10.5, 6.6, "• Gravity separation", fontsize=13, color='#2C3E50')
    ax.text(10.5, 6.1, "• Static vs dynamic accel", fontsize=13, color='#2C3E50')
    ax.text(10.5, 5.6, "• Magnitude: √(x²+y²+z²)", fontsize=13, color='#2C3E50')
    
    # Quality
    ax.text(0.5, 4.2, "Quality Checks:", fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(0.5, 3.6, "• Reject windows with >20% missing data", fontsize=14, color='#34495E')
    ax.text(0.5, 3.1, "• Detect signal clipping", fontsize=14, color='#34495E')
    ax.text(0.5, 2.6, "• Flag excessive motion artifacts", fontsize=14, color='#34495E')

def slide5_windowing(ax):
    """Slide 5: Windowing & Why 0% Overlap"""
    ax.text(0.5, 7.2, "Window Parameters:", fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(1.0, 6.6, "• Window size: 10 seconds", fontsize=16, color='#34495E')
    ax.text(1.0, 6.1, "• Overlap: 0% (critical!)", fontsize=16, fontweight='bold', color='#E74C3C')
    
    # 70% overlap visualization (BAD)
    ax.add_patch(mpatches.Rectangle((0.5, 4.0), 6, 1.8, facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
    ax.text(3.5, 5.5, "❌ 70% Overlap (WRONG)", fontsize=14, fontweight='bold', color='#E74C3C', ha='center')
    
    # Window 1
    ax.add_patch(mpatches.Rectangle((0.8, 4.6), 2.5, 0.4, facecolor='#3498DB', alpha=0.7))
    ax.text(2.05, 4.8, "Win 1: [0-10s]", fontsize=10, color='white', ha='center', va='center')
    # Window 2 (overlapping)
    ax.add_patch(mpatches.Rectangle((1.55, 4.15), 2.5, 0.4, facecolor='#E74C3C', alpha=0.7))
    ax.text(2.8, 4.35, "Win 2: [3-13s]", fontsize=10, color='white', ha='center', va='center')
    ax.text(5.5, 4.4, "7s shared!\n→ LEAKAGE", fontsize=11, color='#E74C3C', fontweight='bold')
    
    # 0% overlap visualization (GOOD)
    ax.add_patch(mpatches.Rectangle((8, 4.0), 6, 1.8, facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(11, 5.5, "✓ 0% Overlap (CORRECT)", fontsize=14, fontweight='bold', color='#27AE60', ha='center')
    
    # Window 1
    ax.add_patch(mpatches.Rectangle((8.3, 4.6), 2.2, 0.4, facecolor='#3498DB', alpha=0.7))
    ax.text(9.4, 4.8, "Win 1: [0-10s]", fontsize=10, color='white', ha='center', va='center')
    # Window 2 (no overlap)
    ax.add_patch(mpatches.Rectangle((10.7, 4.6), 2.2, 0.4, facecolor='#27AE60', alpha=0.7))
    ax.text(11.8, 4.8, "Win 2: [10-20s]", fontsize=10, color='white', ha='center', va='center')
    ax.text(13.2, 4.4, "0s shared\n→ NO LEAK", fontsize=11, color='#27AE60', fontweight='bold')
    
    # Explanation
    ax.text(0.5, 3.2, "Why this matters:", fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(0.5, 2.6, "• With overlap: if Window 1 in train and Window 2 in test → they share data!", fontsize=14, color='#34495E')
    ax.text(0.5, 2.1, "• Model \"cheats\" by learning from test data patterns", fontsize=14, color='#34495E')
    ax.text(0.5, 1.6, "• 0% overlap ensures complete independence between windows", fontsize=14, color='#27AE60', fontweight='bold')

def slide6_features_ppg(ax):
    """Slide 6: PPG Features"""
    ax.text(0.5, 7.2, "PPG Time-Domain Features (per wavelength: green, infrared, red):", fontsize=16, fontweight='bold', color='#2C3E50')
    
    features = [
        ("Statistical", "mean, std, min, max, median"),
        ("Percentiles", "p5, p10, p25, p50, p75, p90, p95, p99"),
        ("Distribution", "skewness, kurtosis, IQR, range"),
        ("Energy", "RMS, trim mean, TKE"),
    ]
    y = 6.4
    for cat, feats in features:
        ax.text(0.8, y, f"• {cat}:", fontsize=14, fontweight='bold', color='#3498DB')
        ax.text(3.5, y, feats, fontsize=14, color='#34495E')
        y -= 0.5
    
    ax.text(0.5, 4.2, "Derivative Features:", fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(0.8, 3.7, "• First derivative: mean, std, mean_abs", fontsize=14, color='#34495E')
    ax.text(0.8, 3.2, "• Second derivative: std (signal acceleration)", fontsize=14, color='#34495E')
    
    # Summary box
    ax.add_patch(mpatches.FancyBboxPatch((10, 3.0), 5, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#EBF5FB', edgecolor='#3498DB', linewidth=2))
    ax.text(12.5, 5.2, "Total PPG Features", fontsize=14, fontweight='bold', color='#3498DB', ha='center')
    ax.text(12.5, 4.5, "~40 per wavelength", fontsize=16, color='#2C3E50', ha='center')
    ax.text(12.5, 3.9, "× 3 wavelengths", fontsize=16, color='#2C3E50', ha='center')
    ax.text(12.5, 3.3, "= 120 features", fontsize=18, fontweight='bold', color='#E74C3C', ha='center')

def slide7_features_hrv(ax):
    """Slide 7: HRV Features"""
    ax.text(0.5, 7.2, "Heart Rate Variability (from PPG peak detection):", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # HRV features table
    headers = ["Feature", "Meaning", "↔ Effort"]
    col_x = [0.5, 4.0, 11.5]
    y = 6.4
    ax.add_patch(mpatches.Rectangle((0.3, y-0.25), 14, 0.5, facecolor='#9B59B6'))
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=14, fontweight='bold', color='white', va='center')
    
    data = [
        ["hr_mean", "Mean heart rate (BPM)", "↑ effort → ↑ HR"],
        ["mean_ibi", "Mean inter-beat interval (ms)", "↑ effort → ↓ IBI"],
        ["sdnn", "Std of IBI (overall HRV)", "↑ effort → ↓ SDNN"],
        ["rmssd", "Successive diff RMS (parasympathetic)", "↑ effort → ↓ RMSSD"],
    ]
    for i, row in enumerate(data):
        bg = '#F5EEF8' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle((0.3, y-0.75-i*0.55), 14, 0.55, facecolor=bg))
        for j, val in enumerate(row):
            color = '#27AE60' if j == 2 else '#2C3E50'
            ax.text(col_x[j], y-0.5-i*0.55, val, fontsize=13, color=color, va='center', 
                   fontweight='bold' if j == 2 else 'normal')
    
    # Validation box
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.5), 14, 2.2, boxstyle="round,pad=0.1", 
                                          facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(7.3, 3.4, "✓ Validated Correlations (physiologically correct)", fontsize=16, fontweight='bold', color='#27AE60', ha='center')
    ax.text(1.0, 2.7, "HR vs Borg: r = +0.43", fontsize=14, color='#2C3E50')
    ax.text(5.5, 2.7, "IBI vs Borg: r = -0.46", fontsize=14, color='#2C3E50')
    ax.text(10.0, 2.7, "RMSSD vs Borg: r = -0.23", fontsize=14, color='#2C3E50')
    ax.text(7.3, 2.0, "All correlations match expected physiological direction!", fontsize=14, color='#27AE60', ha='center', fontweight='bold')

def slide8_features_eda_imu(ax):
    """Slide 8: EDA & IMU Features"""
    # EDA section
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 5.0), 7, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(3.8, 7.2, "EDA Features", fontsize=18, fontweight='bold', color='#27AE60', ha='center')
    eda_feats = ["• mean, std, slope", "• SCR count (responses)", "• SCR amplitude", "• Tonic + Phasic levels"]
    for i, f in enumerate(eda_feats):
        ax.text(0.6, 6.5-i*0.45, f, fontsize=13, color='#2C3E50')
    
    # IMU section
    ax.add_patch(mpatches.FancyBboxPatch((8, 5.0), 7, 2.5, boxstyle="round,pad=0.1", 
                                          facecolor='#D6EAF8', edgecolor='#3498DB', linewidth=2))
    ax.text(11.5, 7.2, "IMU Features", fontsize=18, fontweight='bold', color='#3498DB', ha='center')
    imu_feats = ["• mean, std, variance", "• energy, entropy", "• zero crossings", "• peak count"]
    for i, f in enumerate(imu_feats):
        ax.text(8.3, 6.5-i*0.45, f, fontsize=13, color='#2C3E50')
    
    # IMU axes
    ax.text(0.5, 4.2, "IMU computed per axis:", fontsize=14, fontweight='bold', color='#2C3E50')
    ax.text(0.8, 3.7, "acc_x, acc_y, acc_z, acc_x_dyn, acc_y_dyn, acc_z_dyn, gyro_x, gyro_y, gyro_z, acc_mag, gyro_mag", 
           fontsize=12, color='#34495E')
    
    # Total
    ax.add_patch(mpatches.FancyBboxPatch((5, 1.5), 6, 1.5, boxstyle="round,pad=0.1", 
                                          facecolor='#FCF3CF', edgecolor='#F39C12', linewidth=2))
    ax.text(8, 2.7, "Total Features", fontsize=16, fontweight='bold', color='#F39C12', ha='center')
    ax.text(8, 2.1, "~300 per window", fontsize=20, fontweight='bold', color='#2C3E50', ha='center')

def slide9_fusion(ax):
    """Slide 9: Feature Fusion"""
    ax.text(0.5, 7.2, "Merge all modalities on window center timestamp:", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Fusion diagram
    modalities = [("PPG", '#E74C3C'), ("HRV", '#9B59B6'), ("EDA", '#27AE60'), ("IMU", '#3498DB')]
    for i, (mod, color) in enumerate(modalities):
        ax.add_patch(mpatches.FancyBboxPatch((1, 5.5-i*0.8), 2.5, 0.6, boxstyle="round,pad=0.05", 
                                              facecolor=color, edgecolor='white', linewidth=2))
        ax.text(2.25, 5.8-i*0.8, f"{mod} features", fontsize=14, color='white', ha='center', va='center')
        # Arrow
        ax.annotate('', xy=(5.5, 4.5), xytext=(3.7, 5.8-i*0.8),
                   arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))
    
    # Merge box
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 3.8), 3.5, 1.5, boxstyle="round,pad=0.1", 
                                          facecolor='#F39C12', edgecolor='white', linewidth=2))
    ax.text(7.25, 4.8, "MERGE", fontsize=18, fontweight='bold', color='white', ha='center')
    ax.text(7.25, 4.3, "on t_center", fontsize=14, color='white', ha='center')
    
    # Arrow to output
    ax.annotate('', xy=(11, 4.5), xytext=(9.2, 4.5),
               arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))
    
    # Output
    ax.add_patch(mpatches.FancyBboxPatch((11, 3.5), 4, 2, boxstyle="round,pad=0.1", 
                                          facecolor='#1ABC9C', edgecolor='white', linewidth=2))
    ax.text(13, 5.0, "Combined", fontsize=16, fontweight='bold', color='white', ha='center')
    ax.text(13, 4.4, "Features", fontsize=16, fontweight='bold', color='white', ha='center')
    ax.text(13, 3.8, "~300 cols", fontsize=14, color='white', ha='center')
    
    # Explanation
    ax.text(0.5, 2.3, "Key: Inner join ensures all modalities present for each window", fontsize=14, color='#34495E')
    ax.text(0.5, 1.8, "Result: One row per 10-second window with all sensor features", fontsize=14, color='#34495E')

def slide10_alignment(ax):
    """Slide 10: Label Alignment"""
    ax.text(0.5, 7.2, "Match Borg ratings to feature windows via ADL intervals:", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # ADL labels format
    ax.add_patch(mpatches.Rectangle((0.5, 5.5), 6, 1.3, facecolor='#EBF5FB', edgecolor='#3498DB', linewidth=2))
    ax.text(3.5, 6.5, "ADL Labels", fontsize=14, fontweight='bold', color='#3498DB', ha='center')
    ax.text(0.8, 6.0, "t_start | t_end | activity | borg", fontsize=11, family='monospace', color='#2C3E50')
    ax.text(0.8, 5.7, "0      | 60   | walking | 3", fontsize=11, family='monospace', color='#34495E')
    
    # Feature windows
    ax.add_patch(mpatches.Rectangle((8, 5.5), 6, 1.3, facecolor='#FCF3CF', edgecolor='#F39C12', linewidth=2))
    ax.text(11, 6.5, "Feature Windows", fontsize=14, fontweight='bold', color='#F39C12', ha='center')
    ax.text(8.3, 6.0, "t_center | features...", fontsize=11, family='monospace', color='#2C3E50')
    ax.text(8.3, 5.7, "5       | [300 values]", fontsize=11, family='monospace', color='#34495E')
    
    # Alignment arrow
    ax.annotate('', xy=(7.8, 5.0), xytext=(6.7, 5.0),
               arrowprops=dict(arrowstyle='->', color='#27AE60', lw=3))
    ax.text(7.25, 4.5, "Match if:\nt_start ≤ t_center ≤ t_end", fontsize=12, color='#27AE60', ha='center')
    
    # Result
    ax.add_patch(mpatches.FancyBboxPatch((4, 2.5), 8, 1.5, boxstyle="round,pad=0.1", 
                                          facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(8, 3.7, "Alignment Result", fontsize=16, fontweight='bold', color='#27AE60', ha='center')
    ax.text(8, 3.1, "1199 labeled windows from 3 subjects", fontsize=16, color='#2C3E50', ha='center')
    
    # Note
    ax.text(0.5, 1.5, "Note: Using merge_asof with 5-second tolerance for minor timestamp misalignments", 
           fontsize=13, color='#7F8C8D', style='italic')

def slide11_sanitization(ax):
    """Slide 11: Data Sanitization"""
    ax.text(0.5, 7.2, "Remove metadata columns (prevent leakage):", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Excluded
    ax.add_patch(mpatches.Rectangle((0.5, 5.0), 6, 2.0, facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
    ax.text(3.5, 6.7, "❌ EXCLUDED", fontsize=14, fontweight='bold', color='#E74C3C', ha='center')
    excluded = ["t_start, t_end (time info)", "window_id, *_idx (identifiers)", "subject, activity (metadata)"]
    for i, e in enumerate(excluded):
        ax.text(0.8, 6.1-i*0.45, f"• {e}", fontsize=13, color='#C0392B')
    
    # Kept
    ax.add_patch(mpatches.Rectangle((8, 5.0), 6, 2.0, facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=2))
    ax.text(11, 6.7, "✓ KEPT", fontsize=14, fontweight='bold', color='#27AE60', ha='center')
    kept = ["All numeric feature columns", "t_center (for alignment only)", "Columns with <50% NaN"]
    for i, k in enumerate(kept):
        ax.text(8.3, 6.1-i*0.45, f"• {k}", fontsize=13, color='#27AE60')
    
    # NaN handling
    ax.text(0.5, 4.2, "NaN Handling:", fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(0.8, 3.7, "• Drop columns with >50% missing values", fontsize=14, color='#34495E')
    ax.text(0.8, 3.2, "• Impute remaining NaN with median (robust to outliers)", fontsize=14, color='#34495E')
    ax.text(0.8, 2.7, "• HRV features: ~16% NaN (windows with <5 detected beats)", fontsize=14, color='#34495E')
    
    # Final
    ax.add_patch(mpatches.FancyBboxPatch((5, 1.3), 6, 1.0, boxstyle="round,pad=0.1", 
                                          facecolor='#1ABC9C', edgecolor='white', linewidth=2))
    ax.text(8, 1.8, "Final: 299 usable features", fontsize=16, fontweight='bold', color='white', ha='center')

def slide12_validation(ax):
    """Slide 12: Validation Approach"""
    ax.text(0.5, 7.2, "Leave-One-Subject-Out Cross-Validation (LOSO):", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # LOSO diagram
    folds = [
        ("Fold 1", "Train: healthy + severe", "Test: elderly"),
        ("Fold 2", "Train: elderly + severe", "Test: healthy"),
        ("Fold 3", "Train: elderly + healthy", "Test: severe"),
    ]
    for i, (fold, train, test) in enumerate(folds):
        y = 6.0 - i * 1.2
        ax.text(0.8, y, fold, fontsize=14, fontweight='bold', color='#3498DB')
        ax.add_patch(mpatches.Rectangle((2.5, y-0.25), 5, 0.5, facecolor='#D6EAF8', edgecolor='#3498DB'))
        ax.text(5, y, train, fontsize=12, color='#2C3E50', ha='center', va='center')
        ax.text(7.8, y, "→", fontsize=16, color='#7F8C8D')
        ax.add_patch(mpatches.Rectangle((8.5, y-0.25), 3, 0.5, facecolor='#FADBD8', edgecolor='#E74C3C'))
        ax.text(10, y, test, fontsize=12, color='#2C3E50', ha='center', va='center')
    
    # Why LOSO
    ax.text(0.5, 2.5, "Why LOSO?", fontsize=16, fontweight='bold', color='#2C3E50')
    reasons = [
        "• Tests generalization to completely new individuals",
        "• Prevents subject-specific pattern leakage",
        "• Realistic deployment scenario (model sees new person)"
    ]
    for i, r in enumerate(reasons):
        ax.text(0.8, 2.0-i*0.45, r, fontsize=14, color='#34495E')

def slide13_correlations(ax):
    """Slide 13: Physiological Validation"""
    ax.text(0.5, 7.2, "Correlations with Borg (before any training):", fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Table
    headers = ["Feature", "Expected", "Observed", "Status"]
    col_x = [0.5, 4.5, 7.5, 10.5]
    y = 6.4
    ax.add_patch(mpatches.Rectangle((0.3, y-0.25), 13, 0.5, facecolor='#2C3E50'))
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=14, fontweight='bold', color='white', va='center')
    
    data = [
        ["HR (heart rate)", "+", "r = +0.43", "✓"],
        ["IBI (beat interval)", "−", "r = −0.46", "✓"],
        ["RMSSD (HRV)", "−", "r = −0.23", "✓"],
        ["EDA (skin conduct.)", "+", "r = +0.15", "✓"],
        ["ACC magnitude", "+", "r = +0.38", "✓"],
    ]
    for i, row in enumerate(data):
        bg = '#D5F5E3' if i % 2 == 0 else '#E8F8F5'
        ax.add_patch(mpatches.Rectangle((0.3, y-0.75-i*0.5), 13, 0.5, facecolor=bg))
        for j, val in enumerate(row):
            color = '#27AE60' if j == 3 else '#2C3E50'
            ax.text(col_x[j], y-0.5-i*0.5, val, fontsize=13, color=color, va='center',
                   fontweight='bold' if j == 3 else 'normal')
    
    # Conclusion
    ax.add_patch(mpatches.FancyBboxPatch((2, 1.5), 12, 1.2, boxstyle="round,pad=0.1", 
                                          facecolor='#27AE60', edgecolor='white', linewidth=2))
    ax.text(8, 2.3, "✓ All correlations match physiological expectations!", fontsize=18, fontweight='bold', color='white', ha='center')
    ax.text(8, 1.8, "Pipeline methodology is scientifically valid", fontsize=14, color='white', ha='center')

def slide14_summary(ax):
    """Slide 14: Dataset Summary"""
    ax.text(0.5, 7.2, "Final Dataset Summary:", fontsize=20, fontweight='bold', color='#2C3E50')
    
    # Summary table
    items = [
        ("Subjects", "3 (elderly, healthy, clinical)"),
        ("Labeled windows", "1199"),
        ("Features", "299"),
        ("Window size", "10 seconds"),
        ("Overlap", "0% (no leakage)"),
        ("Borg range", "0-10 (mean ~4.5)"),
    ]
    for i, (label, value) in enumerate(items):
        y = 6.3 - i * 0.7
        ax.text(1.0, y, f"{label}:", fontsize=16, fontweight='bold', color='#3498DB')
        ax.text(6.0, y, value, fontsize=16, color='#2C3E50')
    
    # Feature breakdown
    ax.add_patch(mpatches.FancyBboxPatch((10, 3.5), 5, 3.5, boxstyle="round,pad=0.1", 
                                          facecolor='#EBF5FB', edgecolor='#3498DB', linewidth=2))
    ax.text(12.5, 6.7, "Feature Breakdown", fontsize=14, fontweight='bold', color='#3498DB', ha='center')
    breakdown = [("PPG", "~120"), ("HRV", "~18"), ("EDA", "~20"), ("IMU", "~140")]
    for i, (cat, num) in enumerate(breakdown):
        ax.text(10.5, 6.1-i*0.5, f"• {cat}:", fontsize=13, color='#2C3E50')
        ax.text(13.5, 6.1-i*0.5, num, fontsize=13, color='#34495E')

def slide15_flow(ax):
    """Slide 15: Pipeline Flow Diagram"""
    # Vertical flow
    steps = [
        ("RAW DATA", "PPG + EDA + IMU + Borg", '#E74C3C'),
        ("PREPROCESS", "Filter, normalize, clean", '#E67E22'),
        ("WINDOW", "10s segments, 0% overlap", '#F1C40F'),
        ("EXTRACT", "~300 features per window", '#2ECC71'),
        ("FUSE + ALIGN", "Combine modalities + Borg labels", '#3498DB'),
        ("SANITIZE", "Remove metadata, impute NaN", '#9B59B6'),
        ("OUTPUT", "1199 × 299 matrix", '#1ABC9C'),
    ]
    
    for i, (step, desc, color) in enumerate(steps):
        y = 7.0 - i * 0.95
        ax.add_patch(mpatches.FancyBboxPatch((2, y-0.35), 3.5, 0.7, boxstyle="round,pad=0.05", 
                                              facecolor=color, edgecolor='white', linewidth=2))
        ax.text(3.75, y, step, fontsize=13, fontweight='bold', color='white', ha='center', va='center')
        ax.text(6, y, desc, fontsize=13, color='#34495E', va='center')
        if i < len(steps) - 1:
            ax.annotate('', xy=(3.75, y-0.45), xytext=(3.75, y-0.85),
                       arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))

def slide16_decisions(ax):
    """Slide 16: Key Design Decisions"""
    ax.text(0.5, 7.2, "Key Design Decisions:", fontsize=20, fontweight='bold', color='#2C3E50')
    
    decisions = [
        ("Window size", "10 seconds", "Standard for HRV; captures 10-15 heartbeats"),
        ("Overlap", "0%", "Prevents temporal leakage between train/test"),
        ("NaN threshold", "50%", "Balance data retention vs. feature quality"),
        ("Imputation", "Median", "Robust to outliers, preserves distribution"),
        ("Validation", "LOSO CV", "Tests cross-subject generalization"),
    ]
    
    # Header
    col_x = [0.5, 4.0, 6.5]
    y = 6.5
    ax.add_patch(mpatches.Rectangle((0.3, y-0.25), 14.5, 0.5, facecolor='#2C3E50'))
    for i, h in enumerate(["Decision", "Choice", "Rationale"]):
        ax.text(col_x[i], y, h, fontsize=14, fontweight='bold', color='white', va='center')
    
    for i, (dec, choice, reason) in enumerate(decisions):
        bg = '#ECF0F1' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle((0.3, y-0.75-i*0.6), 14.5, 0.6, facecolor=bg))
        ax.text(col_x[0], y-0.45-i*0.6, dec, fontsize=13, color='#2C3E50', va='center')
        ax.text(col_x[1], y-0.45-i*0.6, choice, fontsize=13, fontweight='bold', color='#3498DB', va='center')
        ax.text(col_x[2], y-0.45-i*0.6, reason, fontsize=12, color='#34495E', va='center')

def slide17_limitations(ax):
    """Slide 17: Current Limitations"""
    ax.text(0.5, 7.2, "Current Limitations:", fontsize=20, fontweight='bold', color='#E74C3C')
    
    limitations = [
        ("Small sample size", "n=3 subjects", "Model overfits to individual patterns;\nLOSO CV shows poor generalization"),
        ("Sparse labels", "1 Borg per ADL task", "Not continuous during activity;\nlimits temporal resolution"),
        ("HRV coverage", "~16% windows missing", "Insufficient detected heartbeats;\nrequires imputation"),
    ]
    
    for i, (title, issue, impact) in enumerate(limitations):
        y = 6.0 - i * 1.8
        ax.add_patch(mpatches.FancyBboxPatch((0.5, y-0.8), 14, 1.6, boxstyle="round,pad=0.1", 
                                              facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
        ax.text(1.0, y+0.4, f"{i+1}. {title}", fontsize=16, fontweight='bold', color='#C0392B')
        ax.text(1.0, y-0.1, f"Issue: {issue}", fontsize=13, color='#2C3E50')
        ax.text(1.0, y-0.5, f"Impact: {impact}", fontsize=12, color='#7F8C8D')

def slide18_next_steps(ax):
    """Slide 18: Next Steps"""
    ax.text(0.5, 7.2, "Recommended Next Steps:", fontsize=20, fontweight='bold', color='#2C3E50')
    
    steps = [
        ("Add more subjects", "Improve cross-subject generalization", '#27AE60'),
        ("Continuous Borg collection", "Finer temporal resolution of effort", '#3498DB'),
        ("Feature selection", "Reduce from 300 to most predictive features", '#9B59B6'),
        ("Simpler models", "Less overfitting with small sample size", '#E67E22'),
        ("Domain adaptation", "Transfer learning across subjects", '#E74C3C'),
    ]
    
    for i, (step, desc, color) in enumerate(steps):
        y = 6.0 - i * 1.0
        ax.add_patch(mpatches.Circle((1.0, y), 0.3, facecolor=color, edgecolor='white', linewidth=2))
        ax.text(1.0, y, str(i+1), fontsize=14, fontweight='bold', color='white', ha='center', va='center')
        ax.text(1.6, y+0.1, step, fontsize=16, fontweight='bold', color='#2C3E50')
        ax.text(1.6, y-0.35, desc, fontsize=13, color='#7F8C8D')

def slide19_summary_final(ax):
    """Slide 19: Final Summary"""
    ax.text(8, 7.0, "Summary", fontsize=24, fontweight='bold', color='#2C3E50', ha='center')
    
    # Methodology correct
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 4.5), 7, 2.2, boxstyle="round,pad=0.1", 
                                          facecolor='#D5F5E3', edgecolor='#27AE60', linewidth=3))
    ax.text(4, 6.4, "✓ Methodology Correct", fontsize=18, fontweight='bold', color='#27AE60', ha='center')
    correct = ["No temporal leakage (0% overlap)", "No metadata leakage (sanitized)", 
               "HRV features included", "Physiologically valid correlations"]
    for i, c in enumerate(correct):
        ax.text(0.8, 5.8-i*0.4, f"• {c}", fontsize=12, color='#27AE60')
    
    # Limitation
    ax.add_patch(mpatches.FancyBboxPatch((8.5, 4.5), 7, 2.2, boxstyle="round,pad=0.1", 
                                          facecolor='#FCF3CF', edgecolor='#F39C12', linewidth=3))
    ax.text(12, 6.4, "⚠ Generalization Limited", fontsize=18, fontweight='bold', color='#F39C12', ha='center')
    limits = ["Need more subjects for robust", "cross-subject prediction", 
              "Current data good for", "proof-of-concept"]
    for i, l in enumerate(limits):
        ax.text(8.8, 5.8-i*0.4, f"  {l}", fontsize=12, color='#F39C12')
    
    # Bottom line
    ax.add_patch(mpatches.FancyBboxPatch((2, 1.5), 12, 1.5, boxstyle="round,pad=0.1", 
                                          facecolor='#3498DB', edgecolor='white', linewidth=3))
    ax.text(8, 2.5, "Pipeline is scientifically sound and ready", fontsize=18, fontweight='bold', color='white', ha='center')
    ax.text(8, 1.9, "for scaling with additional subjects", fontsize=18, fontweight='bold', color='white', ha='center')

# Generate all slides
if __name__ == '__main__':
    print("Generating presentation slides...")
    
    create_slide("Borg CR10 Effort Estimation", slide1_objective, "slide01_objective.png", "Predicting Perceived Exertion from Wearables")
    create_slide("Data Collection", slide2_data, "slide02_data.png", "Sensors and Ground Truth")
    create_slide("Pipeline Overview", slide3_pipeline, "slide03_pipeline.png", "End-to-End Processing Flow")
    create_slide("Preprocessing", slide4_preprocessing, "slide04_preprocessing.png", "Signal-Specific Filtering and Cleaning")
    create_slide("Windowing", slide5_windowing, "slide05_windowing.png", "Why 0% Overlap Matters")
    create_slide("Feature Extraction: PPG", slide6_features_ppg, "slide06_features_ppg.png", "Time-Domain and Derivative Features")
    create_slide("Feature Extraction: HRV", slide7_features_hrv, "slide07_features_hrv.png", "Heart Rate Variability Metrics")
    create_slide("Feature Extraction: EDA & IMU", slide8_features_eda_imu, "slide08_features_eda_imu.png", "Electrodermal and Motion Features")
    create_slide("Feature Fusion", slide9_fusion, "slide09_fusion.png", "Combining All Modalities")
    create_slide("Label Alignment", slide10_alignment, "slide10_alignment.png", "Matching Borg Ratings to Windows")
    create_slide("Data Sanitization", slide11_sanitization, "slide11_sanitization.png", "Preventing Leakage and Handling NaN")
    create_slide("Validation Approach", slide12_validation, "slide12_validation.png", "Leave-One-Subject-Out Cross-Validation")
    create_slide("Physiological Validation", slide13_correlations, "slide13_correlations.png", "Verifying Expected Correlations")
    create_slide("Dataset Summary", slide14_summary, "slide14_summary.png", "Final Numbers")
    create_slide("Pipeline Flow", slide15_flow, "slide15_flow.png", "Complete Processing Chain")
    create_slide("Key Design Decisions", slide16_decisions, "slide16_decisions.png", "Choices and Rationale")
    create_slide("Current Limitations", slide17_limitations, "slide17_limitations.png", "Known Issues")
    create_slide("Next Steps", slide18_next_steps, "slide18_next_steps.png", "Recommended Improvements")
    create_slide("Summary", slide19_summary_final, "slide19_summary.png", "Methodology Status")
    
    print(f"\n✓ All slides saved to: {OUT_DIR}")
    print(f"  Total: 19 slides")
