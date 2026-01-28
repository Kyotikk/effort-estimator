#!/usr/bin/env python3
"""
Generate detailed presentation slides on methodology for supervisor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = '/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("PRESENTATION: METHODOLOGY DETAILS")
print("="*80)

#############################################################################
# SLIDE 1: PIPELINE OVERVIEW
#############################################################################

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111)
ax.axis('off')

title = "Effort Estimation Pipeline: Methodology Overview"
ax.text(0.5, 0.98, title, fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

pipeline_text = """
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           1. RAW DATA ACQUISITION                                       │
│                                                                                         │
│    Wrist-worn wearable (Empatica/similar) during Activities of Daily Living            │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                                           │
│    │   PPG   │    │   EDA   │    │   IMU   │    + Borg CR10 self-reports               │
│    │ 32 Hz   │    │ 32 Hz   │    │ 32 Hz   │                                           │
│    └────┬────┘    └────┬────┘    └────┬────┘                                           │
└─────────┼──────────────┼──────────────┼────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           2. PREPROCESSING                                              │
│                                                                                         │
│    PPG:                      EDA:                     IMU:                              │
│    • Linear interpolation    • Linear interpolation   • Linear interpolation            │
│    • Resample to 32 Hz       • Resample to 32 Hz      • Resample to 32 Hz              │
│    • Optional: HPF 0.5 Hz    • Keep raw + quality     • Normalize                       │
│      (remove baseline drift)   columns                                                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           3. WINDOWING                                                  │
│                                                                                         │
│    Sliding windows:                                                                     │
│    • Window size: 10 seconds                                                            │
│    • Overlap: 70% (3 sec hop)                                                          │
│    • Each window → one feature vector                                                   │
│                                                                                         │
│    ┌──────────┐                                                                        │
│    │ Window 1 │──────────────────┐                                                     │
│    └──────────┘                  │                                                      │
│       ┌──────────┐               │  70% overlap                                         │
│       │ Window 2 │───────────────┼──┐                                                   │
│       └──────────┘               │  │                                                   │
│          ┌──────────┐            │  │                                                   │
│          │ Window 3 │────────────┼──┼──┐                                               │
│          └──────────┘            │  │  │                                               │
│    Time ─────────────────────────┴──┴──┴─────────────────►                              │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.02, 0.92, pipeline_text, fontsize=9, family='monospace', va='top', transform=ax.transAxes)

plt.savefig(f'{output_dir}/slide1_pipeline_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide1_pipeline_overview.png")
plt.close()

#############################################################################
# SLIDE 2: FEATURE EXTRACTION DETAILS
#############################################################################

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111)
ax.axis('off')

ax.text(0.5, 0.98, "Feature Extraction: What We Compute", fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

features_text = """
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         PPG FEATURES (Heart Rate Variability)                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Step 1: Peak Detection                    Step 2: IBI Extraction                       │
│  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐          │
│  │     ∧       ∧       ∧           │       │  IBI = [720, 695, 710, 730] ms  │          │
│  │    / \\     / \\     / \\          │  ───► │                                 │          │
│  │ ──/   \\───/   \\───/   \\────    │       │  = time between peaks           │          │
│  │   P1      P2      P3            │       └─────────────────────────────────┘          │
│  └─────────────────────────────────┘                     │                              │
│                                                          ▼                              │
│  Step 3: HRV Metrics                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ • mean_ibi = mean(IBI)                    → Average heart rate level             │   │
│  │ • rmssd = sqrt(mean(diff(IBI)²))          → Beat-to-beat variability (vagal)     │   │
│  │ • sdnn = std(IBI)                         → Overall HRV                          │   │
│  │ • pnn50 = % of |diff(IBI)| > 50ms         → Parasympathetic activity             │   │
│  │ • n_peaks = count(peaks)                  → Heart rate (beats per window)        │   │
│  │ • hr_mean = 60000 / mean_ibi              → Heart rate in BPM                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         EDA FEATURES (Sympathetic Nervous System)                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Raw EDA = Tonic (baseline) + Phasic (responses)                                        │
│                                                                                         │
│  Statistical features from raw signal (eda_cc_*):                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ • eda_cc_mean      Mean skin conductance level                                   │   │
│  │ • eda_cc_std       Standard deviation (variability)                              │   │
│  │ • eda_cc_range     Max - Min (dynamic range)                    ← BEST PREDICTOR │   │
│  │ • eda_cc_slope     Linear trend over window                                      │   │
│  │ • eda_cc_rms       Root mean square                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  Phasic features (eda_phasic_*):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ • eda_phasic_max      Maximum phasic response amplitude                          │   │
│  │ • eda_phasic_energy   Sum of squared phasic values            ← GOOD PREDICTOR   │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         IMU FEATURES (Physical Movement)                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  3-axis accelerometer → Statistical features per window:                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ • acc_x_mean, acc_y_mean, acc_z_mean     Mean acceleration per axis              │   │
│  │ • acc_x_std, acc_y_std, acc_z_std        Movement intensity                      │   │
│  │ • acc_magnitude_mean                      Overall movement (sqrt(x²+y²+z²))      │   │
│  │ • acc_energy                              Total kinetic energy                    │   │
│  │ • acc_entropy                             Movement complexity                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.02, 0.92, features_text, fontsize=8.5, family='monospace', va='top', transform=ax.transAxes)

plt.savefig(f'{output_dir}/slide2_feature_extraction.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide2_feature_extraction.png")
plt.close()

#############################################################################
# SLIDE 3: FEATURE SELECTION
#############################################################################

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111)
ax.axis('off')

ax.text(0.5, 0.98, "Feature Selection: How We Choose Features", fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

selection_text = """
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE SELECTION PIPELINE                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

    Total features extracted: ~200+ features per modality

                    ┌─────────────────────────────┐
                    │    STEP 1: CORRELATION      │
                    │      WITH TARGET            │
                    │                             │
                    │  For each feature f:        │
                    │  r = corr(f, Borg)          │
                    │                             │
                    │  Keep top 100 by |r|        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    STEP 2: REDUNDANCY       │
                    │       PRUNING               │
                    │                             │
                    │  Within each modality:      │
                    │  If corr(f1, f2) > 0.90     │
                    │  Keep the one with higher   │
                    │  correlation to Borg        │
                    │                             │
                    │  Removes redundant features │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    FINAL: ~30-50 FEATURES   │
                    │                             │
                    │  EDA: ~10-15 features       │
                    │  PPG: ~10-15 features       │
                    │  IMU: ~10-15 features       │
                    └─────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              WHY THIS APPROACH?                                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ✓ Simple and interpretable (no black-box feature importance)                           │
│  ✓ Correlation is robust (no overfitting risk in selection)                             │
│  ✓ Redundancy pruning prevents multicollinearity                                        │
│  ✓ Keeps features from all modalities (balanced representation)                         │
│                                                                                         │
│  ✗ Assumes linear relationship (may miss non-linear patterns)                           │
│  ✗ Doesn't account for feature interactions                                             │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXAMPLE: TOP SELECTED FEATURES                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│    Feature              |  r with Borg  |  Interpretation                               │
│    ─────────────────────┼───────────────┼──────────────────────────────────────────     │
│    eda_cc_range         |  +0.60        |  EDA dynamic range ↑ with effort              │
│    ppg_green_n_peaks    |  +0.50        |  Heart rate ↑ with effort                     │
│    eda_phasic_energy    |  +0.48        |  Sympathetic activation ↑ with effort         │
│    ppg_green_mean_ibi   |  -0.46        |  IBI ↓ with effort (faster HR)                │
│    eda_phasic_max       |  +0.38        |  Peak stress response ↑ with effort           │
│    acc_magnitude_std    |  +0.31        |  Movement intensity ↑ with effort             │
│    ppg_green_rmssd      |  -0.25        |  HRV ↓ with effort (less variability)         │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.02, 0.92, selection_text, fontsize=9, family='monospace', va='top', transform=ax.transAxes)

plt.savefig(f'{output_dir}/slide3_feature_selection.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide3_feature_selection.png")
plt.close()

#############################################################################
# SLIDE 4: STATISTICAL VALIDATION
#############################################################################

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111)
ax.axis('off')

ax.text(0.5, 0.98, "Statistical Analysis: Why Correlation (Not ML)", fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

stats_text = """
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         THE PROBLEM: TEMPORAL AUTOCORRELATION                           │
└────────────────────────────────────────────────────────────────────────────────────────┘

   With 70% overlap, adjacent windows share most of their data:

   Window 1:  [─────────────────────────]
   Window 2:      [─────────────────────────]     ← 70% same data!
   Window 3:          [─────────────────────────]

   Result: EDA autocorrelation ≈ 1.0 (adjacent windows nearly identical)


┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         WHY MACHINE LEARNING "FAILS"                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Random Train/Test Split:                                                              │
│                                                                                         │
│   Time:     ─────────────────────────────────────────────────►                          │
│   Windows:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]                          │
│   Split:     T   V   T   T   V   T   V   T   T    V    T    V                           │
│                                                                                         │
│   Window 2 (validation) is IDENTICAL to Window 1 and 3 (training)                       │
│   → Model "predicts" by memorizing, not learning                                        │
│   → R² = 0.89 is FAKE (data leakage!)                                                   │
│                                                                                         │
│   Time-Series Cross-Validation:                                                         │
│   Train on past → Test on future                                                        │
│   → R² < 0 (model cannot generalize to future data)                                     │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         WHY CORRELATIONS ARE VALID                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Correlation = Pearson r between feature and Borg                                      │
│                                                                                         │
│   Key difference from ML:                                                               │
│   • No train/test split → No leakage possible                                           │
│   • Measures overall relationship in the data                                           │
│   • Does NOT claim to predict future values                                             │
│   • Reports what we OBSERVE, not what we can PREDICT                                    │
│                                                                                         │
│   Interpretation:                                                                       │
│   • r = +0.60 means "when EDA range is high, Borg tends to be high"                     │
│   • This is a valid observation, not an inflated prediction                             │
│   • p < 0.001 means this relationship is statistically significant                      │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         WHAT CAN WE CONCLUDE?                                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ✓ CAN say: "EDA and HRV correlate with perceived effort"                              │
│   ✓ CAN say: "These features track effort in the observed data"                         │
│   ✓ CAN say: "EDA is more consistent than HR across patients"                           │
│                                                                                         │
│   ✗ CANNOT say: "We can predict effort with R² = 0.89"                                  │
│   ✗ CANNOT say: "This model will work on new patients"                                  │
│                                                                                         │
│   Future work needed:                                                                   │
│   • Independent windows (no overlap) for proper ML validation                           │
│   • Leave-one-subject-out cross-validation                                              │
│   • Larger sample size (n > 2 patients)                                                 │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.02, 0.92, stats_text, fontsize=9, family='monospace', va='top', transform=ax.transAxes)

plt.savefig(f'{output_dir}/slide4_statistical_validation.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide4_statistical_validation.png")
plt.close()

#############################################################################
# PRINT SUMMARY FOR COPY/PASTE
#############################################################################

print("\n" + "="*80)
print("PRESENTATION CONTENT (for copy/paste)")
print("="*80)

print("""
================================================================================
                    PRESENTATION OUTLINE
================================================================================

SLIDE 1: PIPELINE OVERVIEW
─────────────────────────
1. Raw Data Acquisition
   • Wrist-worn wearable: PPG (32 Hz), EDA (32 Hz), IMU (32 Hz)
   • Self-reported Borg CR10 ratings during ADLs

2. Preprocessing
   • Linear interpolation for missing values
   • Uniform resampling to 32 Hz
   • (Optional) Highpass filter 0.5 Hz for PPG baseline drift

3. Windowing
   • Window size: 10 seconds
   • Overlap: 70%
   • Creates ~100-300 windows per subject

================================================================================

SLIDE 2: FEATURE EXTRACTION
───────────────────────────
PPG → HRV Features:
   • Peak detection → Inter-beat intervals (IBI)
   • mean_ibi: Average heart rate level
   • rmssd: Beat-to-beat variability (vagal tone marker)
   • sdnn: Overall HRV
   • n_peaks: Heart rate (beats per window)

EDA → Sympathetic Features:
   • eda_cc_range: Dynamic range of skin conductance
   • eda_cc_std: Variability
   • eda_phasic_energy: Phasic (response) component energy

IMU → Movement Features:
   • acc_magnitude_mean: Overall movement intensity
   • acc_x/y/z_std: Movement variability per axis

================================================================================

SLIDE 3: FEATURE SELECTION
──────────────────────────
Step 1: Correlation with Target
   • Compute r = corr(feature, Borg) for all features
   • Keep top 100 features by |r|

Step 2: Redundancy Pruning
   • Within each modality (EDA, PPG, IMU):
   • If corr(f1, f2) > 0.90, keep the one with higher |r| to Borg
   • Result: ~30-50 non-redundant features

Why this approach:
   ✓ Simple and interpretable
   ✓ No overfitting in selection
   ✓ Balanced representation from all modalities

================================================================================

SLIDE 4: WHY CORRELATION (NOT ML)
─────────────────────────────────
The Problem: Temporal Autocorrelation
   • 70% overlap → adjacent windows share most data
   • EDA autocorrelation ≈ 1.0

Why ML "Fails":
   • Random train/test split → data leakage
   • R² = 0.89 is FAKE (model memorizes, not learns)
   • Time-series CV shows R² < 0

Why Correlations are Valid:
   • No train/test split → no leakage
   • Reports what we OBSERVE, not predict
   • p < 0.001 confirms statistical significance

What We Can Conclude:
   ✓ EDA and HRV correlate with perceived effort
   ✗ Cannot claim predictive accuracy on new data

================================================================================

KEY FINDINGS SUMMARY
────────────────────
1. EDA is most robust predictor (r = 0.32-0.60 for both patients)
2. HRV variability (RMSSD, SDNN) works for both patients
3. Mean HR only works for elderly (severe has blunted HR response)
4. Wearable-based effort monitoring is feasible for cardiac patients
""")

print(f"\n✓ All slides saved to: {output_dir}/")
