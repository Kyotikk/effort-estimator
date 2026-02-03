#!/usr/bin/env python3
"""
EFFORT ESTIMATION PIPELINE - CURRENT STATE & RECOMMENDATIONS
============================================================

Based on our analysis, here's how the pipeline works and what's optimal.
"""

print("""
================================================================================
EFFORT ESTIMATION PIPELINE OVERVIEW
================================================================================

CURRENT PIPELINE STRUCTURE:
──────────────────────────────────────────────────────────────────────────────

1. PREPROCESSING (preprocessing/*.py)
   ├── ppg.py      → Clean PPG signal, detect peaks, extract HR
   ├── eda.py      → Clean EDA signal, decompose (tonic/phasic)
   ├── imu.py      → Clean accelerometer data
   ├── ecg.py      → Process ECG (if available)
   └── rr.py       → Process RR intervals

2. WINDOWING (windowing/windows.py)
   └── Split continuous signal into 5-second windows

3. FEATURE EXTRACTION (features/*.py)
   ├── ppg_features.py   → 183 PPG features (HR, HRV, signal stats)
   ├── eda_features.py   → 47 EDA features (SCL, SCR, phasic)
   ├── manual_features_imu.py → 60 IMU features (movement intensity)
   └── Outputs: windowed features CSV per modality

4. FUSION & ALIGNMENT (ml/fusion/, ml/alignment.py)
   └── Merge all modality features + align with Borg timestamps
   └── Output: fused_aligned_5.0s.csv

5. FEATURE SELECTION (ml/feature_selection_and_qc.py)  ⚠️ PROBLEMATIC
   └── OLD: Selects top 100 features by pooled correlation
   └── ISSUE: Data leakage - sees test subject data

6. TRAINING (train_multisub_xgboost.py)
   └── OLD: Ridge regression with selected features
   └── ISSUE: Not optimal model/feature combo

================================================================================
WHAT WE LEARNED
================================================================================

BEST APPROACH (per-subject r = 0.57):
─────────────────────────────────────
✅ Model:      RandomForest (n_estimators=100, max_depth=6)
✅ Features:   IMU only (60 features) - movement intensity works best
✅ Evaluation: LOSO with 20-30% random calibration per test subject
✅ Metric:     Per-subject r (NOT pooled r which is misleading)

WHAT DOESN'T HELP:
──────────────────
❌ Adding PPG features (183) → reduces r from 0.57 to ~0.30
❌ Adding EDA features (47) → reduces r 
❌ All features (293) → overfits, poor generalization
❌ HR reserve normalization → no improvement
❌ Pre-filtering features → model does better choosing itself

WHY IMU WORKS BEST:
───────────────────
• Movement intensity directly reflects physical effort
• Less inter-subject variability than HR
• More robust signal (no PPG artifacts)
• 60 features = good balance (not too few, not overfitting)

================================================================================
RECOMMENDED PIPELINE CHANGES
================================================================================

OPTION 1: MINIMAL CHANGE (Quick fix)
────────────────────────────────────
Just change training to use IMU features + RandomForest:

    # In train_multisub_xgboost.py or new script:
    imu_features = [c for c in df.columns if 'acc' in c.lower()]
    model = RandomForestRegressor(n_estimators=100, max_depth=6)
    # Use LOSO with 20% calibration

OPTION 2: PROPER INTEGRATION (Recommended)
──────────────────────────────────────────
Replace feature_selection_and_qc.py with loso_feature_selection.py:

    from ml.loso_feature_selection import get_loso_features
    
    for test_subject in subjects:
        # Features selected using ONLY training data
        feature_cols = get_loso_features(df, test_subject, strategy='imu')
        # ... train and evaluate

================================================================================
PIPELINE FILES TO USE
================================================================================

KEEP (working well):
├── preprocessing/*.py      - Signal preprocessing ✓
├── windowing/windows.py    - Windowing ✓
├── features/*.py           - Feature extraction ✓
├── ml/fusion/              - Data fusion ✓
├── ml/alignment.py         - Borg alignment ✓

REPLACE/UPDATE:
├── ml/feature_selection_and_qc.py  → Use ml/loso_feature_selection.py
├── train_multisub_xgboost.py       → Use ml/train_improved.py

NEW FILES CREATED:
├── ml/best_feature_selection.py    - Data-driven feature selection
├── ml/loso_feature_selection.py    - LOSO-aware feature selection  
├── ml/train_improved.py            - Improved training with RF + IMU
├── compare_approaches.py           - Analysis script
├── ml_expert_approach.py           - Comprehensive model comparison

================================================================================
HOW TO RUN THE IMPROVED PIPELINE
================================================================================

# Step 1: Run preprocessing + windowing + feature extraction (unchanged)
python run_pipeline.py <subject_dir>

# Step 2: Run improved training (NEW)
python ml/train_improved.py <path_to_fused_aligned.csv> <output_dir>

# Or for quick test:
python compare_approaches.py  # Shows all model comparisons

================================================================================
EXPECTED RESULTS
================================================================================

With 5 elderly subjects:
• Per-subject r = 0.55-0.58 (IMU + RF + 20% calibration)
• ±1 Borg accuracy = 55-60%
• MAE = ~1.0 Borg point

With 15-20 subjects (projected):
• Per-subject r = 0.70-0.80
• More stable generalization

================================================================================
""")
