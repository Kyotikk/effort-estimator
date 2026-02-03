#!/usr/bin/env python3
"""
COMPLETE PIPELINE BREAKDOWN
All steps explained in detail, with differences per method highlighted
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    COMPLETE PIPELINE - ALL STEPS EXPLAINED                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: RAW DATA COLLECTION                                                 │
│  ════════════════════════════                                                │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Wearable sensors record continuous data during activities
    • Each sensor streams at its own sampling rate
    • User provides Borg ratings at intervals (not continuous)
    
  DATA STREAMS:
    ┌────────────────┬──────────────┬────────────────────────────────┐
    │ Sensor         │ Sample Rate  │ What it captures               │
    ├────────────────┼──────────────┼────────────────────────────────┤
    │ PPG (Green)    │ ~25-100 Hz   │ Blood volume pulse (heart)     │
    │ PPG (Red)      │ ~25-100 Hz   │ Blood oxygenation              │
    │ PPG (Infrared) │ ~25-100 Hz   │ Deeper tissue perfusion        │
    │ EDA            │ ~4-32 Hz     │ Skin conductance (sweat)       │
    │ Accelerometer  │ ~25-100 Hz   │ Motion in X, Y, Z              │
    │ Gyroscope      │ ~25-100 Hz   │ Rotation in X, Y, Z            │
    │ Borg ratings   │ ~1 per min   │ Subjective effort (0-10)       │
    └────────────────┴──────────────┴────────────────────────────────┘

  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: PREPROCESSING (Per Sensor)                                          │
│  ══════════════════════════════════                                          │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Clean raw signals from noise and artifacts
    • Resample to consistent rate if needed
    • Apply sensor-specific filtering
    
  PPG PREPROCESSING:
    1. Bandpass filter (0.5-4 Hz) → Keep heart rate frequencies
    2. Remove motion artifacts → Discard when accelerometer shows movement
    3. Detect peaks → Find heartbeats for IBI/HR calculation
    4. Quality check → Flag segments with poor signal
    
  EDA PREPROCESSING:
    1. Low-pass filter (<5 Hz) → EDA is slow-changing
    2. Separate tonic (baseline) from phasic (responses)
    3. Detect skin conductance responses (SCRs)
    4. Quality check → Flag segments with artifacts
    
  IMU PREPROCESSING:
    1. High-pass filter → Remove gravity component (for dynamic accel)
    2. Calculate magnitude → sqrt(x² + y² + z²)
    3. Low-pass filter → Remove high-frequency noise
    
  OUTPUT:
    • Clean, filtered signals for each sensor
    • Quality flags per time segment
    • Timestamps aligned to common reference
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: QUALITY CHECKS                                                      │
│  ═══════════════════════                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Identify and flag bad data segments
    • Decide: remove, interpolate, or keep with warning
    
  QUALITY CRITERIA:
    ┌────────────────┬────────────────────────────────────────────────┐
    │ Sensor         │ Quality Checks                                 │
    ├────────────────┼────────────────────────────────────────────────┤
    │ PPG            │ - Signal-to-noise ratio (SNR)                  │
    │                │ - Peak detection success rate                  │
    │                │ - Heart rate in physiological range (30-200)   │
    │                │ - Motion artifact detection                    │
    ├────────────────┼────────────────────────────────────────────────┤
    │ EDA            │ - Value in valid range (0.05-40 µS)            │
    │                │ - No sudden jumps (sensor disconnect)          │
    │                │ - Tonic level stability                        │
    ├────────────────┼────────────────────────────────────────────────┤
    │ IMU            │ - No saturation (sensor maxed out)             │
    │                │ - Reasonable dynamic range                     │
    │                │ - Consistent sampling rate                     │
    └────────────────┴────────────────────────────────────────────────┘
    
  DECISIONS:
    • Good quality (>80% valid) → Keep segment
    • Medium quality (50-80%) → Keep with flag
    • Poor quality (<50%) → Remove from analysis
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: TIME ALIGNMENT                                                      │
│  ═══════════════════════                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Synchronize all sensor streams to same time base
    • Handle different sampling rates
    • Align Borg ratings to sensor data
    
  THE PROBLEM:
    • PPG recorded at 64 Hz, EDA at 4 Hz, IMU at 50 Hz
    • Borg ratings given every ~60 seconds
    • Sensors may have clock drift
    
  THE SOLUTION:
    1. Convert all timestamps to Unix time (common reference)
    2. Resample to common rate OR keep native and align at window level
    3. For Borg: assign to time range (e.g., rating covers previous 30s)
    
  BORG ALIGNMENT STRATEGIES:
    ┌─────────────────┬───────────────────────────────────────────────┐
    │ Strategy        │ Description                                   │
    ├─────────────────┼───────────────────────────────────────────────┤
    │ Point-in-time   │ Borg applies only to moment of rating         │
    │ Backward window │ Borg applies to X seconds BEFORE rating       │
    │ Forward window  │ Borg applies to X seconds AFTER rating        │
    │ Centered window │ Borg applies to X seconds around rating       │
    └─────────────────┴───────────────────────────────────────────────┘
    
  OUR APPROACH:
    • Borg rating applies to sensor data around that timestamp
    • 5-second windows centered on labeled time points
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: WINDOWING (Segmentation)                                            │
│  ════════════════════════════════                                            │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Divide continuous signals into fixed-length segments
    • Each window becomes one "sample" for ML
    
  PARAMETERS:
    • Window size: 5.0 seconds
    • Overlap: 70% (step size = 1.5 seconds)
    
  WHY THESE SETTINGS?
    • 5 seconds: Long enough to capture physiological cycles
      - Heart rate needs ~3-5 beats for stable estimate
      - EDA responses have ~1-5 second latency
    • 70% overlap: More samples, smoother transitions
      - With 584 samples, we need the data augmentation
      
  VISUALIZATION:
    
    Time: 0s      5s      10s     15s     20s
          |-------|-------|-------|-------|
    
    Window 1: [=====]                         (0-5s)
    Window 2:    [=====]                      (1.5-6.5s)
    Window 3:       [=====]                   (3-8s)
    Window 4:          [=====]                (4.5-9.5s)
    ...
    
  OUTPUT:
    • N windows, each with 5 seconds of multi-sensor data
    • Each window has an associated Borg label (if labeled)
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: FEATURE EXTRACTION (Per Window)                                     │
│  ═══════════════════════════════════════                                     │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Convert 5 seconds of raw signals → single feature vector
    • Extract statistical, frequency, and domain-specific features
    
  PPG FEATURES (Heart-related):
    ┌────────────────────────┬────────────────────────────────────────┐
    │ Feature Type           │ Examples                               │
    ├────────────────────────┼────────────────────────────────────────┤
    │ Heart Rate             │ hr_mean, hr_std, hr_min, hr_max        │
    │ Heart Rate Variability │ rmssd, sdnn, pnn50 (from IBI)          │
    │ Signal Stats           │ mean, std, skewness, kurtosis          │
    │ Frequency Domain       │ power in LF, HF bands                  │
    │ Waveform               │ pulse width, amplitude, shape          │
    └────────────────────────┴────────────────────────────────────────┘
    
  EDA FEATURES (Stress/arousal-related):
    ┌────────────────────────┬────────────────────────────────────────┐
    │ Feature Type           │ Examples                               │
    ├────────────────────────┼────────────────────────────────────────┤
    │ Tonic (baseline)       │ scl_mean, tonic_mean, tonic_std        │
    │ Phasic (responses)     │ scr_count, scr_amplitude, scr_rate     │
    │ Signal Stats           │ mean, std, range, slope                │
    │ Derived                │ phasic_energy, stress_index            │
    └────────────────────────┴────────────────────────────────────────┘
    
  IMU FEATURES (Motion-related):
    ┌────────────────────────┬────────────────────────────────────────┐
    │ Feature Type           │ Examples                               │
    ├────────────────────────┼────────────────────────────────────────┤
    │ Statistical            │ mean, std, min, max, range             │
    │ Frequency              │ dominant_freq, spectral_entropy        │
    │ Activity               │ magnitude, jerk, step_count            │
    │ Complexity             │ sample_entropy, fractal_dimension      │
    └────────────────────────┴────────────────────────────────────────┘
    
  TOTAL: 284 features per window
  
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: FEATURE-LEVEL QUALITY CHECK                                         │
│  ═══════════════════════════════════                                         │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Check extracted features for validity
    • Remove windows with too many bad features
    • Handle missing values
    
  CHECKS:
    1. NaN/Inf values → Remove or impute
    2. Constant features → Remove (no information)
    3. Outlier features → Cap or flag
    4. Feature coverage → Require >50% valid features per window
    
  MISSING VALUE HANDLING:
    • Per window: If >50% features missing → Remove window
    • Per feature: If >50% windows missing → Remove feature
    • Remaining NaNs: Impute with column mean
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: SENSOR FUSION                                                       │
│  ════════════════════                                                        │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Combine features from all sensors into single feature vector
    • Handle cases where some sensors have missing data
    
  FUSION APPROACH (Feature-level, Early Fusion):
    
    PPG Features:    [hr_mean, hr_std, rmssd, ...]     → 183 features
    EDA Features:    [scl_mean, scr_count, ...]        →  47 features
    IMU Features:    [acc_mean, acc_std, ...]          →  60 features
                                                        ──────────────
    Combined:        [all concatenated]                → 284 features
    
  WHY EARLY FUSION?
    • Simple and effective
    • Model can learn cross-sensor relationships
    • Works well with enough data
    
  ALTERNATIVE: Late fusion (separate models per sensor, combine predictions)
    • Not used here - would need more data
    
  SAME FOR ALL METHODS? ✅ YES


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: FEATURE SELECTION                                                   │
│  ═════════════════════════                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

  WHAT HAPPENS:
    • Remove redundant or uninformative features
    • Reduce dimensionality to prevent overfitting
    
  SELECTION CRITERIA:
    1. Variance threshold → Remove near-constant features
    2. Correlation filter → Remove highly correlated pairs (keep one)
    3. Missing data → Remove features with >50% NaN
    
  IN OUR PIPELINE:
    • Started with 284 features
    • After cleaning: 284 remain (minimal removal)
    • No aggressive selection (Ridge handles regularization)
    
  SAME FOR ALL METHODS? ✅ YES


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        ⚠️  FROM HERE, METHODS START TO DIFFER! ⚠️                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 10: FEATURE NORMALIZATION (SCALING)                                    │
│  ════════════════════════════════════════                                    │
│                                                                              │
│  ⚠️  THIS IS WHERE METHODS 1 & 4 DIFFER FROM METHODS 2 & 3                   │
└──────────────────────────────────────────────────────────────────────────────┘

  WHY NORMALIZE?
    • Features have different scales (HR ~60-180, EDA ~0.1-40)
    • ML models work better with standardized inputs
    • Prevents large-scale features from dominating
    
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHOD 1: GLOBAL Z-SCORE (Cross-subject raw)                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Formula: z = (x - μ_all) / σ_all                                      │
  │                                                                         │
  │   Where μ_all and σ_all are computed across ALL subjects together       │
  │                                                                         │
  │   Example:                                                              │
  │     All subjects' HR mean = 85, std = 15                                │
  │     P1's HR of 100 → z = (100 - 85) / 15 = +1.0                         │
  │     P2's HR of 100 → z = (100 - 85) / 15 = +1.0  (same!)                │
  │                                                                         │
  │   PROBLEM: Doesn't account for individual baselines                     │
  │   P1 might have resting HR=60, P2 might have resting HR=90              │
  │   Same z-score, but very different effort levels!                       │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHODS 2, 3, 4: PER-SUBJECT Z-SCORE (Personalized)                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Formula: z = (x - μ_person) / σ_person                                │
  │                                                                         │
  │   Where μ_person and σ_person are computed for EACH subject separately  │
  │                                                                         │
  │   Example:                                                              │
  │     P1's HR: mean=70, std=10 → HR of 100 → z = (100-70)/10 = +3.0       │
  │     P2's HR: mean=90, std=10 → HR of 100 → z = (100-90)/10 = +1.0       │
  │                                                                         │
  │   Now the z-scores MEAN something:                                      │
  │     P1 at HR=100 is 3σ above THEIR baseline (high effort!)              │
  │     P2 at HR=100 is 1σ above THEIR baseline (moderate effort)           │
  └─────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 11: TARGET (BORG) HANDLING                                             │
│  ═══════════════════════════════                                             │
│                                                                              │
│  ⚠️  THIS IS WHERE METHOD 3 DIFFERS FROM METHODS 1, 2, 4                     │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHODS 1, 2, 4: RAW BORG (Absolute)                                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Target: y = Borg rating (0-10)                                        │
  │                                                                         │
  │   Model predicts absolute Borg directly                                 │
  │                                                                         │
  │   Example predictions:                                                  │
  │     Input features → Output: "Borg = 4.5"                               │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHOD 3: NORMALIZED BORG (Relative)                                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Target: z_borg = (borg - μ_person_borg) / σ_person_borg               │
  │                                                                         │
  │   Model predicts RELATIVE Borg (deviation from person's average)        │
  │                                                                         │
  │   Example:                                                              │
  │     P1's Borg: mean=3.0, std=1.5                                        │
  │     Actual Borg=5 → z_borg = (5-3)/1.5 = +1.33                          │
  │                                                                         │
  │   Model predicts: z_borg = +1.33                                        │
  │   Meaning: "1.33 standard deviations above this person's average"       │
  │                                                                         │
  │   To get actual Borg: 1.33 × 1.5 + 3.0 = 5.0                            │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 12: TRAIN/TEST SPLIT (Cross-Validation Strategy)                       │
│  ═════════════════════════════════════════════════════                       │
│                                                                              │
│  ⚠️  THIS IS WHERE METHOD 4 DIFFERS FROM METHODS 1, 2, 3                     │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHODS 1, 2, 3: LEAVE-ONE-SUBJECT-OUT (LOSO)                           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Round 1: Train on P1,P2,P3,P4 → Test on P5                            │
  │   Round 2: Train on P1,P2,P3,P5 → Test on P4                            │
  │   Round 3: Train on P1,P2,P4,P5 → Test on P3                            │
  │   Round 4: Train on P1,P3,P4,P5 → Test on P2                            │
  │   Round 5: Train on P2,P3,P4,P5 → Test on P1                            │
  │                                                                         │
  │   PURPOSE: Test generalization to UNSEEN subjects                       │
  │   SIMULATES: Deploying model to new user                                │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHOD 4: WITHIN-SUBJECT 5-FOLD CV                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   For EACH subject separately:                                          │
  │     Fold 1: Train on 80% of P1's data → Test on 20%                     │
  │     Fold 2: Train on different 80% → Test on different 20%              │
  │     ... (5 folds total)                                                 │
  │                                                                         │
  │   Repeat for P2, P3, P4, P5                                             │
  │                                                                         │
  │   PURPOSE: Test prediction for SAME subject                             │
  │   SIMULATES: Having training data from that specific person             │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 13: MODEL TRAINING                                                     │
│  ═══════════════════════                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

  MODEL: Ridge Regression (α = 1.0)
  
  WHY RIDGE?
    • Linear model with L2 regularization
    • Handles many features (284) with few samples (584)
    • Prevents overfitting better than plain linear regression
    • More stable than XGBoost on small data
    
  TRAINING:
    • Input: X = normalized features (N × 284)
    • Output: y = Borg (or z_borg for Method 3)
    • Learns: weights for each feature
    
  SAME FOR ALL METHODS? ✅ YES (same model, same hyperparameters)


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 14: PREDICTION & DENORMALIZATION                                       │
│  ═════════════════════════════════════                                       │
│                                                                              │
│  ⚠️  THIS IS WHERE METHOD 3 HAS AN EXTRA STEP                                │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHODS 1, 2, 4: DIRECT PREDICTION                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Model output = Final Borg prediction                                  │
  │                                                                         │
  │   Input features → Model → Predicted Borg (0-10)                        │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ METHOD 3: DENORMALIZATION REQUIRED                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │   Model output = z_borg_predicted (relative)                            │
  │                                                                         │
  │   Must convert back to absolute Borg:                                   │
  │     Borg_predicted = z_pred × σ_person_borg + μ_person_borg             │
  │                                                                         │
  │   Example:                                                              │
  │     Model predicts: z_borg = +1.5                                       │
  │     Person's stats: μ=3.0, σ=1.5                                        │
  │     Final Borg: 1.5 × 1.5 + 3.0 = 5.25                                  │
  │                                                                         │
  │   ⚠️ REQUIRES knowing the person's μ and σ from calibration!            │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 15: EVALUATION                                                         │
│  ═══════════════════                                                         │
└──────────────────────────────────────────────────────────────────────────────┘

  METRICS:
    • Pearson r: Does prediction track truth? (pattern)
    • MAE: How far off are values? (magnitude)
    • Exact category: % correct LOW/MOD/HIGH
    • Big miss rate: % confusing LOW ↔ HIGH (dangerous errors)
    
  SAME FOR ALL METHODS? ✅ YES (same evaluation)


╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    SUMMARY: WHERE METHODS DIFFER                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────┬───────────┬───────────┬───────────┬───────────┐
│ Pipeline Step       │ Method 1  │ Method 2  │ Method 3  │ Method 4  │
│                     │ Raw Cross │ Feat Norm │ Both Norm │ Within    │
├─────────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 1. Raw Data         │ SAME      │ SAME      │ SAME      │ SAME      │
│ 2. Preprocessing    │ SAME      │ SAME      │ SAME      │ SAME      │
│ 3. Quality Check    │ SAME      │ SAME      │ SAME      │ SAME      │
│ 4. Time Alignment   │ SAME      │ SAME      │ SAME      │ SAME      │
│ 5. Windowing        │ SAME      │ SAME      │ SAME      │ SAME      │
│ 6. Feature Extract  │ SAME      │ SAME      │ SAME      │ SAME      │
│ 7. Feature QC       │ SAME      │ SAME      │ SAME      │ SAME      │
│ 8. Sensor Fusion    │ SAME      │ SAME      │ SAME      │ SAME      │
│ 9. Feature Select   │ SAME      │ SAME      │ SAME      │ SAME      │
├─────────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 10. Feature Norm    │ GLOBAL    │ PER-SUBJ  │ PER-SUBJ  │ PER-SUBJ  │
│ 11. Target (Borg)   │ RAW       │ RAW       │ PER-SUBJ  │ RAW       │
│ 12. CV Strategy     │ LOSO      │ LOSO      │ LOSO      │ 5-FOLD    │
│ 13. Model Training  │ SAME      │ SAME      │ SAME      │ SAME      │
│ 14. Denormalize     │ NO        │ NO        │ YES       │ NO        │
├─────────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 15. Evaluation      │ SAME      │ SAME      │ SAME      │ SAME      │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┘

LEGEND:
  GLOBAL    = Computed across all subjects pooled
  PER-SUBJ  = Computed separately for each subject
  LOSO      = Leave-One-Subject-Out cross-validation
  5-FOLD    = 5-fold CV within each subject separately


KEY DIFFERENCES SUMMARY:
────────────────────────────────────────────────────────────────────────────────

  Method 1 vs 2/3:
    Feature normalization changes from GLOBAL to PER-SUBJECT
    
  Method 3 vs 1/2/4:
    Borg target also normalized per-subject (then denormalized)
    
  Method 4 vs 1/2/3:
    Different CV strategy (within-subject instead of cross-subject)

""")
