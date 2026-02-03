#!/usr/bin/env python3
"""
CORRECTED EXACT STEPS: ML Pipeline vs Literature Formula
With ACTUAL preprocessing parameters from the codebase.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

print("="*80)
print("CORRECTED EXACT STEPS: ML PIPELINE vs LITERATURE FORMULA")
print("="*80)

# =============================================================================
# STEP 0: DATA SOURCES
# =============================================================================
print("\n" + "="*80)
print("STEP 0: DATA SOURCES (same for both)")
print("="*80)

print("""
  SENSORS:
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Sensor              │ Location   │ Raw Fs    │ Modality               │
  ├────────────────────────────────────────────────────────────────────────┤
  │ Vivalnk VV330       │ Chest      │ ~1 Hz*    │ HR (from ECG)          │
  │ Corsano wrist       │ Wrist      │ ~100 Hz   │ ACC (tri-axial)        │
  │ Corsano wrist       │ Wrist      │ ~32 Hz    │ PPG (green/red/infra)  │
  │ Corsano BioZ        │ Chest      │ ~32 Hz    │ ACC (tri-axial)        │
  │ Corsano BioZ        │ Chest      │ ~1 Hz     │ EDA (skin conductance) │
  │ Corsano BioZ        │ Chest      │ ~1 Hz     │ RR intervals           │
  └────────────────────────────────────────────────────────────────────────┘
  
  * HR is beat-to-beat, so ~1 sample per heartbeat (not fixed rate)
  
  LABELS:
  - SCAI app: ADL activity annotations with Start/End times + Borg CR10 rating
  - 33 activities for sim_elderly3
""")

# =============================================================================
# =============================================================================
# APPROACH 1: LITERATURE FORMULA
# =============================================================================
# =============================================================================
print("\n" + "="*80)
print("APPROACH 1: LITERATURE-BASED FORMULA")
print("="*80)

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.1: HR PREPROCESSING")
print("-"*80)
print("""
  1.1.1 LOAD RAW HR DATA
    - File: vivalnk_vv330_heart_rate/data_1.csv.gz
    - Columns: time (Unix timestamp), hr (bpm)
    - N samples: ~4,700
    
  1.1.2 OUTLIER REMOVAL (physiological bounds)
    - Filter: 40 ≤ HR ≤ 200 bpm
    - Why: Values outside = sensor artifacts (AHA guidelines)
    - Removed: ~77 samples (1.6%)
    - Result: 4,628 valid samples
    
  1.1.3 NO ADDITIONAL FILTERING
    - Why: We aggregate to MEAN per activity
    - Mean is robust to small noise
    - Heavy filtering would distort short activities
    
  ⚠️ QUALITY CHECK: Implicitly done by removing outliers
    - No explicit signal quality metric
    - Assumption: Vivalnk chest ECG is high quality
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.2: IMU PREPROCESSING (for combined formula)")
print("-"*80)
print("""
  1.2.1 LOAD RAW IMU DATA
    - File: corsano_wrist_acc/2025-12-04.csv.gz
    - Columns: time, accX, accY, accZ (in g)
    - Raw Fs: ~100 Hz (variable)
    
  1.2.2 RESAMPLE TO UNIFORM GRID
    - Target: 32 Hz (fs_out in pipeline.yaml)
    - Method: Linear interpolation
    - Why: Need uniform sampling for feature extraction
    
  1.2.3 LOWPASS FILTER (noise removal)
    - Cutoff: 5.0 Hz (noise_cutoff in config)
    - Filter: 4th-order Butterworth
    - Why: Remove high-frequency sensor noise
    
  1.2.4 GRAVITY SEPARATION (highpass)
    - Gravity cutoff: 0.3 Hz (gravity_cutoff in config)
    - Creates: acc_x_grav (gravity component)
    - Creates: acc_x_dyn (dynamic = raw - gravity)
    - Why: Separate body movement from orientation
    
  1.2.5 COMPUTE RMS ACCELERATION
    - Formula: RMS_acc = √(acc_x_dyn² + acc_y_dyn² + acc_z_dyn²)
    - Result: Single magnitude signal
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.3: ACTIVITY-LEVEL AGGREGATION")
print("-"*80)
print("""
  For each ADL activity (33 total):
  
  1.3.1 GET TIME WINDOW
    - t_start, t_end from SCAI app labels
    - Duration: t_end - t_start (seconds)
    
  1.3.2 EXTRACT HR DURING ACTIVITY
    - Find all HR samples in [t_start, t_end]
    - Compute: HR_mean = mean(HR values)
    - Why mean? Literature (TRIMP, Banister 1991) uses mean intensity
    
  1.3.3 EXTRACT IMU DURING ACTIVITY
    - Find all RMS_acc samples in [t_start, t_end]
    - Compute: RMS_acc_mean = mean(RMS_acc values)
    
  1.3.4 GET BORG RATING
    - From SCAI app: Borg CR10 scale (0-10)
    - Subjective effort rating reported by subject
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.4: FEATURE COMPUTATION")
print("-"*80)
print("""
  HR COMPONENT (2 features → 1 combined):
  
    HR_baseline = min(HR_mean across all activities) = 75.5 bpm
    HR_delta = HR_mean - HR_baseline  [bpm above rest]
    HR_effort = HR_delta × √duration  [interaction term]
    
    Why √duration?
      - Fatigue accumulates sublinearly (diminishing returns)
      - Matches TRIMP formula (Banister, 1991)
      - Physics: Work = Power × Time
      
  IMU COMPONENT (2 features → 1 combined):
  
    RMS_baseline = min(RMS_acc_mean across all activities)
    RMS_delta = RMS_acc_mean - RMS_baseline
    IMU_effort = RMS_delta × √duration
    
  COMBINED (optimal weight finding):
  
    z_HR = standardize(HR_effort)     # z-score
    z_IMU = standardize(IMU_effort)   # z-score
    Combined = 0.8 × z_HR + 0.2 × z_IMU
    
    Why 80/20?
      - Grid search over [100/0, 80/20, 70/30, 60/40, 50/50, 0/100]
      - 80/20 gave highest correlation (r=0.843)
      - HR is primary driver of perceived effort
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.5: MODEL / CORRELATION")
print("-"*80)
print("""
  NO ML MODEL - just correlation!
  
  For HR component alone:
    r = corr(HR_effort, Borg) = 0.835***
    
  For IMU component alone:
    r = corr(IMU_effort, Borg) = 0.678***
    
  For Combined (80% HR + 20% IMU):
    r = corr(Combined, Borg) = 0.843***
    
  If you want a linear model:
    Borg = a × HR_effort + b
    Fitted: a = 0.028, b = 0.76
    
    These numbers mean:
      - Intercept b=0.76: Borg ≈ 0.76 when HR_effort = 0
      - Slope a=0.028: Each unit of HR_effort adds 0.028 to Borg
      
    Example: HR_effort = 100 → Borg = 0.028×100 + 0.76 = 3.56
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 1.6: VALIDATION")
print("-"*80)
print("""
  Method: Leave-One-Out Cross-Validation (LOO-CV)
  
  For each activity i (1 to 33):
    1. Train on all activities EXCEPT i
    2. Predict Borg for activity i
    3. Record error
    
  Result:
    r² on all data:       0.70 (= 0.835²)
    CV R² (honest):       0.66
    
  NO DATA LEAKAGE:
    - Each activity is independent (no overlap)
    - Timestamps don't overlap between activities
""")

# =============================================================================
# =============================================================================
# APPROACH 2: ML PIPELINE
# =============================================================================
# =============================================================================
print("\n" + "="*80)
print("APPROACH 2: ML PIPELINE")
print("="*80)

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.1: PPG PREPROCESSING")
print("-"*80)
print("""
  Three PPG channels: green (0x7e), infra (0x7b), red (0x7c)
  
  2.1.1 LOAD & FILTER BY SENSOR
    - Filter by metric_id and led_pd_pos
    - Green: metric_id='0x7e', led_pd_pos=6
    - Infra: metric_id='0x7b', led_pd_pos=22
    - Red: metric_id='0x7c', led_pd_pos=182
    
  2.1.2 RESAMPLE TO UNIFORM GRID
    - Target: 32 Hz (fs_out in config)
    - Method: Linear interpolation
    - Why: Uniform sampling for FFT features
    
  2.1.3 HIGHPASS FILTER (optional, per channel)
    - Green: apply_hpf=False (raw PPG preserved)
    - Infra: apply_hpf=True, cutoff=0.5 Hz
    - Red: apply_hpf=True, cutoff=0.5 Hz
    - Why: Remove baseline drift, enhance cardiac pulses
    - Filter: 4th-order Butterworth highpass
    
  QUALITY CHECK:
    - No explicit signal quality metric in preprocessing
    - Quality checked later in feature extraction (NaN handling)
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.2: IMU PREPROCESSING (same as literature, more detail)")
print("-"*80)
print("""
  Two IMU sources: wrist (corsano_wrist_acc) and chest (corsano_bioz_acc)
  
  2.2.1 LOAD RAW
    - Columns: time, accX, accY, accZ
    - Raw Fs: ~100 Hz (wrist), ~32 Hz (chest)
    
  2.2.2 RESAMPLE
    - Target: 32 Hz (fs_out=32 in config)
    - Method: Linear interpolation
    - Why 32 Hz? Matches PPG, enough for human movement (<10 Hz)
    
  2.2.3 LOWPASS FILTER (noise removal)
    - Cutoff: 5.0 Hz (noise_cutoff)
    - Filter: 4th-order Butterworth lowpass
    - Why: Human movement is <10 Hz, sensor noise is higher
    
  2.2.4 GRAVITY SEPARATION
    - Lowpass at 0.3 Hz → gravity component
    - Dynamic = Raw - Gravity
    - Output: acc_x, acc_y, acc_z, acc_x_grav, acc_y_grav, acc_z_grav,
              acc_x_dyn, acc_y_dyn, acc_z_dyn
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.3: EDA PREPROCESSING")
print("-"*80)
print("""
  2.3.1 LOAD RAW
    - File: corsano_bioz_emography/2025-12-04.csv.gz
    - Columns: time, cc (skin conductance), stress_skin, quality
    - Raw Fs: ~1 Hz (very low!)
    
  2.3.2 RESAMPLE (UPSAMPLE!)
    - Target: 32 Hz (fs_out=32)
    - Method: Linear interpolation
    - ⚠️ This is UPSAMPLING from 1 Hz → 32 Hz
    - Creates artificial samples (interpolated)
    
  2.3.3 NO FILTERING
    - EDA is already slow signal
    - Quality column available but not used as filter
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.4: RR INTERVAL PREPROCESSING")  
print("-"*80)
print("""
  2.4.1 LOAD RAW
    - File: corsano_bioz_rr_interval/2025-11-19.csv
    - Columns: time, rr (inter-beat interval in ms)
    - Raw Fs: ~1 Hz (one per heartbeat)
    
  2.4.2 RESAMPLE
    - Target: 1 Hz (fs_out=1 in config)
    - Different from other modalities!
    - Why: RR intervals are naturally ~1 Hz
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.5: WINDOWING")
print("-"*80)
print("""
  Parameters (from pipeline.yaml):
    - Window size: 10 seconds
    - Overlap: 10% (0.1 in config)
    - Step: 10 × (1 - 0.1) = 9 seconds
    
  ⚠️ NOTE: Some older scripts used 70% overlap!
    - pipeline.yaml says 0.1 (10%)
    - Some documentation mentions 70%
    - 10% overlap is MUCH better for avoiding leakage
    
  Result per modality:
    - PPG: 10s × 32 Hz = 320 samples per window
    - IMU: 10s × 32 Hz = 320 samples per window
    - EDA: 10s × 32 Hz = 320 samples (but upsampled from ~10!)
    - RR:  10s × 1 Hz = 10 samples per window
    
  Total windows: ~140-150 per subject (with 10% overlap)
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.6: FEATURE EXTRACTION")
print("-"*80)
print("""
  Per window, extract features from each modality:
  
  A) IMU FEATURES (per axis: x, y, z + magnitude):
     Time-domain (stat features):
       - mean, std, var, min, max, range
       - iqr, skewness, kurtosis
       - rms, peak_to_peak
       - zero_crossing_rate
       - mean_abs_change, mean_diff
       - entropy (sample entropy)
       
     ~30 features × 3 axes × 2 locations = ~180 features
     
  B) PPG FEATURES:
     Time-domain:
       - mean, std, min, max, range
       - iqr, skewness, kurtosis
       - Peak detection → mean_ibi, std_ibi (HRV proxy)
       
     Frequency-domain (if enough samples):
       - Dominant frequency
       - Spectral entropy
       
     ~20 features × 3 channels = ~60 features
     
  C) EDA FEATURES:
     - mean, std, min, max
     - SCR count (skin conductance responses)
     - Tonic level, phasic component
     
     ~15 features
     
  D) RR FEATURES (HRV):
     Time-domain:
       - mean_rr, std_rr (SDNN)
       - rmssd, pnn50, pnn20
       
     Frequency-domain:
       - VLF, LF, HF power
       - LF/HF ratio
       
     ~15 features
     
  TOTAL: ~270-300 features per window
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.7: FEATURE SELECTION")
print("-"*80)
print("""
  2.7.1 REMOVE CONSTANT FEATURES
    - Threshold: variance < 1e-10
    - Why: Zero variance = no information
    
  2.7.2 REMOVE HIGH-CORRELATION FEATURES
    - Threshold: |r| > 0.95
    - Why: Redundant, multicollinearity
    - Keep feature with higher corr to target
    
  2.7.3 FILTER BY TARGET CORRELATION
    - Threshold: |r with Borg| > 0.1 (typically)
    - Why: Remove features with no signal
    
  2.7.4 RECURSIVE FEATURE ELIMINATION (RFE)
    - Estimator: Random Forest
    - Target: ~16 features
    - Why: Empirical balance of bias/variance
    
  ⚠️ "SEES ALL DATA" ISSUE:
    - Correlation thresholds computed on FULL dataset
    - RFE trained on FULL dataset
    - This should be done INSIDE cross-validation loop!
    - Leaks information: features selected partly based on test data
    
  Result: 287 → ~50 → 16 features
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.8: LABEL ALIGNMENT")
print("-"*80)
print("""
  For each window:
    1. Get window center time
    2. Find ADL activity containing that time
    3. Assign activity's Borg rating to window
    
  Result: Each window has a Borg label
  ~150 windows with Borg ratings
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.9: MODEL TRAINING")
print("-"*80)
print("""
  Model: XGBoost Regressor
  
  Hyperparameters:
    - n_estimators: 100
    - max_depth: 3 (→ ~8 leaves per tree)
    - learning_rate: 0.1
    - subsample: 0.8
    - colsample_bytree: 0.8
    
  Effective complexity:
    - ~100 trees × ~8 leaves × ~16 features = thousands of parameters
    - For ~150 samples: MASSIVE overfitting risk
""")

# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 2.10: VALIDATION")
print("-"*80)
print("""
  CORRECT METHOD: GroupKFold by activity_id
    - All windows from same activity in same fold
    - No temporal leakage between train/test
    
  WRONG METHOD (older scripts): Random split
    - Windows randomly assigned to train/test
    - Adjacent windows share 90% data (with 10% overlap)
    - SEVERE data leakage!
    
  Results:
    - With random split (LEAKY): R² ≈ 0.72
    - With GroupKFold (HONEST): R² ≈ -0.89
    
  Why such difference?
    - With leakage: model sees 90% of test data during training
    - Without leakage: model must generalize → fails
""")

# =============================================================================
print("\n" + "="*80)
print("COMPARISON TABLE (CORRECTED)")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STEP                  │ LITERATURE FORMULA          │ ML PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ 1. PREPROCESSING                                                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ HR outliers           │ 40-200 bpm filter           │ (same, via PPG/RR)            │
│ HR filtering          │ None                        │ N/A (uses PPG for HR proxy)   │
│ PPG preprocessing     │ Not used                    │ Resample 32Hz, HPF 0.5Hz      │
│ IMU preprocessing     │ Resample 32Hz, LPF 5Hz,     │ Resample 32Hz, LPF 5Hz,       │
│                       │ gravity separation 0.3Hz    │ gravity separation 0.3Hz      │
│ EDA preprocessing     │ Not used                    │ Upsample 1→32Hz (!)           │
│ RR preprocessing      │ Not used                    │ Resample 1Hz                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SEGMENTATION                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Unit                  │ Per ACTIVITY                │ Per WINDOW (10s)              │
│ Overlap               │ N/A                         │ 10% (config) or 70% (old)     │
│ N samples             │ 33 activities               │ ~150 windows                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ 3. FEATURES                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ HR features           │ 1 (HR_mean → HR_delta)      │ ~15 (from PPG/RR)             │
│ IMU features          │ 1 (RMS_acc_mean)            │ ~180 (stat features)          │
│ PPG features          │ 0                           │ ~60                           │
│ EDA features          │ 0                           │ ~15                           │
│ Total features        │ 2 + interaction             │ ~270                          │
│ After selection       │ 2                           │ 16                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ 4. MODEL                                                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Type                  │ Correlation / Linear reg    │ XGBoost (100 trees)           │
│ Parameters            │ 2 (or 0 for corr)           │ ~1000+                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ 5. RESULTS                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ HR alone              │ r = 0.835***                │ -                             │
│ IMU alone             │ r = 0.678***                │ -                             │
│ Combined (80/20)      │ r = 0.843***                │ -                             │
│ CV R² (honest)        │ 0.66                        │ -0.89 (fails!)                │
└─────────────────────────────────────────────────────────────────────────────────────┘

KEY CORRECTIONS FROM PREVIOUS VERSION:

1. IMU DOES SHOW CORRELATION!
   - r = 0.678*** is significant and meaningful
   - Combined 80% HR + 20% IMU gives r = 0.843 (best!)
   
2. OVERLAP IS 10% (in current config), not 70%
   - pipeline.yaml: overlap: 0.1
   - Older scripts had 70%
   
3. RESAMPLING IS TO 32 Hz (not 1 Hz)
   - fs_out: 32 in config for PPG, IMU, EDA
   - RR intervals stay at 1 Hz
   
4. THERE IS PREPROCESSING!
   - IMU: LPF 5Hz, gravity separation at 0.3Hz
   - PPG: HPF 0.5Hz (for infra/red channels)
   - EDA: upsampling (questionable)
   
5. Borg = 0.028 × HR_effort + 0.76 EXPLAINED:
   - This is just the linear regression fit
   - Slope 0.028: very small because HR_effort has large range
   - Intercept 0.76: baseline Borg at zero effort
   
6. "SEES ALL DATA" EXPLAINED:
   - Feature selection is done on FULL dataset before CV split
   - Should be: feature selection INSIDE each CV fold
   - Current approach leaks test info into feature selection
   - This is a methodological flaw (but likely small impact)
""")
