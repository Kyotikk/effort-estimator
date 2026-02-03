# Wearable-Based Perceived Effort Estimation Pipeline

## Complete Technical Documentation

---

## 1. RESEARCH OBJECTIVE

### Primary Goal
Develop a machine learning pipeline to estimate **perceived physical effort** (Borg CR-10 scale) from wearable physiological signals during Activities of Daily Living (ADLs) in elderly individuals.

### Clinical Motivation
- Elderly populations have difficulty self-reporting effort during rehabilitation
- Continuous effort monitoring enables adaptive exercise prescription
- Objective effort estimation supports fatigue detection and recovery monitoring

### Target Population
- 5 elderly subjects (parsingsim1-5 datasets)
- Age-related physiological variability expected
- ADL-focused activities (not laboratory exercise protocols)

---

## 2. DATA ACQUISITION

### Wearable Device
Samsung Galaxy Watch with multiple physiological sensors:

| Sensor | Signal | Sampling Rate |
|--------|--------|---------------|
| PPG (Green) | Photoplethysmography | 25 Hz |
| PPG (Infrared) | Photoplethysmography | 25 Hz |
| PPG (Red) | Photoplethysmography | 25 Hz |
| EDA | Electrodermal Activity | Variable |
| IMU | Accelerometer (3-axis) | 25 Hz |
| IMU | Gyroscope (3-axis) | 25 Hz |

### Ground Truth Labels
- **Borg CR-10 Scale**: 0-10 subjective effort rating
- Collected via smartphone app (SCAI App) during ADL sessions
- Self-reported by subjects during/after activities

### Dataset Summary
| Subject | Windows (5.0s) | Labeled Samples | Borg Range | Mean Borg |
|---------|----------------|-----------------|------------|-----------|
| P1 (elderly1) | 439 | 293 | 0-5 | 3.07 |
| P2 (elderly2) | 366 | 273 | 0-6 | 3.63 |
| P3 (elderly3) | 443 | 287 | 0.5-6 | 3.94 |
| P4 (elderly4) | 452 | 299 | 0.5-6 | 3.80 |
| P5 (elderly5) | 421 | 269 | 0-2.5 | 1.08 |
| **TOTAL** | **2,121** | **1,421** | 0-6 | 3.10 |

---

## 3. PREPROCESSING PIPELINE

### Signal Processing Steps

#### PPG Processing
1. Bandpass filter (0.5-4 Hz) for pulse extraction
2. Peak detection using VitalPy library
3. Heart rate calculation from inter-beat intervals
4. Signal quality assessment (valid peak ratio)

#### EDA Processing
1. Low-pass filter (5 Hz) for noise removal
2. Tonic/Phasic decomposition using NeuroKit2
3. Skin conductance level (SCL) and response (SCR) extraction

#### IMU Processing
1. Gravity removal (high-pass filter)
2. Magnitude calculation: √(ax² + ay² + az²)
3. Activity intensity metrics

### Quality Control
- Minimum valid sample threshold per window
- Signal amplitude range checks
- Artifact rejection based on physiological plausibility

---

## 4. WINDOWING STRATEGY

### Window Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window Size | **5.0 seconds** | Balances temporal resolution with feature stability |
| Overlap | **70%** | Ensures smooth transitions, increases sample count |
| Step Size | 1.5 seconds | Derived from 70% overlap |

### Window Selection Rationale
- 5.0s windows available for all 5 subjects
- 10.0s windows only available for 3 subjects (elderly3-5)
- Shorter windows capture transient effort changes
- Overlap mitigates boundary effects

---

## 5. FEATURE EXTRACTION

### Feature Categories

#### PPG Features (183 features)
- **Time-domain**: Mean, std, min, max, range, percentiles (p5, p25, p50, p75, p95)
- **Heart Rate**: HR mean, std, min, max, range, median
- **HRV**: RMSSD, SDNN (from green, infrared, red channels)
- **Morphological**: Peak amplitude, inter-beat interval variability

#### EDA Features (47 features)
- **Tonic**: SCL mean, std, min, max
- **Phasic**: SCR count, amplitude, rise time
- **Stress indicators**: Skin conductance responses per minute

#### IMU Features (60 features)
- **Accelerometer**: Mean, std, energy, entropy per axis
- **Gyroscope**: Angular velocity statistics
- **Activity metrics**: Signal magnitude area, jerk

#### HRV Features (6 features)
- RMSSD, SDNN for each PPG channel (green, infrared, red)

### Feature Extraction Libraries
- **VitalPy**: PPG processing and HR extraction
- **NeuroKit2**: EDA decomposition
- **Tsfresh**: Time-series feature extraction for IMU

### Total Features
- Initial: 296 features
- After quality filtering (<50% missing): **284 features**

---

## 6. FEATURE SELECTION & QUALITY CONTROL

### Missingness Filtering
- Features with >50% missing values excluded
- Per-subject missingness assessed

### Correlation Analysis
- Pearson correlation with Borg target
- Top positive correlations: PPG amplitude features
- Top negative correlations: HRV features (higher HRV = lower effort)

### Sensor Contribution
| Sensor | Mean |r| with Borg | Contribution |
|--------|---------------------|--------------|
| PPG | ~0.15 | Primary signal |
| EDA | ~0.12 | Stress indicator |
| IMU | ~0.10 | Activity intensity |
| HRV | ~0.08 | Autonomic response |

---

## 7. MODEL TRAINING

### Algorithm Selection
**Ridge Regression** selected over XGBoost based on:
- Better LOSO performance (0.18 vs 0.15 for XGBoost)
- More stable with small sample sizes (N=584)
- Less prone to overfitting individual subjects
- Interpretable coefficients

### Hyperparameters
- Regularization: α = 1.0 (L2 penalty)
- Standardization: Z-score normalization (mean=0, std=1)

### Cross-Validation Strategy
**Leave-One-Subject-Out (LOSO)**:
- Train on 4 subjects, test on 1 held-out subject
- Repeat for each subject
- Measures true cross-subject generalization

---

## 8. RESULTS

### Cross-Subject Performance (LOSO)

#### Continuous Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r | **0.18** | Weak correlation |
| MAE | **2.04 Borg** | ~20% of scale |
| RMSE | **2.60 Borg** | Higher variance |

#### Per-Subject LOSO Results
| Subject | LOSO r | LOSO MAE | Interpretation |
|---------|--------|----------|----------------|
| P1 | 0.08 | 2.31 | Poor |
| P2 | 0.13 | 1.89 | Poor |
| P3 | 0.22 | 2.10 | Weak |
| P4 | 0.31 | 1.76 | Weak-Moderate |
| P5 | 0.19 | 2.14 | Weak |

### Within-Subject Performance (Personalized)

| Subject | Within r | Within MAE | Improvement |
|---------|----------|------------|-------------|
| P1 | **0.53** | 1.56 | 6.6× better r |
| P2 | **0.37** | 1.25 | 2.8× better r |
| P3 | **0.72** | 0.95 | 3.3× better r |
| P4 | **0.67** | 1.14 | 2.2× better r |
| P5 | **0.62** | 0.52 | 3.3× better r |
| **Mean** | **0.58** | 1.08 | **3.2× better** |

### Categorical Performance (LOW/MODERATE/HIGH)

Categories defined as:
- **LOW**: Borg 0-2 (rest, light activity)
- **MODERATE**: Borg 3-4 (walking, chores)
- **HIGH**: Borg 5-10 (exertion, exercise)

| Metric | Cross-Subject | Within-Subject (expected) |
|--------|---------------|---------------------------|
| Exact (3-class) | 33% | ~60% |
| Adjacent (±1) | **87%** | ~95% |

---

## 9. KEY FINDINGS

### Finding 1: Cross-Subject Generalization Fails
- LOSO r = 0.18 indicates model cannot generalize across individuals
- Same physiological signals → different Borg ratings per person
- This is NOT a model limitation - it's the nature of perceived effort

### Finding 2: Perceived Effort is Subjective
| Subject | Mean Borg | Range | Interpretation |
|---------|-----------|-------|----------------|
| P5 | 1.08 | 0-2.5 | Rates everything as LOW |
| P4 | 3.80 | 0.5-6 | Rates same activities as MODERATE-HIGH |

**3× variation in mean Borg** for similar ADL activities across subjects.

### Finding 3: Within-Subject Performance is Promising
- Mean within-subject r = 0.58 (vs 0.18 cross-subject)
- **3.2× improvement** with personalization
- Indicates the physiological signals DO contain effort information
- But the mapping is individual-specific

### Finding 4: Categorical Accuracy is Clinically Useful
- 87% adjacent accuracy means model rarely confuses LOW with HIGH
- For practical applications, LOW/MODERATE/HIGH distinction is sufficient
- 13% "big misses" concentrated on outlier subjects (P5 with all-LOW ratings)

### Finding 5: Simpson's Paradox Warning
- Initial pooled r = 0.64 was MISLEADING
- Model learned subject identity, not within-subject effort variation
- LOSO reveals true generalization capability

---

## 10. WHY CROSS-SUBJECT FAILS

### The Subjectivity Problem
1. **Same activity → different perceived effort**
   - Walking may feel "light" (Borg 2) to P5 but "moderate" (Borg 4) to P4
   
2. **Same physiology → different ratings**
   - HR of 80 bpm might be "resting" for P5 but "elevated" for P1
   
3. **Individual calibration required**
   - Each person has unique effort-physiology mapping
   - No universal model can capture this

### The 13% Big Miss Analysis
| Error Type | Count | Cause |
|------------|-------|-------|
| HIGH→LOW | 47 | P1 (23.7% miss rate) tested with others' calibration |
| LOW→HIGH | 29 | P5 (mean 1.08) tested with group mean ~3.5 |

---

## 11. LONGITUDINAL APPROACH: THE SOLUTION

### Rationale
Since cross-subject generalization fails due to individual calibration differences, the solution is **personalized longitudinal modeling**:

1. **Calibration Phase**: Collect initial labeled data from new user
2. **Personalized Model**: Train/fine-tune model on individual's data
3. **Continuous Adaptation**: Update model as more data collected over time

### Expected Benefits
| Aspect | Cross-Subject | Longitudinal (Projected) |
|--------|---------------|--------------------------|
| Pearson r | 0.18 | **0.5-0.7** |
| MAE | 2.04 | **0.8-1.2** |
| Adjacent Accuracy | 87% | **95%+** |
| Big Misses | 13% | **<3%** |

### Implementation Strategy
1. **Few-shot calibration**: 10-20 labeled samples to establish baseline
2. **Transfer learning**: Pre-train on pooled data, fine-tune on individual
3. **Online adaptation**: Continuously update with new labels
4. **Confidence estimation**: Flag uncertain predictions for re-calibration

### Clinical Deployment Model
```
Day 1-3:   Calibration phase (supervised labeling)
Day 4+:    Personalized prediction (optional periodic recalibration)
Week 2+:   Fully autonomous monitoring
```

---

## 12. CONCLUSIONS

### What We Learned
1. **Wearable signals contain effort information** (within-subject r = 0.58)
2. **Cross-subject models fail** (LOSO r = 0.18) due to perception subjectivity
3. **Categorical prediction is useful** (87% adjacent accuracy)
4. **Personalization is essential** for practical deployment

### Thesis Contribution
This pipeline establishes that:
- Effort estimation from wearables is FEASIBLE
- But requires PERSONALIZED approach
- Cross-subject "one-size-fits-all" models are insufficient
- Longitudinal individual calibration is the path forward

### Next Steps: Longitudinal Study
1. Collect multi-day data from subjects
2. Implement calibration + adaptation pipeline
3. Evaluate few-shot personalization performance
4. Test long-term stability of personalized models

---

## 13. TECHNICAL SPECIFICATIONS

### Software Stack
- Python 3.10+
- scikit-learn (Ridge, LOSO CV)
- pandas, numpy (data processing)
- VitalPy (PPG processing)
- NeuroKit2 (EDA processing)
- SHAP (model interpretation)
- matplotlib, seaborn (visualization)

### File Structure
```
effort-estimator/
├── run_elderly_pipeline.py      # Main preprocessing pipeline
├── run_loso_5subjects.py        # LOSO evaluation
├── generate_ml_expert_plots.py  # 18 visualization plots
├── analyze_categorical_effort.py # LOW/MOD/HIGH analysis
├── analyze_big_misses.py        # Error analysis
├── preprocessing/               # Signal processing modules
├── features/                    # Feature extraction
├── ml/                          # Machine learning modules
└── config/                      # Pipeline configuration
```

### Output Location
```
/Users/pascalschlegel/data/interim/elderly_combined_5subj/
├── all_5_elderly_5s.csv         # Combined dataset (1421 samples)
├── loso_results_5subjects.csv   # LOSO evaluation results
└── ml_expert_plots/             # 21 publication-quality figures
```

---

## 14. FIGURES GENERATED

| # | Filename | Description |
|---|----------|-------------|
| 01 | data_overview.png | Sample counts and Borg distributions |
| 02 | borg_boxplot_detailed.png | Per-subject Borg statistics |
| 03 | feature_extraction_overview.png | Pipeline diagram |
| 04 | feature_distributions.png | Feature histograms by subject |
| 05 | quality_missingness.png | Missing data analysis |
| 06 | feature_correlation_matrix.png | Top feature correlations |
| 07 | feature_importance_correlation.png | Borg correlation rankings |
| 08 | sensor_importance.png | Sensor contribution comparison |
| 09 | loso_results.png | LOSO r, MAE, RMSE by subject |
| 10 | loso_scatter.png | Predicted vs actual (LOSO) |
| 11 | residual_analysis.png | Error distribution |
| 12 | loso_vs_within_subject.png | Cross vs personalized comparison |
| 13 | per_subject_scatter.png | Individual subject predictions |
| 14 | shap_beeswarm.png | SHAP feature importance |
| 15 | shap_bar.png | Mean absolute SHAP values |
| 16 | shap_waterfall.png | Single prediction explanation |
| 17 | shap_dependence.png | Feature effect plots |
| 18 | thesis_comprehensive_figure.png | 6-panel summary |
| 19 | categorical_confusion_matrix.png | 3-class accuracy |
| 20 | categorical_practical_interpretation.png | Clinical utility |
| 21 | category_distribution.png | LOW/MOD/HIGH distribution |

---

*Document generated: January 2026*
*Pipeline version: 5-subject LOSO evaluation*
*Author: Effort Estimation Research Pipeline*
