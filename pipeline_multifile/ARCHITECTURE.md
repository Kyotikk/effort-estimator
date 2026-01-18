# Architecture Overview

## System Design

### Three-Stage Architecture

```
Raw Sensor Data (7 modalities: 2 IMU + 3 PPG + 1 EDA + 1 RR)
    ↓
[Preprocessing & Feature Extraction]
    ├─ Gravity removal, HPF filtering, resampling
    ├─ Extract 257 statistical & temporal features
    └─ Per-modality processing
    ↓
[Feature Fusion across modalities]
    ├─ Time-align all modalities
    ├─ Combine into single feature matrix
    └─ Handle missing values (forward-fill, NaN removal)
    ↓
[Windowing & Aggregation]
    ├─ 10s, 5s, 2s window lengths
    ├─ 70% overlap
    └─ Calculate features over each window
    ↓
[Quality Checks]
    ├─ Validate feature coverage
    ├─ Check NaN rates
    └─ Visualize distributions
    ↓
[ADL Alignment & Labeling]
    ├─ Parse ADL CSV with effort labels
    ├─ **Filter windows to ADL time range** ⚠️ CRITICAL
    └─ Attach Borg values to windows
    ↓
[Multi-Subject Combination]
    ├─ Merge elderly3, healthy3, severe3
    └─ 1,188 labeled samples total
    ↓
[Feature Selection]
    ├─ Rank by variance per condition
    ├─ Select top 100 per condition
    └─ Create condition-specific selectors
    ↓
[Model Training]
    ├─ Train elderly3 model (R²=0.926)
    ├─ Train healthy3 model (R²=0.405)
    └─ Train severe3 model (R²=0.997) ⭐
    ↓
[Inference]
    ├─ Load condition-specific model + scaler
    ├─ Feature selection + standardization
    └─ Predict Borg effort (0-10)
```

---

## Key Components

### 1. Sensors (7 modalities)

| Modality | Type | Sampling | Purpose |
|----------|------|----------|---------|
| **IMU Bioz** | Accelerometer | 32 Hz | Chest movement |
| **IMU Wrist** | Accelerometer | 32 Hz | Arm movement |
| **PPG Green** | Photoplethysmography | 32 Hz | Green LED heart rate |
| **PPG Infrared** | Photoplethysmography | 32 Hz | IR LED heart rate |
| **PPG Red** | Photoplethysmography | 32 Hz | Red LED perfusion |
| **EDA** | Electrodermal Activity | 32 Hz | Skin conductance (stress) |
| **RR** | Heart Rate Variability | Variable | Respiration intervals |

### 2. Processing Stages

#### Stage 1: Preprocessing
- **Purpose**: Clean raw signals, remove noise/artifacts
- **Per Modality**: Gravity removal, filtering, resampling
- **Output**: Cleaned time series per modality

#### Stage 2: Feature Extraction
- **Purpose**: Compute statistical features from windowed data
- **Method**: 257 hand-crafted features (mean, std, RMS, entropy, etc.)
- **Output**: Feature matrix (windows × features)

#### Stage 3: Fusion
- **Purpose**: Combine all modalities time-aligned
- **Method**: Merge on window centers with forward-fill
- **Output**: Single feature matrix with all modalities

#### Stage 4: Quality Checks
- **Purpose**: Validate data quality
- **Method**: NaN rate, validity rate, distribution analysis
- **Output**: Quality reports + histograms

#### Stage 5: ADL Alignment
- **Purpose**: Attach effort labels to windows
- **Critical**: **Filter windows to ADL recording time range**
- **Why**: Sensors may start before app starts (52 min desync seen!)
- **Output**: Labeled windows with Borg values

#### Stage 6: Multi-Subject Combination
- **Purpose**: Merge all conditions into training dataset
- **Method**: Concatenate + drop unlabeled rows
- **Output**: 1,188 labeled samples across 3 conditions

#### Stage 7: Feature Selection
- **Purpose**: Reduce from 257 → 100 features
- **Method**: Variance-based ranking per condition
- **Output**: Top-100 feature names + selectors

#### Stage 8: Training
- **Purpose**: Train condition-specific models
- **Method**: XGBoost regressors (80-20 train/test)
- **Output**: 3 models + scalers + feature importances

#### Stage 9: Inference
- **Purpose**: Predict effort for new data
- **Method**: Apply condition-specific model
- **Input**: Features + known condition
- **Output**: Borg effort (0-10)

---

## Supported Conditions

### Data Characteristics

| Condition | Population | Samples | Borg Range | Mean Borg | Use Case |
|-----------|-----------|---------|-----------|-----------|----------|
| **elderly3** | Elderly adults | 429 | 0.5-6.0 | 3.30 | Aging populations |
| **healthy3** | Young, healthy | 347 | 0.0-1.5 | 0.28 | Light activities |
| **severe3** | High intensity | 412 | 1.5-8.0 | 4.71 | Extreme effort ⭐ |

### Why Separate Models?

- **Different effort ranges**: healthy3 is mostly 0-1 Borg (light), severe3 is mostly 5-8 (extreme)
- **Different physiologies**: Heart rate, movement patterns differ by condition
- **Better accuracy**: Condition-specific models achieve R² 0.405-0.997 vs poor multi-subject model
- **Domain expertise**: Each population requires different feature relationships

---

## Model Performance

### By Condition (10s windows)

| Condition | Train R² | Test R² | MAE | RMSE | Best For |
|-----------|---------|---------|-----|------|----------|
| **elderly3** | 1.000 | 0.926 | 0.053 | 0.226 | Moderate effort |
| **healthy3** | 1.000 | 0.405 | 0.015 | 0.100 | Light activity |
| **severe3** | 1.000 | 0.997 | 0.026 | 0.112 | High intensity ⭐ |

### Recommendation

- **Production**: Use **severe3** model (R² = 0.997, most robust)
- **Fallback**: Use **elderly3** model (R² = 0.926, moderate effort)
- **Not Recommended**: healthy3 model (limited to light activities, narrow range)

---

## Data Flow Diagram

```
Raw Files                  Preprocessed              Features            Fused
┌──────────────┐           ┌─────────────┐          ┌──────────┐        ┌──────────────┐
│ IMU Bioz CSV │──────────→│ IMU accels  │──────────→│ 69 IMU   │        │              │
│ 32 Hz        │           │ gravity rm  │           │ features │        │              │
└──────────────┘           └─────────────┘          └──────────┘        │              │
                                                                         │ 257 Total    │
┌──────────────┐           ┌─────────────┐          ┌──────────┐        │ Features     │
│ PPG Green    │──────────→│ PPG signal  │──────────→│ 50 PPG   │───────→│ (all modt)    │
│ 32 Hz        │           │ HPF, resamp │           │ features │        │              │
└──────────────┘           └─────────────┘          └──────────┘        │ Per window   │
                                                                         │ (1,188       │
┌──────────────┐           ┌─────────────┐          ┌──────────┐        │ samples)     │
│ EDA + RR     │──────────→│ Cleaned sig │──────────→│ 15 EDA   │        │              │
│ Variable Hz  │           │ interpolat  │           │ + RR     │        │              │
└──────────────┘           └─────────────┘          └──────────┘        └──────────────┘
                                                                                │
                                                                                ↓
                                                                       ┌──────────────────┐
                                                                       │ ADL Labeling     │
                                                                       │ + Time Filtering │
                                                                       │ (keep 1,188 of   │
                                                                       │  3,810 samples)  │
                                                                       └──────────────────┘
                                                                                │
                                                                                ↓
                                                                       ┌──────────────────┐
                                                                       │ Train Models     │
                                                                       │ (3 separate)     │
                                                                       │ R²=0.926-0.997   │
                                                                       └──────────────────┘
```

---

## Critical Features

### 1. **Window Time Filtering**

**Problem**: Sensor windows may not overlap with ADL recording time
```
healthy3 example:
  Sensor start: 1764832240  ← 52 minutes BEFORE ADL app starts!
  ADL start:    1764835353
  Only 1,447 seconds overlap
  
Without filtering: 0 labeled samples ❌
With filtering: 347 labeled samples ✅
```

**Solution**: In [07_ADL_ALIGNMENT.md](07_ADL_ALIGNMENT.md), filter to `[ADL_t_min, ADL_t_max]`

### 2. **Condition-Specific Models**

**Why**: Effort ranges are fundamentally different
- elderly3: full distribution 0.5-6.0 (well-distributed)
- healthy3: 93.7% at 0-1 (extremely narrow!)
- severe3: 50% at extreme (5-8 Borg)

**Result**: Single model would be terrible; 3 separate models are much better

### 3. **Known Condition Requirement**

**Note**: Condition is NOT automatically classified
- Must come from subject metadata, database, or user input
- Enables use of specialized, optimized models
- Simpler architecture than classification + inference

---

## Quality Metrics

All data validated at each stage:

| Stage | Metric | Threshold | Action |
|-------|--------|-----------|--------|
| Preprocessing | NaN rate | < 5% | Pass through |
| Windowing | Valid windows | > 95% | Document coverage |
| Features | NaN rate per feature | < 20% | Warn; > 20% drop |
| Fusion | Coverage | > 90% | Remove rows with NaN |
| Alignment | Labeled | > 80% | Document unlabeled % |
| Selection | Feature importance | top 100 | Select by variance |

---

## Summary

- **Three stages**: Preprocessing → Feature extraction → Model training → Inference
- **7 modalities**: IMU, PPG (3 colors), EDA, RR
- **257 features**: Hand-crafted statistical & physiological metrics
- **3 models**: Separate for elderly, healthy, severe
- **Best performance**: severe3 (R² = 0.997)
- **Key insight**: Time range filtering critical for labels!
