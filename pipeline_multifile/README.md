# Effort Estimator Pipeline - Complete Documentation

**Version**: 2.0 (Comprehensive Multi-File Structure)
**Status**: Production-ready with 3 condition-specific effort models
**Last Updated**: Current Session

---

## 📋 Documentation Index

This folder contains comprehensive documentation of the effort estimation pipeline, organized into logical stages:

### Core Documentation

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and overview
   - Architecture overview
   - Data flow diagram
   - Key components status
   - Supported conditions

2. **[01_DATA_INPUT.md](01_DATA_INPUT.md)** - Data sources and organization
   - Input data sources (sensors, labels)
   - Subject conditions
   - Data directory structure

3. **[02_PREPROCESSING.md](02_PREPROCESSING.md)** - Signal preprocessing per modality
   - IMU preprocessing (chest & wrist)
   - PPG preprocessing (green, IR, red)
   - EDA preprocessing
   - RR preprocessing

4. **[03_WINDOWING.md](03_WINDOWING.md)** - Time-series windowing
   - Window creation process
   - Window validity criteria
   - Window output structure

5. **[04_FEATURE_EXTRACTION.md](04_FEATURE_EXTRACTION.md)** - Feature computation
   - IMU features (138 total)
   - PPG features (50 total)
   - EDA features (10 total)
   - RR features (5 total)
   - Feature matrix output

6. **[05_FUSION.md](05_FUSION.md)** - Multi-modal feature fusion
   - Fusion process
   - Temporal alignment
   - Output structure

7. **[06_QUALITY_CHECKS.md](06_QUALITY_CHECKS.md)** - Data quality validation
   - Quality metrics
   - Validity thresholds
   - Visualization outputs

8. **[07_ADL_ALIGNMENT.md](07_ADL_ALIGNMENT.md)** - Effort label attachment
   - ADL data parsing
   - Window-to-Borg alignment
   - **Time range filtering** (critical feature)
   - Aligned output structure

9. **[08_MULTI_SUBJECT.md](08_MULTI_SUBJECT.md)** - Multi-subject dataset combination
   - Combination process
   - Dataset statistics by condition
   - Insights on condition differences

10. **[09_FEATURE_SELECTION.md](09_FEATURE_SELECTION.md)** - Feature dimensionality reduction
    - Selection method (variance-based)
    - Feature ranking
    - Output artifacts

11. **[10_MODEL_TRAINING.md](10_MODEL_TRAINING.md)** - Model training and evaluation
    - Condition-specific architecture (3 models)
    - Training process
    - Results per condition
    - Model artifacts saved

12. **[11_INFERENCE.md](11_INFERENCE.md)** - Making predictions with trained models
    - Inference process
    - Condition selection
    - Code examples
    - Batch inference

### Reference Documentation

- **[CONFIGURATION.md](CONFIGURATION.md)** - Pipeline configuration files
- **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - All scripts and their purposes
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md)** - Pre-production validation steps
- **[SUMMARY.md](SUMMARY.md)** - Summary statistics and performance

---

## 🎯 Quick Start

### For Running the Full Pipeline:
```bash
# Process all 3 subjects and train models
python run_multisub_pipeline.py

# Train condition-specific effort models
python train_condition_specific_xgboost.py

# Analyze model performance by effort level
python analyze_condition_models.py
```

### For Making Predictions:
```python
# Load a condition-specific model
from features.manual_features_imu import load_model

model = load_model("sim_elderly3")  # or healthy3, severe3
scaler = load_model_scaler("sim_elderly3")

# Get features from preprocessing
features_df = pd.read_csv("fused_10.0s.csv")

# Standardize and predict
features_scaled = scaler.transform(features_df)
effort = model.predict(features_scaled)
```

---

## 🏗️ System Architecture

### Three-Stage Processing:

```
Raw Sensor Data (7 modalities)
    ↓
[Preprocessing] - Clean, normalize, filter per modality
    ↓
[Feature Extraction] - Compute 257 statistical features
    ↓
[Fusion] - Combine all modalities time-aligned
    ↓
[ADL Alignment] - Attach Borg effort labels (with time filtering!)
    ↓
[Multi-Subject Combination] - Merge 3 conditions into one dataset
    ↓
[Feature Selection] - Select top 100 features per condition
    ↓
[Training] - Train 3 separate XGBoost models
    ↓
[Inference] - Predict effort for new data
```

### Three Effort Models:

| Condition | Samples | Borg Range | R² | Best For |
|-----------|---------|-----------|-----|----------|
| **elderly3** | 429 | 0.5-6.0 | 0.926 | Aging adults |
| **healthy3** | 347 | 0.0-1.5 | 0.405 | Light activities |
| **severe3** | 412 | 1.5-8.0 | 0.997 | High intensity ⭐ |

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| Total Labeled Samples | 1,188 |
| Total Features (pre-selection) | 257 |
| Selected Features (per condition) | 100 |
| Window Sizes | 10s, 5s, 2s |
| Window Overlap | 70% |
| Sensor Modalities | 7 (2 IMU + 3 PPG + 1 EDA + 1 RR) |
| Best Model Performance | R² = 0.997 (severe3) |
| Processing Time (per subject) | ~5-10 minutes |

---

## ⚙️ Data Flow

**Input Data Location**: `/Users/pascalschlegel/data/interim/parsingsim3/{subject}/`

**Output Location**: 
- Single subject: `{subject}/effort_estimation_output/`
- Multi-subject combined: `/multisub_combined/multisub_aligned_10.0s.csv`

**Models Location**: `/multisub_combined/models/`
- `sim_elderly3_model.json` + `sim_elderly3_scaler.pkl`
- `sim_healthy3_model.json` + `sim_healthy3_scaler.pkl`
- `sim_severe3_model.json` + `sim_severe3_scaler.pkl`

---

## ✅ Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| Data Preprocessing | ✅ Complete | All 7 modalities implemented |
| Feature Extraction | ✅ Complete | 257 features extracted |
| Fusion | ✅ Complete | Time-aligned combination working |
| ADL Alignment | ✅ Complete | Window time filtering active |
| Multi-Subject | ✅ Complete | 1,188 labeled samples across 3 conditions |
| Feature Selection | ✅ Complete | Top 100 per condition |
| Training | ✅ Complete | 3 models trained (R² 0.405-0.997) |
| Inference | ✅ Ready | Apply condition-specific models |

---

## 🔍 Critical Insights

1. **Condition-Specific Models Required**: Different populations have vastly different effort ranges
   - elderly3 mean: 3.30 Borg
   - healthy3 mean: 0.28 Borg (light activities)
   - severe3 mean: 4.71 Borg (high intensity)

2. **Time Range Filtering is Critical**: Sensors may start recording before ADL app starts
   - healthy3: 52-minute desynchronization
   - Solution: Filter windows to ADL time bounds
   - Result: Got 347 labeled samples (was 0 without filtering!)

3. **Condition Must Be Known**: Models are not auto-classifying
   - Condition comes from subject metadata/database
   - Specify condition when applying effort model

4. **Best Performance on High-Intensity**: severe3 model achieves R² = 0.997
   - Recommendation: Use severe3 model for production
   - Fallback: elderly3 (R² = 0.926) for moderate efforts

---

## 📖 Recommended Reading Order

**First Time?** Start here:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the big picture
2. [01_DATA_INPUT.md](01_DATA_INPUT.md) - See what data goes in
3. [07_ADL_ALIGNMENT.md](07_ADL_ALIGNMENT.md) - Understand the labeling process
4. [10_MODEL_TRAINING.md](10_MODEL_TRAINING.md) - See model results

**Deep Dive?** Read in order:
1. [02_PREPROCESSING.md](02_PREPROCESSING.md)
2. [03_WINDOWING.md](03_WINDOWING.md)
3. [04_FEATURE_EXTRACTION.md](04_FEATURE_EXTRACTION.md)
4. [05_FUSION.md](05_FUSION.md)
5. [06_QUALITY_CHECKS.md](06_QUALITY_CHECKS.md)
6. [07_ADL_ALIGNMENT.md](07_ADL_ALIGNMENT.md)

**Troubleshooting?** See:
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md)

---

## 🚀 Next Steps

1. **Validate Models**: Run validation checklist before production
2. **Test Inference**: Try with new sensor data
3. **Monitor Performance**: Track accuracy in real-world use
4. **Iterate**: Gather feedback and retrain as needed

---

**Questions?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or check specific stage documentation.

**Git Commits**:
- `34db0b8` - Add comprehensive pipeline documentation
- `25d3921` - Remove condition classifier, keep 3 effort models
