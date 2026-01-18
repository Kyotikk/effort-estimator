# Stages 3-6 & 8-9 (Summary)

This folder contains comprehensive documentation. Below is a summary of the remaining stages.

**Full documentation available in README.md** with links to each file.

## Quick Reference

### Stage 3: Windowing
- Fixed-size 10s, 5s, 2s windows
- 70% overlap
- Validity criteria: ≥80% samples present

**File**: See main PIPELINE_DOCUMENTATION.md for details

### Stage 4: Feature Extraction  
- 257 total features extracted
- 138 IMU + 50 PPG + 10 EDA + 5 RR
- Statistical & temporal metrics

**File**: See main PIPELINE_DOCUMENTATION.md for details

### Stage 5: Fusion
- Merge all modalities
- Time-aligned at window centers
- Forward-fill missing values

**File**: See main PIPELINE_DOCUMENTATION.md for details

### Stage 6: Quality Checks
- Validate NaN rates
- Check feature coverage
- Visualize distributions

**File**: See main PIPELINE_DOCUMENTATION.md for details

### Stage 8: Multi-Subject Combination
- Merge elderly3, healthy3, severe3
- 1,188 labeled samples total
- Ready for training

**File**: 08_MULTI_SUBJECT.md (or see main doc)

### Stage 9: Feature Selection
- Variance-based ranking
- Top 100 per condition
- Drop 69 metadata columns

**File**: 09_FEATURE_SELECTION.md (or see main doc)

---

## Complete Stage List

✅ Stage 1: Data Input - [01_DATA_INPUT.md](01_DATA_INPUT.md)
✅ Stage 2: Preprocessing - [02_PREPROCESSING.md](02_PREPROCESSING.md)
✅ Stage 3: Windowing - see [ARCHITECTURE.md](ARCHITECTURE.md)
✅ Stage 4: Feature Extraction - see [ARCHITECTURE.md](ARCHITECTURE.md)
✅ Stage 5: Fusion - [05_FUSION.md](05_FUSION.md) or see ARCHITECTURE
✅ Stage 6: Quality Checks - [06_QUALITY_CHECKS.md](06_QUALITY_CHECKS.md) or see ARCHITECTURE
✅ Stage 7: ADL Alignment - [07_ADL_ALIGNMENT.md](07_ADL_ALIGNMENT.md) ⭐ CRITICAL
✅ Stage 8: Multi-Subject - see main doc
✅ Stage 9: Feature Selection - see main doc
✅ Stage 10: Model Training - [10_MODEL_TRAINING.md](10_MODEL_TRAINING.md)
✅ Stage 11: Inference - [11_INFERENCE.md](11_INFERENCE.md)

---

See [README.md](README.md) for complete index.
