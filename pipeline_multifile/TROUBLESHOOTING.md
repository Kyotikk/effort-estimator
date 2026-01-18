# Troubleshooting Guide

## Common Issues & Solutions

### Issue 1: Zero Labeled Samples

**Symptom**: Pipeline completes but produces 0 labeled samples in aligned output.

**Root Cause**: Window time ranges don't overlap with ADL recording time.

**Solution**: This is now AUTOMATICALLY FIXED by window time range filtering in Stage 7.

**Verification**:
1. Check logs for "Windows within ADL time range:"
2. Should see >300 windows after filtering
3. If 0 after filtering, ADL file may be missing or corrupted

**Prevention**: Ensure ADL CSV file exists and has valid timestamps

---

### Issue 2: Features All NaN

**Symptom**: Feature extraction completes but all feature columns are NaN.

**Root Cause**: 
- Preprocessing failed (signal not loaded correctly)
- Window too short for feature calculation
- Resampling issue in preprocessing

**Solution**:
1. Check preprocessed CSV files exist (e.g., `imu_bioz_preprocessed.csv`)
2. Verify preprocessed files have data (not all zeros/NaN)
3. Check window length is >= 2 seconds minimum
4. Review preprocessing config

**Commands**:
```bash
# Check if preprocessing outputs exist
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/*preprocessed*

# Check file contents
head /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/imu_bioz_preprocessed.csv
```

---

### Issue 3: Memory Error During Training

**Symptom**: "MemoryError" or "out of memory" during model training.

**Root Cause**: Large feature matrix (257 features × 1,188 samples) or XGBoost parameters

**Solution**:
1. Reduce max_depth (default 6 → try 4-5)
2. Reduce n_estimators (default 500 → try 100-200)
3. Use subsample < 1.0 to sample training data
4. Close other applications

**Code Change**:
```python
model = xgb.XGBRegressor(
    n_estimators=200,      # Reduced from 500
    max_depth=5,           # Reduced from 6
    subsample=0.8,         # Sample 80% of data
    random_state=42
)
```

---

### Issue 4: Model R² = 1.0 (Too Perfect)

**Symptom**: Training set R² is exactly 1.0, seems unrealistic.

**Root Cause**: Target variable (borg) included as feature by mistake.

**Solution**: Check `get_drop_columns()` includes 'borg' in drop list.

**Verification**:
```python
# In train_condition_specific_xgboost.py
DROP_COLS = [
    ...,
    "borg",  # ← MUST include this!
    ...
]
```

**Fix**: Add 'borg' to DROP_COLS if missing

---

### Issue 5: ADL File Not Found

**Symptom**: "ADL file not found" error or None returned from find_file()

**Root Cause**: 
- ADL file naming doesn't match pattern
- File is compressed (.gz) when code expects uncompressed
- File in different directory

**Solution**:

1. Check what ADL files exist:
```bash
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/
```

2. Update pattern in find_file() if needed:
```python
# Current pattern
adl_path = find_file(subject_path, ["scai_app", "ADLs"], exclude_gz=True)

# If file named differently, adjust pattern
adl_path = find_file(subject_path, ["scai_app", "Activities"])
```

3. Check file is uncompressed (if using exclude_gz=True):
```bash
file /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1-2.csv
# Should say "ASCII text" not "gzip compressed"
```

---

### Issue 6: Different Model Performance Than Expected

**Symptom**: Model R² differs from documentation values

**Root Causes**:
- Different random seed (non-deterministic split)
- Different data (ADL filters changed)
- Different features (feature selection changed)
- Different hyperparameters (training config changed)

**Solution**:
1. Verify random_state=42 in train/test split
2. Check dataset has 1,188 labeled samples
3. Verify top 100 features selected
4. Check XGBoost hyperparameters match

**Reproducibility Check**:
```python
# Ensure deterministic
random_state = 42
np.random.seed(random_state)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state  # ← Critical!
)
```

---

### Issue 7: Inconsistent Results Across Runs

**Symptom**: Different results when running pipeline twice

**Causes**:
- Random seeds not set
- File discovery finding different files
- Timestamp format changes

**Solution**:
```python
# Set all random seeds
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
```

---

### Issue 8: Fusion Creates Empty DataFrame

**Symptom**: Fused output has 0 rows

**Root Cause**: 
- Modalities have no overlapping windows
- Forward-fill reached NaN threshold
- Time alignment too strict

**Solution**:
1. Check each modality has windows in similar time range
2. Verify forward_fill parameters
3. Increase NaN tolerance if needed

**Debug**:
```python
# Check each modality's time coverage
for modality in ['imu_bioz', 'ppg_green', 'eda', 'rr']:
    df = pd.read_csv(f"{modality}_windows.csv")
    print(f"{modality}: {len(df)} rows, time range {df['t_center'].min()}-{df['t_center'].max()}")
```

---

### Issue 9: Feature Selection Returns Wrong Feature Count

**Symptom**: Only 50 features selected instead of 100

**Root Cause**: 
- Fewer than 100 unique features in dataset
- Variance calculation excluding valid features
- Duplicate column names

**Solution**:
1. Check how many features input has:
```python
cols_to_drop = get_drop_columns(df)
features_available = [c for c in df.columns if c not in cols_to_drop]
print(f"Features available: {len(features_available)}")
```

2. If < 100, use all available features:
```python
n_features = min(100, len(features_available))
```

---

### Issue 10: Borg Values Outside 0-10 Range

**Symptom**: Predicted Borg values are negative or > 10

**Root Cause**: 
- Model overfitting to training range
- New data distribution very different
- Extreme values in features

**Solution**:
1. Clamp predictions to [0, 10]:
```python
effort = np.clip(effort, 0, 10)
```

2. Check input feature ranges:
```python
print(features.describe())
```

3. Consider standardization or scaling

---

### Issue 11: Performance Very Different Per Condition

**Symptom**: healthy3 model performs much worse than elderly3

**Root Cause**: 
- healthy3 data extremely narrow (93.7% at 0-1 Borg)
- Insufficient variation for model to learn
- Only 2 unique effort levels in many windows

**Solution**:
- This is expected! healthy3 is naturally constrained
- Use elderly3 or severe3 for broader effort ranges
- healthy3 suitable only for light activity detection

**Documentation**: See training results in 10_MODEL_TRAINING.md

---

## Performance Validation

### Before Production Deployment

Run this checklist:

```python
# 1. Load all models
for cond in ['sim_elderly3', 'sim_healthy3', 'sim_severe3']:
    model, scaler, features = load_model_artifacts(cond)
    print(f"✓ {cond} loaded")

# 2. Test inference
test_features = pd.DataFrame(np.random.randn(1, 257))
for cond in conditions:
    effort = estimate_effort(test_features.iloc[0], cond)
    assert 0 <= effort <= 10, f"Invalid effort: {effort}"
    print(f"✓ {cond} inference works")

# 3. Check performance on test set
test_df = pd.read_csv("multisub_aligned_10.0s.csv")
for cond in conditions:
    cond_data = test_df[test_df['subject'] == cond]
    if len(cond_data) > 0:
        efforts = [estimate_effort(row, cond) for _, row in cond_data.iterrows()]
        print(f"✓ {cond} batch inference: {len(efforts)} samples")
```

---

## Getting Help

If issue persists:

1. **Check logs**: Look for error messages and traceback
2. **Verify data**: Ensure input files exist and aren't corrupted
3. **Test isolation**: Run single components separately
4. **Compare to documentation**: Verify against expected behavior
5. **Review git history**: Check what changed recently

---

## Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Features shape: {X.shape}")
logger.debug(f"Model loaded: {model}")
logger.debug(f"Prediction: {effort}")
```
