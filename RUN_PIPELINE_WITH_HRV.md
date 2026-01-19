# Running the Pipeline with HRV Recovery Rate

## Quick Start (3 Commands)

```bash
# 1. Run single-subject pipeline
python run_pipeline.py config/pipeline.yaml

# 2. Run multi-subject training (3 subjects)
python run_pipeline_complete.py

# 3. View results in:
#    - Single-subject: data/effort_estimation_output/*/fused_aligned_10.0s.csv
#    - Multi-subject: data/effort_estimation_output/combined_features_10.0s.csv
```

---

## Step-by-Step: Single Subject Pipeline

### 1. Verify Configuration
```bash
# Check that config has ADL path (for effort labels)
cat config/pipeline.yaml | grep -A5 "targets:"
```

Expected output includes ADL file path, e.g.:
```yaml
targets:
  imu:
    adl_path: /path/to/sim_elderly3/scai_app/ADLs.csv
```

### 2. Run Preprocessing & Feature Extraction (Phases 1-4)
```bash
python run_pipeline.py config/pipeline.yaml
```

Pipeline will:
1. **Phase 1**: Preprocess IMU, PPG, EDA, RR signals
2. **Phase 2**: Create 10-second windows (70% overlap)
3. **Phase 3**: Extract 50+ features from each modality
4. **Phase 4**: Fuse all modalities → `fused_features_10.0s.csv`

**Check**: Does `fused_features_10.0s.csv` exist?
```bash
ls data/effort_estimation_output/sim_elderly3/fused_features_10.0s.csv
```

### 3. HRV Recovery Computation (Phase 5b - NEW)

When pipeline reaches "▶ Aligning fused features with HRV Recovery Rate labels", it will:

1. **Load PPG signal** from `ppg_green_preprocessed.csv`
2. **Parse ADL intervals** from ADL CSV (with Borg labels)
3. **For each activity**:
   - Extract 1-min baseline before activity
   - Extract activity phase (effort)
   - Extract 5-min recovery phase (post-activity)
   - Compute SDNN (HRV) for each phase
   - Calculate recovery rate = (HRV_recovery - HRV_effort) / 300 seconds

**Expected output**:
```
▶ Aligning fused features with HRV Recovery Rate labels
  Computing HRV recovery rates and aligning fused features (10.0s windows)...
  ✓ Computing HRV recovery rates and assigning to windows
  
  ✓ Activity 0: HRV_Recovery_Rate=0.735 ms/s (Borg=5, HRV 45.2→32.1→53.8 ms, assigned to 12 windows)
  ✓ Activity 1: HRV_Recovery_Rate=0.452 ms/s (Borg=7, HRV 42.1→28.5→46.9 ms, assigned to 9 windows)
  
  ✓ Successfully processed 8/10 activities
  ✓ Assigned HRV recovery rates to 95 windows
  ✓ HRV Recovery Rate range: 0.234 to 1.523 ms/s
  ✓ Interpretation:
      Fast recovery (>1.0 ms/s):  Low effort, good fitness
      Moderate (0.5-1.0 ms/s):   Normal effort
      Slow recovery (<0.5 ms/s): High effort or fatigued
```

### 4. Feature Selection (Phase 6)
```
▶ Feature selection with correlation pruning + quality checks
  Running feature selection (10.0s windows)...
  Loading data from: data/effort_estimation_output/sim_elderly3/fused_aligned_10.0s.csv
  ✓ Using HRV Recovery Rate as target variable (primary)
  ✓ Loaded 95 labeled samples
  
  🎯 Feature Selection (top 100 by correlation with target)...
  ✓ Selected 70 features after pruning
  
  ✓ Feature selection complete (10.0s)
```

### 5. Verify Output Files

```bash
# Check that HRV recovery rates were added
python -c "
import pandas as pd
df = pd.read_csv('data/effort_estimation_output/sim_elderly3/fused_aligned_10.0s.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)[:10]}...')  # First 10 cols
print(f'HRV Recovery Rates:')
print(df[['window_id', 'hrv_recovery_rate', 'hrv_baseline', 'activity_borg']].head(10))
print(f'Missing HRV: {df[\"hrv_recovery_rate\"].isna().sum()} out of {len(df)}')
"
```

Expected output:
```
Shape: (95, 273)
Columns: ['window_id', 'start_idx', 'end_idx', 'valid', 't_start', 't_center', 't_end', 'n_samples', 'win_sec', ...]
HRV Recovery Rates:
   window_id  hrv_recovery_rate  hrv_baseline  activity_borg
0      12345              0.735          45.2              5
1      12346              0.735          45.2              5
2      12347              0.735          45.2              5
...

Missing HRV: 0 out of 95
```

---

## Multi-Subject Training

### 1. Run Complete Pipeline (All 3 Subjects)

```bash
python run_pipeline_complete.py
```

This will:
1. Process each subject (sim_elderly3, sim_healthy3, sim_severe3)
   - Runs full pipeline (phases 1-6)
   - Generates HRV recovery rates for each
2. Combine all aligned features
3. Run feature selection on combined dataset
4. Display summary statistics

### 2. Expected Output

```
=== Processing subject: sim_elderly3 ===
✓ PREPROCESSING COMPLETE
...
=== Processing subject: sim_healthy3 ===
✓ PREPROCESSING COMPLETE
...
=== Processing subject: sim_severe3 ===
✓ PREPROCESSING COMPLETE
...

===============
COMBINING DATASETS
===============
  sim_elderly3: 180 samples (95 labeled)
  sim_healthy3: 200 samples (102 labeled)
  sim_severe3: 210 samples (98 labeled)

✓ Saved combined dataset: data/effort_estimation_output/combined_features_10.0s.csv

======================================
FEATURE SELECTION + QC
======================================
  ✓ Using HRV Recovery Rate as target variable (primary)
  ✓ Loaded 295 labeled samples
  ✓ 267 features (after metadata removal)
  
  Selecting top 100 features by correlation...
  ✓ Feature selection complete: 70 features selected
  
======================================
SUMMARY
======================================
  sim_elderly3: 180 samples (95 labeled)
  sim_healthy3: 200 samples (102 labeled)
  sim_severe3: 210 samples (98 labeled)

Total: 590 samples (295 labeled)
Features before selection: 267
Features after selection: 70
```

### 3. Check Combined Results

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/effort_estimation_output/combined_features_10.0s.csv')

print(f'Combined dataset: {df.shape}')
print(f'Subjects: {df[\"subject\"].unique()}')
print(f'HRV recovery rates available: {df[\"hrv_recovery_rate\"].notna().sum()}')

# Stats by subject
for subject in df['subject'].unique():
    sub_df = df[df['subject'] == subject]
    labeled = sub_df['hrv_recovery_rate'].notna().sum()
    print(f'{subject}: {len(sub_df)} total, {labeled} with HRV recovery')

# Overall stats
print(f'\nHRV Recovery Rate statistics:')
print(df['hrv_recovery_rate'].describe())
"
```

---

## Troubleshooting

### Issue: "HRV recovery computation failed"

**What to check**:
1. **PPG data quality**: Check for dropout or noise
   ```bash
   python -c "
   import pandas as pd
   ppg = pd.read_csv('data/effort_estimation_output/sim_elderly3/ppg_green/ppg_green_preprocessed.csv')
   print(f'PPG shape: {ppg.shape}')
   print(f'PPG missing: {ppg[\"value\"].isna().sum()}')
   print(f'PPG range: {ppg[\"value\"].min()} to {ppg[\"value\"].max()}')
   "
   ```

2. **ADL data format**: Check that ADL CSV has required columns
   ```bash
   python -c "
   import pandas as pd
   adl = pd.read_csv('path/to/ADL.csv')
   print(f'Columns: {list(adl.columns)}')
   print(f'Required: t_start, t_end, borg')
   print(adl.head())
   "
   ```

3. **Recovery window timing**: Check if recovery period overlaps with next activity
   - Solution: Increase `recovery_buffer_sec` in `compute_hrv_recovery.py` line ~60

### Issue: "No windows labeled" / "0 windows with recovery rates"

**Causes**:
- Recovery phases don't overlap with fused feature windows
- ADL timestamps don't match PPG timestamps (timezone issue)
- Insufficient PPG data during recovery

**Solution**:
1. Verify timestamps are in same timezone (Unix seconds or datetime)
2. Check window creation didn't skip recovery periods
3. Ensure `recovery_buffer_sec=300` (5 minutes) doesn't exceed data length

### Issue: "Using Borg RPE Scale as target (fallback)"

**Meaning**: HRV recovery failed, reverted to Borg-based alignment

**To debug**:
1. Check logs for specific error in HRV computation
2. Verify PPG signal quality (especially heart rate detection)
3. Check RR interval range is 300-2000 ms (30-200 bpm)

**Fallback is safe**: Model will still train on Borg labels with same performance as before

---

## Comparing Results: Borg vs HRV Recovery

### To Compare Performance

```bash
# View both baseline and new metrics
echo "=== Checkpoint: Borg-based (baseline) ==="
git show 0913ddf:METHODOLOGY.md | grep -A3 "Executive Summary"

echo "=== Current: HRV Recovery-based ==="
head -20 METHODOLOGY.md | grep -A3 "Executive Summary"
```

### Expected Differences

| Metric | Borg-based | HRV Recovery | Why |
|--------|-----------|--------------|-----|
| **Target type** | Discrete (0-10) | Continuous (float) | HRV is measured value |
| **# Samples** | 800-1000 | 300-500 | Different phase (recovery vs activity) |
| **R² score** | 0.9259 | 0.93-0.95 (expected) | More stable target variable |
| **RMSE** | 0.6528 | Lower (expected) | Continuous target has less quantization |
| **Feature importance** | IMU, PPG, EDA balanced | PPG + EDA dominant | Recovery depends on autonomic system |

### Generate Comparison Report

```python
# save as compare_borg_vs_hrv.py
import pandas as pd
import numpy as np

# Load Borg-based results
borg_df = pd.read_csv('data/effort_estimation_output/combined_features_10.0s.csv')
borg_count = borg_df['borg'].notna().sum()
print(f"Borg-based labeled: {borg_count}")
print(f"Borg range: {borg_df['borg'].min():.0f}-{borg_df['borg'].max():.0f}")

# Load HRV recovery results (after this implementation)
hrv_df = pd.read_csv('data/effort_estimation_output/combined_features_10.0s.csv')
hrv_count = hrv_df['hrv_recovery_rate'].notna().sum()
print(f"\nHRV Recovery-based labeled: {hrv_count}")
print(f"HRV Recovery range: {hrv_df['hrv_recovery_rate'].min():.3f}-{hrv_df['hrv_recovery_rate'].max():.3f} ms/s")

# Cross-comparison (only samples with both)
both = borg_df[borg_df['borg'].notna() & borg_df['hrv_recovery_rate'].notna()]
print(f"\nOverlap: {len(both)} samples with both Borg and HRV")
if len(both) > 0:
    corr = both['borg'].corr(both['hrv_recovery_rate'])
    print(f"Correlation: {corr:.3f}")
```

---

## Monitoring HRV Metrics During Pipeline

### Track in Real-time

```bash
# In another terminal, monitor output directory
watch -n 5 "ls -lh data/effort_estimation_output/*/fused_aligned_*.csv | tail -5"
```

### Extract HRV Statistics at Each Stage

```bash
# After pipeline completes, generate summary
python << 'EOF'
import pandas as pd
from pathlib import Path

output_dir = Path('data/effort_estimation_output')

for subject_dir in output_dir.glob('sim_*'):
    aligned_file = subject_dir / 'fused_aligned_10.0s.csv'
    if aligned_file.exists():
        df = pd.read_csv(aligned_file)
        hrv_col = df['hrv_recovery_rate']
        
        print(f"\n{subject_dir.name}:")
        print(f"  Total windows: {len(df)}")
        print(f"  With HRV labels: {hrv_col.notna().sum()}")
        print(f"  HRV mean: {hrv_col.mean():.3f} ms/s")
        print(f"  HRV std: {hrv_col.std():.3f}")
        print(f"  HRV range: {hrv_col.min():.3f} to {hrv_col.max():.3f}")
        
        # Categorize by recovery speed
        fast = (hrv_col > 1.0).sum()
        normal = ((hrv_col >= 0.5) & (hrv_col <= 1.0)).sum()
        slow = (hrv_col < 0.5).sum()
        print(f"  Distribution: Fast={fast}, Normal={normal}, Slow={slow}")
EOF
```

---

## Performance Validation

### Before Starting Training

1. **Check data quality**:
   ```bash
   # Verify all preprocessing outputs exist
   find data/effort_estimation_output -name "*_preprocessed.csv" | wc -l
   # Should be 6 per subject (imu_bioz, imu_wrist, ppg_green, ppg_infra, ppg_red, eda)
   ```

2. **Check HRV computation success rate**:
   ```bash
   grep "Successfully processed" data/effort_estimation_output/*/fused_aligned_10.0s.csv
   # Should show 100% or close to 100%
   ```

3. **Check feature selection results**:
   ```bash
   ls data/effort_estimation_output/*/feature_selection_qc/qc_10.0s/
   # Should have: features_selected_pruned.csv, pca_loadings.csv, etc.
   ```

### After Training (Next Step)

Follow [Performance Metrics Documentation](METHODOLOGY.md#performance-metrics) to evaluate:
- R² score (target: >0.92)
- RMSE (target: <0.7)
- MAE (target: <0.5)
- Feature importance (rank PPG/EDA/IMU features)

---

## Next: Training Models

Once HRV recovery rates are computed and validated:

```bash
# Single-subject XGBoost training
python train_multisub_xgboost.py --mode single

# Multi-subject cross-validation
python train_multisub_xgboost.py --mode cross_val

# Production model (all data)
python train_multisub_xgboost.py --mode production
```

See [METHODOLOGY.md - Section 6: Training](METHODOLOGY.md#section-6-training) for details.

---

**Status**: Ready to test! Run `python run_pipeline.py config/pipeline.yaml` to start.
