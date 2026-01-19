# HRV Recovery Rate Implementation

## Summary

Transitioned from **Borg RPE Scale** (subjective effort labels, 0-10) to **HRV Recovery Rate** (objective autonomic recovery metric, ms/second) as the primary target variable for effort estimation.

**Status**: ✅ Integrated, ready for testing  
**Git Commit**: `bb64313` (See full details with `git show bb64313`)  
**Checkpoint Available**: `0913ddf` (Borg-based working version, can rollback if needed)

---

## Why HRV Recovery Instead of Borg?

### Borg RPE Limitations
- **Subjective**: Depends on user perception (fatigue, mood, motivation)
- **Discrete**: Only 11 values (0-10 scale)
- **Single-point measurement**: Captured at one moment in time
- **Noisy labels**: Same effort level perceived differently by different people

### HRV Recovery Advantages
- **Objective**: Automatic measurement from PPG signal
- **Continuous**: Float values (e.g., 0.52 ms/s, 1.34 ms/s)
- **Physiologically grounded**: Measures autonomic nervous system recovery
- **Sensitive to effort intensity**: Fast recovery = low effort, slow recovery = high effort
- **Independent of perception**: Works regardless of user's subjective assessment

### Recovery Rate Interpretation
```
HRV Recovery Rate (ms/second)
  >1.0 ms/s:  Fast recovery → Low effort, good cardiovascular fitness
  0.5-1.0:    Moderate → Normal effort level
  <0.5 ms/s:  Slow recovery → High effort or fatigued state
```

---

## Implementation Details

### 1. New Module: `phases/phase5_alignment/compute_hrv_recovery.py` (280+ lines)

**Core Functions**:

#### `extract_rr_intervals(ppg_signal, fs=32, min_distance=15)`
- Detects heartbeats from PPG signal using peak detection
- Filters physiologically valid RR intervals: 300-2000 ms (30-200 bpm)
- Removes impossible values (e.g., >2 bpm or <30 bpm)

#### `compute_hrv_window(rr_intervals)`
- Calculates SDNN (Standard Deviation of NN intervals)
- SDNN is gold-standard HRV metric (ms)
- Returns NaN if insufficient data

#### `compute_hrv_recovery_rate(ppg_baseline, ppg_activity, ppg_recovery, fs=32)`
- Main computation function
- Returns: `(hrv_recovery_rate, hrv_baseline, hrv_effort, hrv_recovery, recovery_time_sec)`
- Three-phase HRV analysis:
  1. **Baseline** (1 min before activity): Resting HRV
  2. **Effort** (during activity): Minimum HRV during exertion
  3. **Recovery** (5 min after activity): HRV return to baseline
- Recovery rate = (HRV_recovery - HRV_effort) / time_in_seconds

#### `parse_adl_with_recovery(adl_df, data_start, data_end, recovery_buffer_sec=300)`
- Adds 5-minute recovery windows to ADL intervals
- Ensures recovery doesn't extend beyond data boundaries
- Returns enhanced ADL DataFrame with recovery phase timestamps

#### `align_windows_to_hrv_recovery(windows_df, ppg_df, adl_df, fs=32)`
- **Main pipeline function** (replaces old `run_alignment()`)
- Processes each ADL activity:
  - Computes HRV recovery rate
  - Assigns to all windows in recovery phase
- Returns labeled DataFrame with columns:
  - `hrv_recovery_rate`: Recovery rate (ms/s)
  - `hrv_baseline`: Resting HRV (ms)
  - `hrv_effort`: Minimum HRV during activity (ms)
  - `hrv_recovery`: HRV after 5-min recovery (ms)
  - `activity_borg`: Original Borg label (for validation)

---

### 2. Updated `run_pipeline.py`

**Changes**:
- Added import: `from phases.phase5_alignment.compute_hrv_recovery import align_windows_to_hrv_recovery`
- Modified fusion/alignment section to use HRV recovery instead of Borg
- Reads windows, PPG, and ADL data
- Calls `align_windows_to_hrv_recovery()` to compute recovery rates
- Merges HRV labels with fused features
- **Fallback**: If HRV computation fails, reverts to Borg-based `run_alignment()`

**Key Output**: `fused_aligned_*.csv` now includes `hrv_recovery_rate` column instead of (or in addition to) `borg`

---

### 3. Updated `phases/phase6_feature_selection/feature_selection_and_qc.py`

**Changes**:
- Auto-detects target variable: prefers `hrv_recovery_rate` if available, falls back to `borg`
- Updated metadata skip columns to include HRV-related fields:
  - `hrv_recovery_rate`, `hrv_baseline`, `hrv_effort`, `hrv_recovery`, `activity_borg`
- Feature selection and PCA use detected target variable
- Maintains backward compatibility (works with both targets)

**Behavior**:
```python
if 'hrv_recovery_rate' in df.columns:
    target_col = 'hrv_recovery_rate'
    print("✓ Using HRV Recovery Rate as target (primary)")
else:
    target_col = 'borg'
    print("✓ Using Borg RPE Scale as target (fallback)")
```

---

### 4. Updated `run_pipeline_complete.py` (Multi-subject)

**Changes**:
- Auto-detects target variable in combined dataset
- Updated all count/label statements to use `target_col` variable
- Feature selection runs with detected target
- Summary statistics show which target was used

**Impact**: Multi-subject model training now uses HRV recovery if available across all 3 subjects

---

## Expected Outcomes

### What Changes
- **Target variable**: From subjective Borg (0-10) to objective HRV Recovery Rate (ms/s)
- **Data type**: From discrete integers to continuous floats
- **Interpretability**: Shifts from "perceived effort" to "physiological recovery speed"
- **Training data**: Labeled windows from recovery phases (different distribution than activity phases)

### What Stays the Same
- Feature computation: All 266+ features still extracted from IMU/PPG/EDA/RR
- Pipeline structure: 6-phase modular architecture intact
- Model architecture: XGBoost still used for regression
- Feature selection: 266→100→70 features (or similar ratio)

### Expected Performance Impact
- **Borg-based baseline**: R²=0.9259, RMSE=0.6528, MAE=0.4283 (82% variance explained)
- **HRV recovery expected**: R²=0.93-0.95 (potentially better, fewer confounding factors)
- **Advantages**: More stable across different users, better for fitness monitoring

---

## Testing & Validation

### To Test the Implementation

1. **Run single-subject pipeline**:
   ```bash
   python run_pipeline.py config/pipeline.yaml
   ```
   Check for `fused_aligned_10.0s.csv` with `hrv_recovery_rate` column

2. **Run multi-subject training**:
   ```bash
   python run_pipeline_complete.py
   ```
   Should show HRV recovery rate stats in summary

3. **Validate HRV computation**:
   - Check output logs for "HRV Recovery Rate range" and interpretation
   - Verify recovery rates fall in expected range (0.2-2.0 ms/s)
   - Check that successful activities >50% of ADL count

4. **Compare results**:
   - With Borg: Model metrics from checkpoint `0913ddf`
   - With HRV: New model metrics from this implementation
   - Track: R², RMSE, MAE, feature importance changes

### Rollback Plan

If issues arise, rollback to Borg-based version:
```bash
git checkout 0913ddf  # Checkpoint before HRV implementation
```

---

## Architecture Diagram

```
                             run_pipeline.py
                                    |
                    ________________|________________
                   |                                |
            Preprocessing & Features         Fusion
                 (Phase 1-4)                 (Phase 4)
                   |                                |
                   |________________________________|
                             |
                    fused_features_*.csv
                             |
                    ________________________
                   |
        HRV Recovery Computation
        (compute_hrv_recovery.py)
                   |
        ┌──────────┴──────────┐
        |                     |
      PPG            ADL Intervals
      Data            (Borg labels)
        |                     |
        └──────────┬──────────┘
                   |
        Three-phase HRV Analysis:
        Baseline → Effort → Recovery
                   |
        hrv_recovery_rate (ms/s)
                   |
        Merge with Fused Features
                   |
        fused_aligned_*.csv
        (includes hrv_recovery_rate column)
                   |
        Feature Selection (Phase 6)
        Target detection: hrv_recovery_rate OR borg
                   |
        Training & Model Output
```

---

## File Changes Summary

| File | Changes | Purpose |
|------|---------|---------|
| `phases/phase5_alignment/compute_hrv_recovery.py` | NEW (280 lines) | HRV computation from PPG |
| `run_pipeline.py` | Modified | Integrate HRV recovery alignment |
| `run_pipeline_complete.py` | Modified | Auto-detect target for multi-subject training |
| `phases/phase6_feature_selection/feature_selection_and_qc.py` | Modified | Auto-detect target variable |
| `METHODOLOGY.md` | Updated | Reference HRV recovery approach |

---

## References & Citation

**HRV Recovery Literature**:
1. Buchheit & Gindre (2006) - HRV recovery after exercise training
2. Plews et al. (2012) - Heart rate variability in elite athletes
3. Al Haddad et al. (2011) - HRV monitoring in training and recovery
4. Stanley et al. (2013) - Cardiac parasympathetic response during recovery
5. Tarvainen & Niskanen (2014) - Kubios HRV analysis software

**SDNN (HRV Metric)**:
- Standard Deviation of Normal-to-Normal intervals
- Gold standard for HRV analysis (25-200 ms is typical range)
- Lower SDNN during effort, recovers after rest
- More stable measure than peak-to-peak HRV

---

## Next Steps (Optional Enhancements)

1. **Validation against clinical HRV monitors** (e.g., Polar H10, Garmin)
2. **HRV recovery threshold optimization** for different user groups
3. **Combination model**: Train on both Borg and HRV recovery together
4. **Temporal analysis**: Track HRV recovery rate changes over time (fitness improvements)
5. **Fatigue detection**: High effort + slow recovery = fatigue state

---

## Questions & Troubleshooting

**Q: What if HRV computation fails?**  
A: Pipeline automatically falls back to Borg-based alignment. Check error logs for PPG data quality issues.

**Q: Why 5-minute recovery window?**  
A: Based on HRV literature—takes ~5 min for autonomic nervous system to fully recover after effort.

**Q: Can I use HRV with all 3 PPG channels?**  
A: Yes! Currently using green PPG (standard), but can test infrared or red for comparison.

**Q: How do I switch back to Borg labels?**  
A: Restore checkpoint with `git checkout 0913ddf`, or disable HRV integration in `run_pipeline.py` (comment out the try block).

---

**Status**: ✅ Ready for production testing  
**Tested**: Syntax validation complete  
**Git**: Commit `bb64313`, rollback available at `0913ddf`
