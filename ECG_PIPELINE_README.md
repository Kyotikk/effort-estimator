# ECG-based RMSSD Physiological Effort Label Pipeline

## Overview

This pipeline builds **physiological effort labels from ECG data** using heart rate variability (HRV) metrics, specifically RMSSD (Root Mean Square of Successive Differences in RR intervals). The pipeline is designed for generating ML training labels for effort estimation tasks.

**Pipeline Stages:**
1. **Stage 1 (ECG Preprocessing)**: Raw ECG → clean RR intervals
2. **Stage 2 (RMSSD Windowing)**: RR intervals → windowed RMSSD features
3. **Stage 3 (Effort Labels)**: RMSSD windows → binary/continuous effort labels
4. **Stage 4 (Borg Validation)**: Effort labels → validation vs. Borg RPE ratings

---

## Architecture & Design Principles

### Module Structure
```
signals/
  ├── __init__.py
  └── ecg_preprocess.py          # R-peak detection, RR extraction, artifact removal
  
features/
  ├── __init__.py
  └── ecg_hrv_features.py        # RMSSD windowing, lnRMSSD computation
  
scripts/
  ├── ecg_to_rr.py               # CLI: ECG CSV → RR intervals
  ├── rr_to_rmssd_windows.py     # CLI: RR intervals → RMSSD windows
  ├── rmssd_to_effort_labels.py  # CLI: RMSSD → effort labels (TO IMPLEMENT)
  └── effort_borg_validation.py  # CLI: Borg validation (TO IMPLEMENT)

config/
  └── ecg_processing.yaml        # All configurable parameters
```

### Configuration-Driven Approach
- All parameters in **`config/ecg_processing.yaml`** (no hardcoding except sensible defaults)
- CLI scripts accept parameter overrides via `--` arguments
- Config uses YAML with nested sections for each stage

### Fail-Loudly Design
- Missing input files → explicit error with path shown
- Invalid parameter ranges → validation and clear error messages
- Insufficient data → warning logs and quality flags (not silent failures)
- All warning/error messages go to `logging.ERROR` and `logging.WARNING`

---

## Stage 1: ECG Preprocessing (ECG → RR Intervals)

### Purpose
Convert raw ECG time series into clean RR (R-to-R) intervals with artifact removal and quality metrics.

### Key Algorithm: Pan-Tompkins Style R-Peak Detection
1. **Bandpass filter** (5-40 Hz): Remove baseline drift and noise
2. **Derivative**: Compute dE/dt to find steep transitions
3. **Squaring**: Emphasize peaks (non-linear enhancement)
4. **Window integration**: Accumulate signal energy
5. **Adaptive threshold**: Threshold = 0.5 × max(filtered signal)
6. **Peak picking**: Find local maxima with 300ms minimum distance

### Artifact Removal (Two-Pass)
1. **Physiological bounds**: Remove RR intervals <300ms or >2000ms
2. **Statistical outliers**: Remove RR intervals >1.5×IQR from Q1/Q3

### Input Format (ECG CSV)

```csv
t,ecg
0.000,100.2
0.004,101.5
0.008,102.1
...
```

- **t**: Timestamp (seconds or any monotonic unit, will be converted relative to first sample)
- **ecg**: ECG voltage (mV or arbitrary units; filter is agnostic to scale)
- **Requirements**: Uniform sampling (e.g., 250 Hz)

### Output Format (RR Intervals CSV)

```csv
session_id,peak_index,t_rr,rr_ms,is_valid,reason
session_001,42,0.168,850,True,
session_001,89,0.356,752,True,
session_001,135,0.540,184,False,rr_too_short
session_001,136,0.725,922,True,
...
```

- **session_id**: Identifier for ECG session
- **peak_index**: Index of R-peak in raw ECG array
- **t_rr**: Timestamp of RR interval (seconds, at midpoint of consecutive peaks)
- **rr_ms**: RR interval duration (milliseconds)
- **is_valid**: Boolean; True if passed artifact checks
- **reason**: String; explanation if invalid (e.g., "rr_too_short", "rr_too_long", "iqr_outlier")

### Quality Summary (JSON)

```json
{
  "session_id": "session_001",
  "n_samples": 75000,
  "n_peaks_detected": 150,
  "n_rr_intervals": 149,
  "n_rr_valid": 145,
  "n_rr_artifact": 4,
  "pct_artifact": 2.7,
  "rr_mean_ms": 847.3,
  "rr_std_ms": 42.1,
  "rr_min_ms": 701,
  "rr_max_ms": 998,
  "sampling_rate_hz": 250,
  "processing_notes": "OK"
}
```

### CLI Example

```bash
# Basic usage
python scripts/ecg_to_rr.py \
  --ecg-csv data/raw/ecg_session_001.csv \
  --output-rr data/interim/rr_session_001.csv \
  --sampling-rate 250 \
  --session-id session_001

# Verbose output
python scripts/ecg_to_rr.py \
  --ecg-csv data/raw/ecg_session_001.csv \
  --output-rr data/interim/rr_session_001.csv \
  --output-quality data/interim/rr_quality_session_001.json \
  --sampling-rate 250 \
  --session-id session_001 \
  --verbose

# Custom RR bounds (for athletes with very low heart rate)
python scripts/ecg_to_rr.py \
  --ecg-csv data/raw/ecg_athlete.csv \
  --output-rr data/interim/rr_athlete.csv \
  --sampling-rate 250 \
  --session-id athlete_001 \
  --min-rr 400 \
  --max-rr 2500
```

### Assumptions & Limitations

- **ECG uniform sampling**: Code assumes constant sampling rate (e.g., 250 Hz)
- **R-peak polarity**: R-peaks are positive deflections in ECG (standard configuration)
- **Physiological RR range**: Default 300-2000 ms (configurable for edge cases)
- **High-quality ECG**: Algorithm works best with ECG sampling ≥200 Hz; lower rates degrade R-peak detection
- **Single-lead ECG OK**: Works with any single ECG lead (typically II or modified V5)

---

## Stage 2: RMSSD Windowing (RR Intervals → Windowed RMSSD)

### Purpose
Compute windowed RMSSD (Root Mean Square of Successive Differences) features from RR intervals for time-series analysis.

### Key Metrics

**RMSSD** (Heart Rate Variability):
$$\text{RMSSD} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n-1} (RR_i - RR_{i+1})^2}$$

where $RR_i$ are successive RR intervals in milliseconds.

**lnRMSSD** (Log-transformed):
$$\text{lnRMSSD} = \ln(\text{RMSSD})$$

Interpretation:
- **Higher RMSSD**: Greater parasympathetic (vagal) tone, lower physiological stress
- **Lower RMSSD**: Reduced vagal tone, physiological stress or high sympathetic drive (typical during effort)

### Windowing Strategy

**Time-based windows** with configurable overlap:
- **Window length**: 60 seconds (default)
- **Overlap**: 50% (default) → step size = 30 seconds
- **Example**: Windows [0-60s], [30-90s], [60-120s], ...

Quality flags:
- **n_rr_valid**: Count of valid RR intervals in window
- **frac_valid**: Fraction of RR intervals marked as valid in original RR CSV
- **Sparse window flag**: Window with <3 valid RR intervals gets `rmssd=NaN`

### Input Format (RR Intervals CSV from Stage 1)

```csv
session_id,peak_index,t_rr,rr_ms,is_valid,reason
session_001,42,0.168,850,True,
session_001,89,0.356,752,True,
session_001,135,0.540,922,True,
...
```

### Output Format (RMSSD Windows CSV)

```csv
session_id,window_id,t_start,t_end,t_center,n_rr_total,n_rr_valid,frac_valid,rmssd,ln_rmssd
session_001,0,0.0,60.0,30.0,45,43,0.956,65.2,4.179
session_001,1,30.0,90.0,60.0,46,45,0.978,72.1,4.279
session_001,2,60.0,120.0,90.0,47,46,0.979,68.3,4.225
session_001,3,90.0,150.0,120.0,45,44,0.978,71.5,4.271
...
```

- **session_id**: From original RR intervals
- **window_id**: Sequential window number (0-indexed)
- **t_start, t_end**: Window time boundaries (seconds)
- **t_center**: Window center time (used for alignment with other modalities)
- **n_rr_total**: Total RR intervals falling in window (including invalid)
- **n_rr_valid**: Valid RR intervals only
- **frac_valid**: Ratio n_rr_valid / n_rr_total
- **rmssd**: RMSSD in milliseconds (NaN if <3 valid RR intervals)
- **ln_rmssd**: Natural log of RMSSD (NaN if RMSSD is NaN)

### CLI Example

```bash
# Basic usage (60s windows, 50% overlap)
python scripts/rr_to_rmssd_windows.py \
  --rr-csv data/interim/rr_session_001.csv \
  --output-windows data/interim/rmssd_windows_session_001.csv

# Custom window length (5-minute windows)
python scripts/rr_to_rmssd_windows.py \
  --rr-csv data/interim/rr_session_001.csv \
  --output-windows data/interim/rmssd_windows_5min.csv \
  --window-length 300 \
  --overlap 0.5 \
  --min-rr-per-window 5

# Batch processing all sessions
for rr_file in data/interim/rr_*.csv; do
  session=$(basename "$rr_file" .csv)
  python scripts/rr_to_rmssd_windows.py \
    --rr-csv "$rr_file" \
    --output-windows "data/interim/rmssd_${session}.csv" \
    --window-length 60 \
    --overlap 0.5
done
```

### Assumptions & Limitations

- **Sufficient RR count**: Requires ≥3 valid RR intervals per window for valid RMSSD
  - Sparse sessions may produce many NaN values → check `frac_valid` column
- **Time continuity**: Windows are created as sliding intervals; gaps in RR data still produce windows (but with NaN RMSSD if insufficient data)
- **Stationary assumption**: RMSSD assumes RR intervals are quasi-stationary within 60s windows; longer-term trends need post-windowing analysis
- **No artifact re-weighting**: RMSSD uses only `is_valid=True` RR intervals; invalid intervals are excluded (not re-weighted)

---

## Expected Workflow: End-to-End Example

### Setup
```bash
# Create data directories
mkdir -p data/raw data/interim data/processed

# Copy raw ECG CSV to data/raw/
# (See below for synthetic example generation)
cp /path/to/raw_ecg.csv data/raw/ecg_session_001.csv
```

### Run Pipeline

```bash
# STAGE 1: ECG → RR Intervals
python scripts/ecg_to_rr.py \
  --ecg-csv data/raw/ecg_session_001.csv \
  --output-rr data/interim/rr_session_001.csv \
  --output-quality data/interim/rr_quality_session_001.json \
  --sampling-rate 250 \
  --session-id session_001 \
  --verbose

# STAGE 2: RR Intervals → RMSSD Windows
python scripts/rr_to_rmssd_windows.py \
  --rr-csv data/interim/rr_session_001.csv \
  --output-windows data/interim/rmssd_windows_session_001.csv \
  --window-length 60 \
  --overlap 0.5 \
  --verbose

# STAGE 3: RMSSD Windows → Effort Labels (TO IMPLEMENT)
# python scripts/rmssd_to_effort_labels.py \
#   --rmssd-csv data/interim/rmssd_windows_session_001.csv \
#   --output-labels data/processed/effort_labels_session_001.csv \
#   --label-strategy rmssd_recovery

# STAGE 4: Effort Labels → Borg Validation (TO IMPLEMENT)
# python scripts/effort_borg_validation.py \
#   --effort-csv data/processed/effort_labels_session_001.csv \
#   --borg-csv data/raw/borg_ratings_session_001.csv \
#   --output-validation data/processed/validation_session_001.json
```

### Inspect Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Check RR intervals
rr_df = pd.read_csv('data/interim/rr_session_001.csv')
print(f"Total RR intervals: {len(rr_df)}")
print(f"Valid RR: {rr_df['is_valid'].sum()}")
print(f"Mean RR: {rr_df[rr_df['is_valid']]['rr_ms'].mean():.1f} ms")

# Check RMSSD windows
rmssd_df = pd.read_csv('data/interim/rmssd_windows_session_001.csv')
print(f"\nRMSSD Windows: {len(rmssd_df)}")
print(f"Windows with valid RMSSD: {rmssd_df['rmssd'].notna().sum()}")
print(f"RMSSD range: {rmssd_df['rmssd'].min():.1f} - {rmssd_df['rmssd'].max():.1f} ms")

# Plot RMSSD over time
plt.figure(figsize=(12, 4))
plt.plot(rmssd_df['t_center'], rmssd_df['rmssd'], 'o-', label='RMSSD')
plt.xlabel('Time (s)')
plt.ylabel('RMSSD (ms)')
plt.legend()
plt.title(f'RMSSD Time Series - {rr_df.iloc[0]["session_id"]}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/interim/rmssd_timeseries.png', dpi=100)
```

---

## Testing: Synthetic Data Generation

To test the pipeline without real ECG data, generate synthetic ECG:

```python
import numpy as np
import pandas as pd

def generate_synthetic_ecg(duration_s=300, sampling_rate_hz=250, heart_rate_bpm=70):
    """Generate synthetic ECG with realistic R-peak morphology."""
    t = np.arange(0, duration_s, 1/sampling_rate_hz)
    n_samples = len(t)
    
    # Baseline (P-QRS-T complex repeated)
    rr_interval_s = 60 / heart_rate_bpm
    n_peaks = int(duration_s / rr_interval_s)
    peak_times = np.array([i * rr_interval_s for i in range(n_peaks)])
    
    ecg = np.zeros(n_samples)
    for peak_t in peak_times:
        peak_idx = int(peak_t * sampling_rate_hz)
        if peak_idx < n_samples:
            # R-peak (positive deflection)
            ecg[max(0, peak_idx-10):min(n_samples, peak_idx+15)] += np.hanning(25)[:, None].flatten()
    
    # Add noise
    ecg += np.random.normal(0, 0.05, n_samples)
    
    # Normalize
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min()) * 100
    
    df = pd.DataFrame({'t': t, 'ecg': ecg})
    return df

# Generate and save
df_ecg = generate_synthetic_ecg(duration_s=300, sampling_rate_hz=250, heart_rate_bpm=75)
df_ecg.to_csv('data/raw/ecg_synthetic_session_001.csv', index=False)
print(f"Generated synthetic ECG: {len(df_ecg)} samples at 250 Hz")
```

---

## Performance & Scalability

| Pipeline Stage | Input | Output | Typical Runtime |
|---|---|---|---|
| Stage 1 (ECG→RR) | 300s ECG @ 250 Hz (75K samples) | ~150 RR intervals | ~0.5 sec |
| Stage 2 (RR→RMSSD) | ~150 RR intervals | ~10 windows (60s length) | ~0.05 sec |
| Batch (10 sessions) | 10 × ECG CSV | 10 × RMSSD CSV | ~15 sec |

**Memory**: Negligible (all DataFrames <1MB for typical sessions)

---

## Known Limitations & Future Work

### Current Limitations
1. **Single-session processing**: Scripts process one session at a time (easy to batch externally)
2. **No automatic parameter tuning**: R-peak thresholds are fixed; may need manual tuning for unusual ECG morphologies
3. **No motion artifact handling**: Assumes ECG is relatively clean; doesn't handle large baseline shifts from motion
4. **RMSSD alone**: HRV has many components (frequency domain features, entropy, etc.); RMSSD captures only time-domain parasympathetic tone

### Stage 3 & 4 (To Implement)
- **Effort label strategy**: Decision function from RMSSD windows to binary/continuous effort labels
- **Borg validation**: Alignment with Borg RPE ratings and correlation analysis
- **Confidence scoring**: Per-window confidence in effort classification based on RR quality

---

## Configuration Reference

All parameters are in **`config/ecg_processing.yaml`**. Key sections:

```yaml
stage_1_ecg_preprocessing:
  r_peak_detection:
    bandpass_low_hz: 5        # Tunable: lower = more baseline drift
    bandpass_high_hz: 40      # Tunable: higher = more high-freq noise
    threshold_factor: 0.5     # Tunable: 0.3-0.7; lower = more detections
    
  artifact_removal:
    min_rr_ms: 300            # Tunable: lower for very high HR
    max_rr_ms: 2000           # Tunable: higher for very low HR
    outlier_iqr_multiplier: 1.5  # Standard (1.5 = default, 3.0 = lenient)

stage_2_rmssd_windowing:
  windowing:
    window_length_s: 60       # Tunable: 30, 60, 300 (seconds)
    overlap_frac: 0.5         # Tunable: 0.0-0.75
    
  rmssd:
    min_rr_per_window: 3      # Tunable: 3-10 for robustness
```

---

## Support & Troubleshooting

### Issue: "No R-peaks detected"
- Check: ECG has sufficient amplitude (not flat line)
- Try: Lower `threshold_factor` in config (default 0.5 → try 0.3)
- Verify: ECG column name matches `--ecg-column` argument

### Issue: "Too many artifacts removed"
- Check: Heart rate is physiological (30-200 bpm)
- Try: Adjust `--min-rr` and `--max-rr` to match expected heart rate
- Example: If expected HR ~90 bpm, then RR ~ 667 ms; set `--min-rr 500 --max-rr 1000`

### Issue: "RMSSD windows are mostly NaN"
- Check: Input RR CSV has many `is_valid=False` entries
- Try: Revisit Stage 1 quality (plot ECG, verify R-peak detection visually)
- Try: Increase `--min-rr-per-window` threshold if window has sufficient data

### Issue: "FileNotFoundError: data/raw/..."
- Verify input file exists: `ls -la data/raw/`
- Use absolute path if relative path doesn't work
- Check file permissions: `chmod 644 data/raw/*.csv`

---

## References

- **Pan-Tompkins R-Peak Detection**: Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm." *IEEE Trans. Biomed. Eng.*, 32(3), 230-236.
- **RMSSD in HRV Analysis**: Task Force Report (1996). "Heart rate variability: Standards of measurement, physiological interpretation, and clinical use." *Eur. Heart J.*, 17(3), 354-381.
- **Effort Detection from HRV**: Applications in exercise physiology (e.g., detecting recovery vs. active effort from RMSSD dynamics)

---

## Files in This Module

- **`signals/ecg_preprocess.py`**: ECG signal processing (R-peak detection, RR extraction, artifact removal)
- **`features/ecg_hrv_features.py`**: RMSSD windowing and feature computation
- **`scripts/ecg_to_rr.py`**: CLI for Stage 1 (ECG → RR)
- **`scripts/rr_to_rmssd_windows.py`**: CLI for Stage 2 (RR → RMSSD)
- **`config/ecg_processing.yaml`**: Configuration template
- **`ECG_PIPELINE_README.md`**: This file

