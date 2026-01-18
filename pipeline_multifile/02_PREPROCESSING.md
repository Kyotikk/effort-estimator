# Stage 2: Preprocessing

## Purpose

Convert raw sensor signals into clean, normalized time series ready for windowing and feature extraction.

---

## 2.1 IMU Preprocessing (Chest & Wrist)

**File**: `preprocessing/imu.py`
**Function**: `preprocess_imu(df, cfg)`

### Input
- Raw acceleration data (X, Y, Z axes)
- Sampling rate: 32 Hz
- Format: CSV with columns `time`, `x`, `y`, `z`

### Processing Steps

**Step 1: Load Data**
```python
df = pd.read_csv(imu_path)
# Columns: time, x, y, z
# Example: 2024-10-15 09:23:45.123, -0.045, 0.987, -0.156
```

**Step 2: Gravity Removal (High-Pass Filter)**
- Butterworth IIR high-pass filter
- Cutoff frequency: 0.3-0.5 Hz
- Purpose: Removes DC bias and static gravity (9.8 m/s²)
- Formula: $a_{dynamic} = a_{raw} - a_{gravity}$

```python
from scipy.signal import butter, filtfilt

b, a = butter(4, 0.3, btype='high', fs=32)
x_filtered = filtfilt(b, a, df['x'].values)
y_filtered = filtfilt(b, a, df['y'].values)
z_filtered = filtfilt(b, a, df['z'].values)
```

**Step 3: Normalization (to 'g' units)**
- Divide by gravitational acceleration (9.81 m/s²)
- Scales to ±1g typical range
- Makes values unit-independent

```python
G = 9.81
x_normalized = x_filtered / G
y_normalized = y_filtered / G
z_normalized = z_filtered / G
```

**Step 4: Timestamp Conversion**
- Convert to Unix seconds (UTC)
- Handles various timestamp formats
- Ensures temporal consistency

**Step 5: Resampling (if needed)**
- Target frequency: 50 Hz (original 32 Hz)
- Linear interpolation
- Ensures consistent time spacing

### Output
- File: `{imu_type}_preprocessed.csv`
- Columns: `t_sec` (Unix seconds), `acc_x_dyn`, `acc_y_dyn`, `acc_z_dyn`
- Format: CSV, one row per sample
- Sampling rate: 32-50 Hz (consistent)

### Quality Checks
✅ No NaN values
✅ Sampling rate verified
✅ Timestamp monotonically increasing
✅ Acceleration magnitudes reasonable (typically ±0.2g for normal movement)

---

## 2.2 PPG Preprocessing (Green, IR, Red)

**File**: `preprocessing/ppg.py`
**Function**: `preprocess_ppg(df, cfg)`

### Input
- Raw PPG signal from LED
- Sampling rate: 32 Hz
- Format: CSV with columns `time`, `value`
- Range: Typically 0-4095 (12-bit ADC)

### Processing Steps

**Step 1: Load Data**
```python
df = pd.read_csv(ppg_path)
# Columns: time, value
# Example: 2024-10-15 09:23:45.123, 2048
```

**Step 2: High-Pass Filter (motion artifact removal)**
- Butterworth IIR high-pass filter
- Cutoff frequency: 0.5 Hz
- Purpose: Removes DC offset and very low frequency noise
- Condition: Applied differently per channel
  - Green: No HPF (preserve signal)
  - IR & Red: Apply HPF (remove artifacts)

```python
# For IR/Red only
b, a = butter(4, 0.5, btype='high', fs=32)
value_filtered = filtfilt(b, a, df['value'].values)
```

**Step 3: Motion Artifact Handling**
- Detect physiologically implausible jumps
- Forward-fill or interpolate over short gaps
- Keep original value if gap < 1 second

**Step 4: Timestamp Conversion**
- Convert to Unix seconds (UTC)
- Verify monotonic increase
- Handle missing samples (interpolate)

**Step 5: Resampling**
- Target frequency: 8 Hz (original 32 Hz)
- Reason: PPG signal is low-frequency (heart rate ~60-100 bpm = 1-1.7 Hz)
- Method: Linear interpolation

### Output
- Files: `ppg_{color}_preprocessed.csv` (green, infra, red)
- Columns: `t_sec`, `value` (normalized PPG signal)
- Sampling rate: 8 Hz
- Range: Typically -1000 to +1000 (after HPF)

### Quality Checks
✅ Signal amplitude reasonable (motion artifacts removed)
✅ No clipping at ADC limits
✅ Missing value gaps < 5 seconds
✅ Timestamp continuity verified

---

## 2.3 RR Preprocessing (Heart Rate Variability)

**File**: `preprocessing/ecg.py` or `preprocessing/rr.py`
**Function**: `preprocess_rr(df, cfg)`

### Input
- RR interval data (time between heartbeats)
- Format: CSV with columns `time`, `rr`
- RR values: milliseconds (typical 500-2000 ms)
- Sampling rate: Variable (typically 0.5-2 Hz)

### Processing Steps

**Step 1: Load Data**
```python
df = pd.read_csv(rr_path)
# Columns: time, rr (in milliseconds)
# Example: 2024-10-15 09:23:45.123, 750
```

**Step 2: Physiological Validation**
- Remove outliers outside 300-2000 ms (heart rate 30-200 bpm)
- Detect and mark missing/invalid values

```python
valid_mask = (df['rr'] >= 300) & (df['rr'] <= 2000)
df_valid = df[valid_mask].copy()
```

**Step 3: Interpolation**
- Linear interpolation for gaps up to 10 seconds
- Maintains temporal continuity
- Preserves RR interval relationships

**Step 4: Resampling to 1 Hz**
- Convert to regular 1 Hz sampling
- Preserves RR interval information
- Easier for feature extraction

**Step 5: Frequency Conversion (optional)**
- Compute respiration frequency: $f = 1 / RR_{seconds}$
- Alternatively: Store RR intervals as-is

### Output
- File: `rr_preprocessed.csv`
- Columns: `t_sec`, `rr_interval_ms` (or `respiratory_freq_hz`)
- Sampling rate: 1 Hz
- Range: 300-2000 ms typical

### Quality Checks
✅ All values in physiological range
✅ No extreme outliers
✅ Missing value gaps < 10 seconds
✅ Timestamp ordered

---

## 2.4 EDA Preprocessing

**File**: `preprocessing/eda.py`
**Function**: `preprocess_eda(df, cfg)`

### Input
- Electrodermal activity (skin conductance)
- Sampling rate: 32 Hz
- Format: CSV with columns `time`, `value`
- Range: Typically 0-100 µS (microsiemens)

### Processing Steps

**Step 1: Load Data**
```python
df = pd.read_csv(eda_path)
# Columns: time, value
# Example: 2024-10-15 09:23:45.123, 2.5
```

**Step 2: Smoothing (Savitzky-Golay Filter)**
- Window length: 11 samples
- Polynomial order: 3
- Purpose: Remove measurement noise, preserve signal features

```python
from scipy.signal import savgol_filter

value_smooth = savgol_filter(df['value'], window_length=11, polyorder=3)
```

**Step 3: Outlier Detection**
- Mark values > mean + 5×std as outliers
- Forward-fill or interpolate

**Step 4: Resampling to 2 Hz**
- EDA is slow-varying signal (main frequency < 1 Hz)
- 2 Hz sufficient for feature extraction

**Step 5: Component Extraction (optional)**
- Tonic component (slow drift): baseline conductance
- Phasic component (fast peaks): stress responses
- Methods: Deconvolution, median filtering

### Output
- File: `eda_preprocessed.csv`
- Columns: `t_sec`, `scl_tonic`, `scl_phasic` (or just `value`)
- Sampling rate: 2 Hz
- Range: 0-100 µS typical

### Quality Checks
✅ Values in normal physiological range
✅ Smooth without clipping
✅ No NaN values
✅ Baseline stable

---

## 2.5 Configuration

**File**: `config/pipeline.yaml`

```yaml
preprocessing:
  imu_bioz:
    gravity_cutoff: 0.5        # Hz
    target_freq: 50            # Hz (if resampling)
    apply_norm: true           # Normalize to 'g'
  
  imu_wrist:
    gravity_cutoff: 0.5
    target_freq: 50
    apply_norm: true
  
  ppg_green:
    apply_hpf: false           # Keep DC
    hpf_cutoff: 0.5           # Hz (if applied)
    target_freq: 8            # Hz
  
  ppg_infrared:
    apply_hpf: true           # Remove motion artifacts
    hpf_cutoff: 0.5
    target_freq: 8
  
  ppg_red:
    apply_hpf: true
    hpf_cutoff: 0.5
    target_freq: 8
  
  eda:
    smooth_window: 11         # samples
    smooth_polyorder: 3       # polynomial order
    target_freq: 2            # Hz
  
  rr:
    valid_range: [300, 2000]  # milliseconds
    target_freq: 1            # Hz
```

---

## Summary

| Modality | Input | Output | Key Processing |
|----------|-------|--------|-----------------|
| **IMU** | Raw accel 32 Hz | Filtered 32-50 Hz | Gravity removal + HPF |
| **PPG** | Raw signal 32 Hz | Filtered 8 Hz | HPF (some channels) + resample |
| **EDA** | Raw conductance 32 Hz | Smoothed 2 Hz | Savitzky-Golay + resample |
| **RR** | RR intervals (var Hz) | Regular 1 Hz | Validation + interpolation |

**Output**: All modalities cleaned, normalized, resampled to consistent rates → Ready for windowing!
