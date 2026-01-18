# Stage 1: Preprocessing

## Overview

Preprocessing transforms raw sensor data from multiple sources into clean, uniform time-series ready for windowing and feature extraction.

**Key objectives:**
1. Load raw data from CSV files (compressed .gz format)
2. Validate data integrity (timestamps, value ranges)
3. Apply sensor-specific cleaning (noise removal, baseline subtraction)
4. Resample to uniform target frequency (32 Hz)
5. Handle missing values and outliers

---

## Preprocessing Pipeline by Modality

### 1. IMU (Accelerometer) - `preprocessing/imu.py`

**Input:** Raw triaxial accelerometer data (variable sample rate)

**Steps:**
1. **Load:** Read CSV with columns `time`, `acc_x`, `acc_y`, `acc_z`
2. **Resample:** Interpolate to target frequency (32 Hz)
3. **Remove gravity:** Highpass filter (cutoff 0.3 Hz, 4th-order Butterworth)
   - Gravity ~9.81 m/s² is low-frequency component
   - This preserves dynamic acceleration
4. **Remove noise:** Lowpass filter (cutoff 5 Hz, 4th-order Butterworth)
5. **Output:** Clean dynamic acceleration as 3 columns

**Configuration (pipeline.yaml):**
```yaml
preprocessing:
  imu:
    noise_cutoff: 5.0      # Lowpass cutoff (Hz)
    gravity_cutoff: 0.3    # Highpass cutoff (Hz)
    normalise: false
```

**Output columns:**
- `time` (seconds)
- `acc_x_dyn` (m/s², gravity-removed)
- `acc_y_dyn` (m/s², gravity-removed)
- `acc_z_dyn` (m/s², gravity-removed)

**Example:**
```
time,acc_x_dyn,acc_y_dyn,acc_z_dyn
0.000,-0.042,0.031,0.152
0.031,0.018,-0.065,0.189
0.062,-0.089,0.127,-0.041
...
```

---

### 2. PPG Wrist Sensors (3 variants) - `preprocessing/ppg.py`

**Input:** Raw photoplethysmogram data from 3 different LED positions

**Sensors:**
| Variant | LED Position | Signal ID | Expected Strength | Preprocessing |
|---------|--------------|-----------|-------------------|----------------|
| GREEN | 6 | 0x7e | 8,614 units (baseline) | No HPF |
| INFRA | 22 | 0x7b | 5,024 units (-42%) | HPF 0.5 Hz |
| RED | 182 | 0x7c | 2,731 units (-68%) | HPF 0.5 Hz |

**Steps (for each variant):**
1. **Load:** Read CSV, filter by metric_id and led_pd_pos
2. **Resample:** Interpolate to 32 Hz uniform grid
3. **Apply HPF** (if enabled in config):
   - Butterworth 4th-order highpass at 0.5 Hz
   - Removes baseline drift
   - Enhances weak cardiac pulsations
4. **Output:** Single column (PPG voltage)

**Configuration (pipeline.yaml):**
```yaml
preprocessing:
  ppg_green:
    time_col: time
    metric_id: "0x7e"
    led_pd_pos: 6
    do_resample: true
    apply_hpf: false           # ← GREEN doesn't need HPF

  ppg_infra:
    time_col: time
    metric_id: "0x7b"
    led_pd_pos: 22
    do_resample: true
    apply_hpf: true            # ← Apply HPF
    hpf_cutoff: 0.5

  ppg_red:
    time_col: time
    metric_id: "0x7c"
    led_pd_pos: 182
    do_resample: true
    apply_hpf: true            # ← Apply HPF
    hpf_cutoff: 0.5
```

**Output columns:**
- `t_sec` (seconds from start)
- `value` (PPG signal, voltage units)

**Example:**
```
t_sec,value
0.000,1024
0.031,1031
0.062,1018
0.094,1009
...
```

**Why Different HPF Settings?**

Signal quality analysis revealed:
- **GREEN PPG:** Clean signal with good baseline, 8,614 unit amplitude
- **INFRA PPG:** Noisy signal, 5,024 units, visible baseline drift
- **RED PPG:** Very noisy signal, only 2,731 units, heavy baseline drift

Highpass filtering at 0.5 Hz removes low-frequency drift while preserving cardiac pulsations (~1 Hz for resting HR ~60 bpm, 1 Hz = 60 beats/min).

**Before HPF (RED PPG):**
```
Raw signal amplitude: 2,731 units
Baseline visible: Yes (slow drift)
Cardiac pulsation barely visible: Yes
Signal-to-noise ratio: Very low (~0.5)
```

**After HPF (RED PPG):**
```
Signal amplitude: Still low (~1,500 units)
Baseline: Removed
Cardiac pulsation: Enhanced
Signal-to-noise ratio: Improved (~1.5x)
```

---

### 3. EDA (Electrodermal Activity) - `preprocessing/eda.py`

**Input:** Raw skin conductance data (variable sample rate)

**Steps:**
1. **Load:** Read CSV with columns `time`, `eda_cc` (continuous conductance), `eda_stress_skin`
2. **Resample:** Interpolate to 32 Hz
3. **Baseline subtraction:** Remove tonic (slow) component to isolate phasic (fast) responses
4. **Lowpass filter:** Apply 1 Hz cutoff to smooth
5. **Output:** Two channels (tonic and phasic)

**Configuration (pipeline.yaml):**
```yaml
preprocessing:
  eda:
    time_col: time
    do_resample: true
```

**Output columns:**
- `t_sec` (seconds)
- `eda_cc` (continuous conductance in µS, tonic + phasic)
- `eda_stress_skin` (phasic stress response in µS, after baseline removal)

**Example:**
```
t_sec,eda_cc,eda_stress_skin
0.000,2.541,0.031
0.031,2.543,0.028
0.062,2.548,0.035
...
```

**Two Components:**
- **EDA_CC (tonic):** Slow changes reflecting overall arousal/stress level
- **EDA_Stress_Skin (phasic):** Fast responses to stimulus/effort changes

Both are used for feature extraction.

---

### 4. RR (Respiratory Rate) - `preprocessing/rr.py`

**Input:** Event-based respiratory intervals (non-uniform timestamps)

**Status:** ⚠️ **Infrastructure exists but NOT USED** (non-uniform sampling issue)

**Problem:**
- RR data comes as one timestamp per breath
- Intervals vary (6-8 seconds between breaths at rest)
- Cannot create uniform-frequency time-series easily
- Windowing strategy undefined

**Example raw data:**
```
time,rr
0.0,0.0
6.2,0.16
12.4,0.16
18.9,0.14
...
```

Cannot resample to 32 Hz uniformly - would have large gaps and duplicates.

**Future Solutions:**
1. **Aggregation:** Compute RR statistics per window (mean interval, variability)
2. **Interpolation:** Spline-fit RR curve and resample
3. **Event detection:** Extract breath detection peaks, interpolate phasic signal

Currently, RR features are not computed. Infrastructure prepared for future implementation.

---

## Preprocessing Output Structure

After preprocessing all modalities:

```
effort_estimation_output/parsingsim3_sim_elderly3/
├── imu_bioz/
│   └── imu_preprocessed.csv          [Time, 3 acceleration components]
├── ppg_green/
│   └── ppg_green_preprocessed.csv    [Time, PPG voltage (no HPF)]
├── ppg_infra/
│   └── ppg_infra_preprocessed.csv    [Time, PPG voltage (HPF applied)]
├── ppg_red/
│   └── ppg_red_preprocessed.csv      [Time, PPG voltage (HPF applied)]
├── eda/
│   └── eda_preprocessed.csv          [Time, tonic, phasic]
└── rr/
    └── rr_preprocessed.csv           [Time, RR intervals] (NOT USED YET)
```

---

## Quality Checks

Preprocessing includes optional quality checks:

```
✓ File exists and is readable
✓ Time column is monotonically increasing
✓ No excessive NaN values (>10% removed)
✓ Sampling rate is approximately target (±5%)
✓ Value ranges reasonable (no inf or extreme outliers)
✓ Data length sufficient for windowing
```

---

## Key Decisions & Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **Target 32 Hz** | Common for wearable sensors; sufficient for cardiac/respiratory signals (Nyquist > 20 Hz) |
| **HPF only on RED/INFRA** | GREEN signal clean; HPF adds noise for strong signals |
| **0.5 Hz HPF cutoff** | Balances DC removal with HR preservation (~1-3 Hz for exercise) |
| **Gravity removal for IMU** | Focus on dynamic acceleration; gravity is context-independent |
| **4th-order filters** | Good rolloff (~80 dB/decade) without ringing |
| **No decimation after filtering** | Keep high temporal resolution for short windows (2-10s) |

---

## Performance Impact

Preprocessing adds ~2-5 seconds per modality (total <1 minute for full dataset).

**Preprocessing time:**
- IMU: ~1 sec
- PPG (3 variants): ~3 sec total
- EDA: ~1 sec
- **Total: ~5 seconds**

Caching implemented - if preprocessed CSV exists, it's reloaded (instantaneous).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check path in config/pipeline.yaml |
| Resampling fails | Check time column exists and is numeric |
| HPF creates instability | Reduce filter order or increase cutoff |
| All-NaN output | Data may be corrupt or wrong metric_id |
| Frequency mismatch | Interpolation will handle rate variations ±10% |

