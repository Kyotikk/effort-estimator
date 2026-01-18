# Stage 3: Feature Extraction

## Overview

Feature extraction computes mathematical features from windowed sensor signals. These features summarize signal properties (mean, frequency content, morphology, etc.) that correlate with effort level.

**Total features:** 188 across all modalities

---

## Feature Extraction by Modality

### IMU Features: 30 features

**Extracted from:** Dynamic acceleration (gravity-removed) in 3 axes

**Source file:** `features/manual_features_imu.py`

**Feature categories:**

#### Acceleration Statistics (9 features, 3 per axis)
```
Per axis (x, y, z):
  - imu_acc_X_mean       Mean acceleration
  - imu_acc_X_std        Standard deviation
  - imu_acc_X_rms        Root mean square (energy)
```

#### Energy & Magnitude (6 features)
```
  - imu_magnitude_mean   Acceleration magnitude √(x²+y²+z²) - mean
  - imu_magnitude_std    Acceleration magnitude - std
  - imu_acceleration_energy  Total kinetic energy
  - imu_jerk_x           Change in acceleration over time (x-axis)
  - imu_jerk_y           Change in acceleration over time (y-axis)
  - imu_jerk_z           Change in acceleration over time (z-axis)
```

#### Frequency-Domain Features (9 features)
```
  - imu_spectral_entropy         Shannon entropy of frequency spectrum
  - imu_dominant_frequency       Peak frequency in FFT
  - imu_power_low_freq           Power 0-1 Hz (postural)
  - imu_power_mid_freq           Power 1-3 Hz (moderate movement)
  - imu_power_high_freq          Power 3-32 Hz (high activity)
  - imu_freq_peak_magnitude      Magnitude at dominant frequency
  - imu_zero_crossing_rate       Rate of signal crosses zero
  - imu_autocorrelation_lag1     Lag-1 autocorrelation
  - imu_signal_to_noise_ratio    SNR estimate
```

#### Higher-Order Statistics (3 features)
```
  - imu_skewness         Asymmetry of distribution
  - imu_kurtosis         Tail heaviness
  - imu_entropy          Shannon entropy of values
```

#### Dynamic Features (3 features)
```
  - imu_mean_absolute_change     Average change magnitude
  - imu_interquartile_range      Q3 - Q1
  - imu_peak_to_peak             Max - Min
```

**Example interpretation:**
- **High magnitude + high jerk** → Active movement (exercise)
- **Low magnitude + stable** → Rest or light activity
- **High entropy + broad spectrum** → Irregular movement
- **High low-freq power** → Posture shift

---

### PPG Features: 44 features per variant × 3 = 132 total

**Extracted from:** Photoplethysmogram voltage (1 per PPG variant)

**Source file:** `features/ppg_features.py`

**Feature categories:**

#### Heart Rate Metrics (6 features)
```
  - ppg_X_hr_mean              Average heart rate (beats/min)
  - ppg_X_hr_std               Heart rate variability (beat-to-beat)
  - ppg_X_hr_min               Minimum HR in window
  - ppg_X_hr_max               Maximum HR in window
  - ppg_X_hrv_rmssd            Root mean square of successive RR intervals
  - ppg_X_hrv_sdnn             Standard deviation of all RR intervals
```

#### Spectral Features (8 features)
```
  - ppg_X_freq_peak            Dominant frequency (Hz)
  - ppg_X_freq_peak_power      Power at dominant frequency
  - ppg_X_power_vlf            Very low freq power (0.0-0.04 Hz)
  - ppg_X_power_lf             Low freq power (0.04-0.15 Hz)
  - ppg_X_power_hf             High freq power (0.15-0.4 Hz)
  - ppg_X_spectral_entropy     Entropy of power spectrum
  - ppg_X_peak_frequency_width Bandwidth at peak
  - ppg_X_relative_power_hf    HF power / total power ratio
```

#### Morphological Features (12 features)
```
  - ppg_X_mean               Signal mean (DC level)
  - ppg_X_std                Signal std (AC amplitude)
  - ppg_X_min                Minimum value
  - ppg_X_max                Maximum value
  - ppg_X_range              Max - Min
  - ppg_X_median             Median value
  - ppg_X_iqr                Interquartile range (Q3-Q1)
  - ppg_X_mad                Median absolute deviation
  - ppg_X_p95_p5             95th percentile - 5th percentile
  - ppg_X_p90_p10            90th percentile - 10th percentile
  - ppg_X_skewness           Asymmetry of distribution
  - ppg_X_kurtosis           Tail heaviness
```

#### Dynamics Features (8 features)
```
  - ppg_X_zero_crossing_rate       Rate signal crosses mean value
  - ppg_X_mean_cross_rate          Rate signal crosses mean value (alternative)
  - ppg_X_slope_mean               Mean slope of signal
  - ppg_X_slope_std                Std of slopes (variability)
  - ppg_X_energy                   Sum of squared values
  - ppg_X_entropy                  Shannon entropy of values
  - ppg_X_peak_to_peak             Max - Min (waveform amplitude)
  - ppg_X_mean_absolute_change     Average absolute change per sample
```

#### Advanced Morphology (10 features)
```
  - ppg_X_tke_p95_abs              Top-k kurtosis estimate
  - ppg_X_autocorr_lag1            Lag-1 autocorrelation
  - ppg_X_autocorr_lag5            Lag-5 autocorrelation
  - ppg_X_cross_rate_above_mean    % samples above mean
  - ppg_X_crest_factor             Peak / RMS ratio
  - ppg_X_impulse_factor           Peak / mean abs value
  - ppg_X_margin_factor            Peak / sqrt(mean absolute)
  - ppg_X_shape_factor             RMS / mean absolute
  - ppg_X_clearance_factor         Peak / cube root of mean cubed
  - ppg_X_peak_count               Number of local maxima
```

**Note:** "X" = green, infra, or red

**Example interpretation:**
- **High HR mean** → Elevated heart rate (effort/stress)
- **High HRV (RMSSD)** → Parasympathetic activity (recovery)
- **Low HRV** → Sympathetic dominance (stress/exertion)
- **High spectral entropy** → Irregular heart rhythm
- **Strong peak at 1-2 Hz** → Clean cardiac signal

---

### EDA Features: 26 features

**Extracted from:** Two EDA channels (continuous conductance + phasic stress)

**Source file:** `features/eda_features.py`

**Channels:**
1. `eda_cc` - Continuous conductance (tonic + phasic combined)
2. `eda_stress_skin` - Phasic stress response (fast changes)

**Features per channel (13 each):**

#### Level & Amplitude (4 features per channel = 8 total)
```
  - eda_X_mean               Mean conductance level (µS)
  - eda_X_std                Conductance variability
  - eda_X_min                Minimum conductance
  - eda_X_max                Maximum conductance
```

#### Range & Spread (4 features per channel = 8 total)
```
  - eda_X_range              Max - Min
  - eda_X_median             Median conductance
  - eda_X_iqr                Interquartile range
  - eda_X_mad                Median absolute deviation
```

#### Dynamics & Slope (3 features per channel = 6 total)
```
  - eda_X_mean_abs_diff      Average absolute change (differencing)
  - eda_X_slope              Overall slope (start to end)
  - eda_X_entropy            Shannon entropy of values
```

**Example interpretation:**
- **High eda_cc level** → High baseline arousal
- **High eda_stress_skin range** → Strong phasic responses (effort events)
- **Steep eda_stress_skin slope** → Increasing stress over window
- **Low eda_X_entropy** → Stable stress state (no response)

---

## Feature Count Summary

```
Modality            Features    Features/Window    Active?
────────────────────────────────────────────────────────
IMU                   30             30           ✓ Yes
PPG Green             44             44           ✓ Yes
PPG Infra             44             44           ✓ Yes
PPG Red               44             44           ✓ Yes
EDA                   26             26           ✓ Yes
RR (Respiratory)       0              0           ✗ No (non-uniform)
────────────────────────────────────────────────────────
TOTAL:               188            188           ✓
```

---

## Feature Extraction Process

**Pseudocode:**
```python
def extract_features(signal_window, fs, modality):
    """
    signal_window: ndarray of signal values in window
    fs: sampling frequency (Hz)
    modality: 'imu', 'ppg', 'eda', etc.
    
    Returns: dict of feature_name → value
    """
    features = {}
    
    # Time-domain statistics
    features['mean'] = np.mean(signal_window)
    features['std'] = np.std(signal_window)
    features['min'] = np.min(signal_window)
    features['max'] = np.max(signal_window)
    ...
    
    # Frequency domain
    fft = np.fft.fft(signal_window)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(signal_window), 1/fs)
    features['freq_peak'] = freqs[np.argmax(power)]
    ...
    
    # Morphology
    peaks, _ = signal.find_peaks(signal_window)
    features['peak_count'] = len(peaks)
    ...
    
    # Other statistics
    features['entropy'] = entropy(signal_window)
    features['zero_cross_rate'] = np.mean(np.diff(signal_window > 0))
    ...
    
    return features
```

---

## Feature Extraction Output

After feature extraction:

```
effort_estimation_output/parsingsim3_sim_elderly3/
├── imu_bioz/
│   ├── imu_features_10.0s.csv      [429 rows, 31 cols: window_id + 30 features]
│   ├── imu_features_5.0s.csv
│   └── imu_features_2.0s.csv
├── ppg_green/
│   ├── ppg_green_features_10.0s.csv [429 rows, 45 cols: window_id + 44 features]
│   ├── ppg_green_features_5.0s.csv
│   └── ppg_green_features_2.0s.csv
├── ppg_infra/
│   ├── ppg_infra_features_10.0s.csv
│   └── [5.0s, 2.0s variants]
├── ppg_red/
│   ├── ppg_red_features_10.0s.csv
│   └── [5.0s, 2.0s variants]
└── eda/
    ├── eda_features_10.0s.csv      [429 rows, 27 cols: window_id + 26 features]
    └── [5.0s, 2.0s variants]
```

**Example feature CSV:**
```
window_id,imu_acc_x_mean,imu_acc_x_std,...,imu_entropy
0,0.042,0.089,...,2.314
1,-0.031,0.095,...,2.401
...
```

---

## Why These Features?

### Time-Domain Features
- **Robust to frequency shifts** (HR changes don't break them)
- **Computationally fast** (no FFT needed)
- **Interpretable** (mean, std are intuitive)
- **Used for:** Basic signal statistics

### Frequency-Domain Features
- **Capture rhythm patterns** (HR peak, spectral distribution)
- **Invariant to amplitude scaling** (normalized)
- **Used for:** HRV, autonomic nervous system state

### Morphological Features
- **Capture waveform shape** (peaks, skewness, kurtosis)
- **Discriminate signal types** (clean vs noisy)
- **Used for:** Signal quality assessment

### Entropy Features
- **Measure randomness/complexity**
- **High entropy:** Noisy/irregular signal
- **Low entropy:** Clean/predictable signal
- **Used for:** Pattern recognition

---

## Quality Control

Feature extraction includes QC:

```
✓ No NaN values in output (filled with mean if necessary)
✓ Features are finite (no inf values)
✓ Feature ranges reasonable (min/max within 5σ of mean)
✓ Frequency-domain features only computed if signal adequate length
✓ HR only computed if peaks detected successfully
```

---

## Computation Time

Feature extraction is the slowest stage:

**Time estimates:**
- **IMU (30 features):** ~100ms per window
- **PPG (44 features):** ~150ms per window
- **EDA (26 features):** ~50ms per window

**For 429 × 5 modalities = 2,145 windows:**
- Serial: ~5 minutes
- Parallel (4 cores): ~2 minutes
- **Total for all 3 window lengths:** ~10 minutes

**Optimization:** Caching - if feature file exists, reloaded instantly.

---

## Feature Engineering Insights

### Why PPG Needs 44 Features
PPG signals are complex:
- Heart rate varies (mean HR not enough)
- Heart rate variability carries autonomic info
- Spectral shape depends on noise and signal quality
- Multiple features needed to capture these nuances

### Why EDA Has Fewer Features
EDA is simpler:
- 2 channels (vs 3 for IMU, 1 for PPG)
- Slower dynamics (baseline changes over minutes)
- Features mostly descriptive statistics

### Why IMU Has Moderate Count
IMU measures movement:
- 3 axes of acceleration (9 base features)
- Jerk (change in acceleration) adds info
- Frequency spectrum shows movement type
- 30 features sufficient to characterize activity

---

## Next Step: Alignment & Fusion

These features are now ready to be aligned with Borg effort labels and fused. See [04_ALIGNMENT_AND_FUSION.md](04_ALIGNMENT_AND_FUSION.md).

