# Stage 4: Feature Extraction

## Purpose

Compute statistical and physiological features from windowed time series data.

**Total Features**: 257 base features across all 7 modalities

---

## 4.1 IMU Features (Bioz & Wrist)

### Features Per Axis (X, Y, Z)

**Statistical Features** (5):
- Mean: $\bar{x} = \frac{1}{N} \sum x_i$
- Standard Deviation: $\sigma = \sqrt{\frac{1}{N}\sum(x_i - \bar{x})^2}$
- Min: $\min(x)$
- Max: $\max(x)$
- Median: $\text{median}(x)$

**Quartile Features** (4):
- Q25 (25th percentile)
- Q50 (median, same as above)
- Q75 (75th percentile)
- IQR (Interquartile Range): $Q75 - Q25$

**Temporal Features** (3):
- Zero-crossings: Count of sign changes
- Mean slope: $\frac{1}{N-1}\sum |x_{i+1} - x_i|$
- Peak count: Number of local maxima

**Energy Features** (3):
- RMS (Root Mean Square): $\sqrt{\frac{1}{N}\sum x_i^2}$
- Total power: $\sum x_i^2$
- Peak-to-peak: $\max(x) - \min(x)$

**Distribution Features** (3):
- Skewness: Asymmetry of distribution
- Kurtosis: "Tailedness" of distribution
- Entropy: $-\sum p_i \log p_i$ (information content)

**Magnitude Features** (5):
- 3D Resultant: $R = \sqrt{x^2 + y^2 + z^2}$
- Mean of magnitude
- Std of magnitude
- Min/Max of magnitude

### Summary: IMU Features

```
Per axis (x, y, z): 23 features each
  • Statistical: 5
  • Quartiles: 4
  • Temporal: 3
  • Energy: 3
  • Distribution: 3
Per IMU: 3 axes × 23 = 69 features

Total:
  • Chest IMU (bioz): 69 features
  • Wrist IMU: 69 features
  • IMU Total: 138 features
```

---

## 4.2 PPG Features (Green, Infrared, Red)

### Per PPG Channel

**Heart Rate Metrics** (6):
- Heart Rate (bpm): From peak detection
- Heart Rate Mean: Over window
- Heart Rate Std: Variability
- Heart Rate Min/Max: Range
- Peak count: Number of heartbeats

**Heart Rate Variability** (5):
- RMSSD: RMS of successive RR differences
- SDNN: Std of RR intervals
- NN50: Count of RR differences > 50ms
- pNN50: Percentage of NN50
- Coefficient of Variation: SDNN / mean RR

**Signal Quality** (3):
- Signal amplitude: Max - Min
- Signal quality score: % valid beats
- SNR estimate: Signal to noise ratio

**Statistical Features** (3):
- Mean of PPG signal
- Std of PPG signal
- Skewness of PPG signal

### Summary: PPG Features

```
Per color channel: 17 features each

Total:
  • PPG Green: 17 features
  • PPG Infrared: 17 features
  • PPG Red: 17 features
  • PPG Total: 51 features
```

---

## 4.3 EDA Features

### Skin Conductance Features (5):

- **SCL (Skin Conductance Level)**: Mean conductance (sympathetic baseline)
- **SCL Std**: Conductance variability
- **SCR (Skin Conductance Response)**: Peak rises and responses
- **Phasic Energy**: Rate of change intensity (stress response)
- **Tonic Level**: Baseline conductance value

### Distribution Features (2):

- Mean of EDA signal
- Std of EDA signal

### Summary: EDA Features

```
Total EDA features: 7
```

---

## 4.4 RR Features (Heart Rate Variability)

### Respiration Metrics (5):

- **Mean RR**: Average RR interval (ms)
- **RR Std**: Standard deviation of intervals
- **Respiration Rate**: Breaths per minute (1/RR)
- **RR Coefficient of Variation**: Std / Mean (regularity)
- **RR Entropy**: Complexity of respiration pattern

### Summary: RR Features

```
Total RR features: 5
```

---

## 4.5 Complete Feature Summary

### Feature Count by Modality

| Modality | Features | Details |
|----------|----------|---------|
| IMU Bioz | 69 | 3 axes × 23 per-axis features |
| IMU Wrist | 69 | 3 axes × 23 per-axis features |
| PPG Green | 17 | HR + HRV + quality |
| PPG Infra | 17 | HR + HRV + quality |
| PPG Red | 17 | HR + HRV + quality |
| EDA | 7 | Conductance + stress |
| RR | 5 | Respiration variability |
| **TOTAL** | **257** | **All modalities** |

---

## 4.6 Feature Computation Example

### Example: 10-Second IMU Window at 50Hz

**Input**: 500 acceleration samples
```python
x = [0.01, -0.02, 0.015, ..., -0.008]  # 500 values in g units
y = [0.02, 0.01, -0.03, ..., 0.012]
z = [-0.98, -0.99, -0.97, ..., -0.99]
```

**Compute Statistics**:
```python
mean_x = 0.0042          # Average acceleration
std_x = 0.0234           # Variability
min_x = -0.0892          # Min acceleration
max_x = 0.0754           # Max acceleration
median_x = 0.0018        # Middle value
rms_x = np.sqrt(np.mean(x**2)) = 0.0251   # Energy
```

**Compute Temporal**:
```python
zero_cross_x = np.sum(np.diff(np.sign(x)) != 0) = 23
mean_slope_x = np.mean(np.abs(np.diff(x))) = 0.0045
peak_count_x = detect_peaks(x) = 12
```

**Compute Distribution**:
```python
skewness_x = stats.skew(x) = 0.23
kurtosis_x = stats.kurtosis(x) = 2.1
entropy_x = -np.sum(p * np.log(p)) = 4.8
```

**3D Magnitude**:
```python
magnitude = np.sqrt(x**2 + y**2 + z**2)
mean_mag = np.mean(magnitude) = 1.0234
std_mag = np.std(magnitude) = 0.0342
```

**Output**: 23 features for X axis (repeat for Y, Z) → 69 IMU features

---

## 4.7 Feature Matrix Output

**File**: Per modality after feature extraction

### Structure

```
window_id | t_center | [23 x-axis features] | [23 y-axis features] | [23 z-axis features]
w_00000   | 1234567  | 0.0042 | 0.0234 | ... | 0.0125 | 0.0198 | ...
w_00001   | 1234570  | 0.0051 | 0.0241 | ... | 0.0132 | 0.0205 | ...
```

### Feature Files

- `imu_bioz_features_{window_length}s.csv` (69 features)
- `imu_wrist_features_{window_length}s.csv` (69 features)
- `ppg_green_features_{window_length}s.csv` (17 features)
- `ppg_infra_features_{window_length}s.csv` (17 features)
- `ppg_red_features_{window_length}s.csv` (17 features)
- `eda_features_{window_length}s.csv` (7 features)
- `rr_features_{window_length}s.csv` (5 features)

---

## Summary

- **Method**: Hand-crafted statistical and physiological features
- **Total**: 257 features computed per window
- **Modalities**: 7 sensor types
- **Time Scales**: 10s, 5s, 2s window sizes
- **Output**: Feature CSV files per modality
- **Next**: Fusion of all modalities into single matrix
