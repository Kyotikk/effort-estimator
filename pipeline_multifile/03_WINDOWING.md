# Stage 3: Windowing

## Purpose

Segment preprocessed time series into fixed-size windows suitable for feature extraction, with overlap for temporal continuity.

---

## 3.1 Window Creation Process

**File**: `windowing/windows.py`
**Function**: `create_windows(time_series, window_length_sec, overlap_percent, sampling_freq)`

### Input
- Preprocessed time series (any modality)
- Window length: 10s, 5s, or 2s
- Overlap: 70% (configurable)
- Sampling frequency: modality-dependent (8-50 Hz)

### Algorithm

**Step 1: Calculate Window Parameters**

```python
window_samples = int(window_length_sec * sampling_freq)
stride_samples = int(window_samples * (1 - overlap_percent))

# Example: 10s window @ 50Hz, 70% overlap
# window_samples = 500
# stride_samples = 150 (30% of window)
```

**Step 2: Create Window Boundaries**

```python
windows = []
for start_idx in range(0, len(time_series) - window_samples, stride_samples):
    end_idx = start_idx + window_samples
    t_start = time_series[start_idx]
    t_end = time_series[end_idx - 1]
    t_center = (t_start + t_end) / 2
    
    windows.append({
        'window_id': f'w_{len(windows):05d}',
        't_start': t_start,
        't_center': t_center,
        't_end': t_end,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_samples': window_samples
    })
```

**Step 3: Extract Measurements Per Window**

```python
# For each window, compute statistics
for window in windows:
    start_idx = window['start_idx']
    end_idx = window['end_idx']
    
    segment = time_series[start_idx:end_idx]
    
    window['mean'] = np.mean(segment)
    window['std'] = np.std(segment)
    window['min'] = np.min(segment)
    window['max'] = np.max(segment)
    window['median'] = np.median(segment)
    # ... more statistics
```

**Step 4: Validate Windows**

```python
for window in windows:
    # Check if window contains enough samples
    expected_samples = window_samples
    actual_samples = window['n_samples']
    validity_threshold = 0.8  # 80% of expected
    
    if actual_samples >= validity_threshold * expected_samples:
        window['valid'] = 1
    else:
        window['valid'] = 0
```

### Example Window Sequence

**Configuration**: 10s windows, 70% overlap, at 50 Hz

```
Timeline (seconds):
0s ─────── 10s ─────── 20s ─────── 30s ─────── 40s

Window 0:  [0-10s]       t_center=5s
Window 1:  [3-13s]       t_center=8s       (3s stride)
Window 2:  [6-16s]       t_center=11s
Window 3:  [9-19s]       t_center=14s
Window 4:  [12-22s]      t_center=17s
...
```

**Stride Calculation**:
- Window length: 10s
- Overlap: 70%
- Non-overlapping portion: 30% × 10s = 3s
- Stride: 3s between window starts

---

## 3.2 Window Validity Criteria

A window is marked `valid=1` if:

1. **Sufficient Samples** (≥80% of expected)
   - Expected samples = window_length × sampling_freq
   - Example: 10s @ 50Hz = 500 expected, need ≥400 actual

2. **Valid Time Range**
   - Start and end within sensor recording range
   - No NaN in timestamps

3. **Data Quality** (post-features)
   - ≥95% of windows valid in feature extraction

### Example Validity Checks

```python
# IMU at 50Hz, 10s window
expected_samples_imu = 10 * 50 = 500
min_required = 0.8 * 500 = 400

if actual_imu_samples >= 400:
    valid_imu = 1

# PPG at 8Hz, 10s window
expected_samples_ppg = 10 * 8 = 80
min_required = 0.8 * 80 = 64

if actual_ppg_samples >= 64:
    valid_ppg = 1
```

---

## 3.3 Window Output Structure

**File**: `{modality}_windows_{window_length}s.csv`

### Columns

```
window_id        | Unique identifier (w_00000, w_00001, ...)
t_start          | Start timestamp (Unix seconds)
t_center         | Center timestamp (Unix seconds)
t_end            | End timestamp (Unix seconds)
valid            | 1=valid, 0=invalid
n_samples        | Number of samples in window
win_sec          | Window length in seconds
[modality]_mean  | Mean of signal
[modality]_std   | Standard deviation
[modality]_min   | Minimum value
[modality]_max   | Maximum value
[modality]_median| Median value
...              | Additional statistics
```

### Example Row

```
window_id: w_00042
t_start:   1700000000.2
t_center:  1700000005.2
t_end:     1700000010.2
valid:     1
n_samples: 501
win_sec:   10.0
mean:      0.0125
std:       0.0342
min:       -0.0891
max:       0.0754
```

---

## 3.4 Multi-Window Sizes

Pipeline creates windows for **three time scales**:

| Window Size | Overlap | Stride | Use Case |
|-------------|---------|--------|----------|
| **10s** | 70% | 3s | Broader temporal context, standard |
| **5s** | 70% | 1.5s | Finer temporal resolution |
| **2s** | 70% | 0.6s | High-frequency changes |

**Output**: 3 separate window files per modality
- `{modality}_windows_10.0s.csv`
- `{modality}_windows_5.0s.csv`
- `{modality}_windows_2.0s.csv`

---

## 3.5 Quality Assurance

**Checks after windowing**:

✅ Window count reasonable (not too many, not too few)
✅ Time continuity (no gaps or overlaps > threshold)
✅ Validity distribution (>95% valid windows)
✅ Timestamp monotonically increasing
✅ No duplicate window IDs

**Example QA Output**:
```
Total windows created: 3,524
Valid windows: 3,412 (96.8%)
Invalid windows: 112 (3.2%)
  → Mostly at start/end (expected)

Time coverage: 100% (no gaps > 1 window)
Timestamp check: ✓ Monotonic
Window IDs: ✓ Unique
```

---

## Summary

- **Purpose**: Divide time series into fixed-size segments
- **Method**: Fixed-length windows with overlap
- **Overlap**: 70% to capture temporal continuity
- **Validity**: ≥80% of expected samples required
- **Output**: Windowed features per modality
- **Next**: Features extracted from each window
