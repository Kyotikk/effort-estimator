# Stage 2: Windowing

## Overview

Windowing segments long preprocessed time-series into fixed-duration windows suitable for feature extraction and model training.

**Key objectives:**
1. Create overlapping windows from continuous signals
2. Generate window metadata (t_start, t_center, t_end)
3. Ensure consistent window handling across all modalities
4. Support multiple window lengths (5s, 10s, 30s)

---

## Window Size Comparison Results

| Window | Overlap | Step | N Windows (per subject) | N Labeled (combined) | Performance (r) |
|--------|---------|------|------------------------|---------------------|-----------------|
| **5s** | 10% | 4.5s | ~350 | 855 | **0.626-0.644** |
| 10s | 10% | 9.0s | ~170 | 424 | 0.548-0.567 |
| 30s | 10% | 27s | ~55 | 100 | 0.184-0.364 |

**Recommendation:** Use 5s windows for optimal balance of sample size and temporal resolution.

---

## Windowing Parameters

### Configuration

```yaml
# pipeline.yaml or run_elderly_pipeline.py
windowing:
  overlap: 0.1              # 10% overlap (90% stride)
  window_lengths_sec: [5.0]  # Primary window size
```

### Parameter Definitions

- **Window length (L):** Fixed duration in seconds
- **Overlap (O):** Fraction of window that overlaps with previous
- **Step/Stride:** S = L × (1 - O)
- **t_center:** Midpoint timestamp of window (used for alignment)

### Examples

**5s window (recommended):**
```
Window 1: t=0.0 to t=5.0   → t_center = 2.5
Window 2: t=4.5 to t=9.5   → t_center = 7.0
Window 3: t=9.0 to t=14.0  → t_center = 11.5
Step = 5.0 × (1 - 0.1) = 4.5s
```

**10s window:**
```
Window 1: t=0.0 to t=10.0  → t_center = 5.0
Window 2: t=9.0 to t=19.0  → t_center = 14.0
Window 3: t=18.0 to t=28.0 → t_center = 23.0
Step = 10.0 × (1 - 0.1) = 9.0s
```

---

## Windowing Statistics by Subject

### sim_elderly3 (parsingsim3)

| Window Length | Recording (~min) | N Windows | Step |
|---------------|------------------|-----------|------|
| 5s | ~28 | ~370 | 4.5s |
| 10s | ~28 | ~185 | 9.0s |
| 30s | ~28 | ~60 | 27s |

### sim_elderly4 (parsingsim4)

| Window Length | Recording (~min) | N Windows | Step |
|---------------|------------------|-----------|------|
| 5s | ~25 | ~330 | 4.5s |
| 10s | ~25 | ~165 | 9.0s |
| 30s | ~25 | ~55 | 27s |

### sim_elderly5 (parsingsim5)

| Window Length | Recording (~min) | N Windows | Step |
|---------------|------------------|-----------|------|
| 5s | ~25 | ~330 | 4.5s |
| 10s | ~25 | ~165 | 9.0s |
| 30s | ~25 | ~55 | 27s |

---

## Window Structure

Each window is represented as a row with metadata:

### Window DataFrame Columns

```
window_id    : Unique identifier within modality
start_idx    : First sample index in preprocessed time-series
end_idx      : Last sample index (exclusive)
valid        : Boolean (true if window has sufficient samples)
n_samples    : Number of samples in window
t_start      : Start time in Unix timestamp (seconds)
t_center     : Center time (used for alignment with Borg labels)
t_end        : End time in Unix timestamp (seconds)
win_sec      : Window duration (5.0, 10.0, or 30.0)
```

### Example Output

```csv
window_id,start_idx,end_idx,valid,n_samples,t_start,t_center,t_end,win_sec
0,0,160,true,160,1764912000.0,1764912002.5,1764912005.0,5.0
1,144,304,true,160,1764912004.5,1764912007.0,1764912009.5,5.0
2,288,448,true,160,1764912009.0,1764912011.5,1764912014.0,5.0
```

---

## Windowing Algorithm

### Implementation (windowing/windows.py)

```python
def create_windows(df, fs, win_sec, overlap=0.1):
    """
    Create overlapping windows from preprocessed signal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed data with 'time' or 't_sec' column
    fs : float
        Sampling frequency (Hz), typically 32 Hz
    win_sec : float
        Window length in seconds (5.0, 10.0, or 30.0)
    overlap : float
        Overlap fraction (0.1 = 10% overlap)
    
    Returns:
    --------
    List of window dictionaries with metadata
    """
    n_samples = len(df)
    samples_per_window = int(fs * win_sec)
    step_samples = int(samples_per_window * (1 - overlap))
    
    windows = []
    window_id = 0
    
    for start_idx in range(0, n_samples - samples_per_window + 1, step_samples):
        end_idx = start_idx + samples_per_window
        
        # Get time values
        t_start = df.iloc[start_idx]['time']
        t_end = df.iloc[end_idx - 1]['time']
        t_center = (t_start + t_end) / 2
        
        windows.append({
            'window_id': window_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'valid': True,
            'n_samples': samples_per_window,
            't_start': t_start,
            't_center': t_center,
            't_end': t_end,
            'win_sec': win_sec,
        })
        window_id += 1
    
    return windows
```

---

## Why Different Overlaps?

### Current: 10% Overlap

```
Pros:
- More independent windows (less redundancy)
- Faster processing (fewer windows)
- Better for GroupKFold CV (less correlation between windows)

Cons:
- May miss short events at window boundaries
- Fewer total samples
```

### Alternative: 70% Overlap (used in some pipelines)

```
Pros:
- More windows = more training data
- Better coverage of signal transitions
- Standard in some HRV analyses

Cons:
- High correlation between adjacent windows
- GroupKFold less effective (correlated windows in different folds)
- Slower processing
```

### Recommendation

**Use 10% overlap for this dataset because:**
1. GroupKFold CV requires independent windows
2. 5s windows already provide good coverage
3. 855 samples is sufficient for training

---

## Tolerance for Fusion

When fusing features across modalities, windows must align by t_center:

| Window Size | Recommended Tolerance | Reason |
|-------------|----------------------|--------|
| 5s | 2.0s | Half-window tolerance |
| 10s | 5.0s | Half-window tolerance |
| 30s | 15.0s | Half-window tolerance |

**Issue with wrong tolerance:**
- 10s windows with 2s tolerance → sensor offset (4s) > tolerance → 0 fused rows
- Must use tolerance ≥ expected sensor offset

---

## Sample Count Calculation

For a recording of duration $D$ seconds:

$$N_{windows} = \left\lfloor \frac{D - L}{S} \right\rfloor + 1$$

Where:
- $D$ = total duration (seconds)
- $L$ = window length (seconds)
- $S$ = step size = $L \times (1 - overlap)$

**Example (D=1500s, L=5s, overlap=0.1):**
```
S = 5.0 × 0.9 = 4.5s
N = floor((1500 - 5) / 4.5) + 1 = floor(332.2) + 1 = 333 windows
```

---

## Handling Edge Cases

### Incomplete Final Window

If remaining samples < samples_per_window:
- **Current approach:** Discard incomplete window
- **Alternative:** Pad with zeros or NaN
- **Recommendation:** Discard (cleaner data)

### Gap in Signal

If time gap > 2 × step_size:
- **Current approach:** Create window anyway (may span gap)
- **Alternative:** Mark window as invalid
- **Recommendation:** Trust preprocessing to handle gaps

### Variable Sampling Rate

If actual fs differs from expected:
- **Current approach:** Resample in preprocessing
- **Result:** Consistent window sizes post-resample

---

## Output Files

After windowing, features are extracted per window:

```
/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/
  effort_estimation_output/elderly_sim_elderly3/
    ├── imu_bioz/
    │   └── imu_features_5.0s.csv
    ├── imu_wrist/
    │   └── imu_features_5.0s.csv
    ├── ppg_green/
    │   ├── ppg_green_features_5.0s.csv
    │   └── ppg_green_hrv_features_5.0s.csv
    ├── ppg_infra/
    │   ├── ppg_infra_features_5.0s.csv
    │   └── ppg_infra_hrv_features_5.0s.csv
    ├── ppg_red/
    │   ├── ppg_red_features_5.0s.csv
    │   └── ppg_red_hrv_features_5.0s.csv
    └── eda/
        ├── eda_features_5.0s.csv
        └── eda_advanced_features_5.0s.csv
```

---

## Best Practices

1. **Choose window size based on:**
   - Expected activity duration (5s good for ADLs)
   - Required temporal resolution
   - Minimum samples needed for statistics

2. **Use consistent overlap** across all modalities

3. **Verify t_center alignment** before fusion

4. **Check valid flag** before using window data

5. **Log window counts** to detect data issues early

---

## Summary

| Parameter | 5s (Recommended) | 10s | 30s |
|-----------|------------------|-----|-----|
| Window length | 5.0s | 10.0s | 30.0s |
| Overlap | 10% | 10% | 10% |
| Step | 4.5s | 9.0s | 27.0s |
| Samples per window (32 Hz) | 160 | 320 | 960 |
| Fusion tolerance | 2.0s | 5.0s | 15.0s |
| Performance (r) | 0.626-0.644 | 0.548-0.567 | 0.184-0.364 |
