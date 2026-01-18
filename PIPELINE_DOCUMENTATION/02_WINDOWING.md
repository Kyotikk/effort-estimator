# Stage 2: Windowing

## Overview

Windowing segments long preprocessed time-series into fixed-duration overlapping windows suitable for feature extraction and model training.

**Key objectives:**
1. Create non-overlapping or overlapping windows from continuous signals
2. Generate window metadata (start time, end time, valid sample count)
3. Ensure consistent window handling across all modalities
4. Support multiple window lengths (2s, 5s, 10s) from same data

---

## Windowing Parameters

**Configuration (pipeline.yaml):**
```yaml
windowing:
  overlap: 0.7              # 70% overlap (30% stride)
  window_lengths_sec: [10.0, 5.0, 2.0]
```

**Definition:**
- **Window length (L):** Fixed duration in seconds
- **Overlap (O):** Fraction of window that overlaps with previous
- **Stride:** S = L × (1 - O) = L × 0.3
- **Example (10s window):** 
  - Window 1: 0.0-10.0s
  - Window 2: 3.0-13.0s (stride = 3s)
  - Window 3: 6.0-16.0s
  - ...

---

## Windowing Statistics

For total recording duration **~420 seconds** (7 minutes):

| Window Length | Stride | N Windows | Overlap |
|---------------|--------|-----------|---------|
| **10.0s** | 3.0s | 140 | 70% |
| **5.0s** | 1.5s | 280 | 70% |
| **2.0s** | 0.6s | 700 | 70% |

**After labeling alignment:** 
- 10.0s: 429 labeled windows
- 5.0s: ~800+ windows
- 2.0s: ~2000+ windows

(Numbers increase because testing on full 3-day dataset in practice)

---

## Window Structure

Each window is represented as a row with metadata:

**Window CSV format:**
```
window_id,start_idx,end_idx,valid,n_samples,t_start,t_end
0,0,320,true,320,0.0,10.0
1,96,416,true,320,3.0,13.0
2,192,512,true,320,6.0,16.0
...
```

**Columns:**
- `window_id`: Unique identifier within modality
- `start_idx`: First sample index in preprocessed time-series
- `end_idx`: Last sample index (exclusive)
- `valid`: Boolean (true if window has sufficient samples)
- `n_samples`: Number of samples in window
- `t_start`: Start time in seconds
- `t_end`: End time in seconds

---

## Windowing Algorithm

**Pseudocode:**
```python
def create_windows(df, fs, win_sec, overlap):
    """
    df: preprocessed dataframe with 'time' column
    fs: sampling frequency (Hz)
    win_sec: window length (seconds)
    overlap: overlap fraction (0-1)
    """
    n_samples = len(df)
    duration = n_samples / fs
    
    samples_per_window = int(fs * win_sec)
    stride = int(samples_per_window * (1 - overlap))
    
    windows = []
    for window_id in range(0, n_samples, stride):
        start_idx = window_id
        end_idx = min(window_id + samples_per_window, n_samples)
        
        if end_idx - start_idx < samples_per_window * 0.8:
            valid = false  # < 80% complete window
        else:
            valid = true
        
        t_start = start_idx / fs
        t_end = end_idx / fs
        
        windows.append({
            'window_id': len(windows),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'valid': valid,
            'n_samples': end_idx - start_idx,
            't_start': t_start,
            't_end': t_end
        })
    
    return DataFrame(windows)
```

---

## Windowing Output

After windowing all modalities:

```
effort_estimation_output/parsingsim3_sim_elderly3/
├── imu_bioz/
│   ├── imu_windows_10.0s.csv       [429 windows, 7 columns]
│   ├── imu_windows_5.0s.csv        [~800 windows]
│   └── imu_windows_2.0s.csv        [~2000 windows]
├── ppg_green/
│   ├── ppg_green_windows_10.0s.csv
│   ├── ppg_green_windows_5.0s.csv
│   └── ppg_green_windows_2.0s.csv
├── ppg_infra/
│   ├── ppg_infra_windows_10.0s.csv
│   ├── ppg_infra_windows_5.0s.csv
│   └── ppg_infra_windows_2.0s.csv
├── ppg_red/
│   ├── ppg_red_windows_10.0s.csv
│   ├── ppg_red_windows_5.0s.csv
│   └── ppg_red_windows_2.0s.csv
└── eda/
    ├── eda_windows_10.0s.csv
    ├── eda_windows_5.0s.csv
    └── eda_windows_2.0s.csv
```

---

## Key Windowing Decisions

### Why 70% Overlap?

**Trade-off:**
- **Less overlap (e.g., 50%):** Fewer windows, simpler data
- **More overlap (e.g., 90%):** More windows, more redundancy

**70% chosen because:**
1. Sufficient data augmentation for ML (increases effective samples)
2. Smooth transitions in features across time
3. Standard in signal processing (common choice in literature)
4. Still computationally efficient

### Why Three Window Lengths?

| Length | Use Case | Sample Rate Impact |
|--------|----------|-------------------|
| **10.0s** | Primary training | Normal effort duration (~10 beat cycles) |
| **5.0s** | Intermediate | Quick effort changes, shorter bursts |
| **2.0s** | Fast detection | Real-time applications, quick response |

**10s is primary:** Most features require adequate signal cycles.
- HR estimation: ~10 beats at 100 bpm = 6 seconds minimum
- HRV estimation: ~20 beats = 12 seconds minimum
- 10s provides safety margin

**2-5s is experimental:** May lose some HRV information.

### Why Not Sliding Window?

We use fixed-stride sliding windows (30% stride) instead of:
- **No overlap:** Loses information, too sparse
- **100% overlap:** Extreme redundancy, slow computation
- **Continuous rolling:** Same as 70% overlap in effect

---

## Windowing Edge Cases

**Last window handling:**
- If final window < 80% complete, marked as `valid=false`
- Excluded from feature extraction
- Not used in training

**Example (10s windows on 420s total):**
```
Last complete window: 410-420s
Partial window: 413-423s (only 410-420 exist = 10s, valid)
Would-be window: 416-426s (only 416-420 exist = 4s < 80%, invalid)
```

**Synchronization across modalities:**
- All modalities use SAME window times (same `t_start`, `t_end`)
- Different modalities may have different `start_idx` due to different file starts
- Alignment ensures time-synchronization later

---

## Example: Detailed 10s Window Breakdown

**Input preprocessed data (10 second sample):**
```
IMU preprocessed (fs=32 Hz, 320 samples total):
time,acc_x_dyn,acc_y_dyn,acc_z_dyn
0.000,-0.042,0.031,0.152
0.031,0.018,-0.065,0.189
...
9.969,0.145,-0.018,-0.031
[320 samples = 10 seconds at 32 Hz]
```

**Window metadata:**
```
window_id=0: t_start=0.0s, t_end=10.0s, start_idx=0, end_idx=320, n_samples=320
```

**Next window (3s stride):**
```
window_id=1: t_start=3.0s, t_end=13.0s, start_idx=96, end_idx=416
[samples 96-415 from preprocessed data, 320 samples = 10 seconds]
```

---

## Computational Efficiency

**Windowing is O(n) and very fast:**
- For 1 hour (3600s) of data at 32 Hz: 115,200 samples
- Creating 10s windows: < 1ms
- All 3 window lengths: < 5ms total
- Negligible compared to feature extraction

**Caching:** Window metadata cached; recomputed only if preprocessed data changes

---

## Quality Checks During Windowing

1. ✓ All windows have correct duration (within ±1 sample)
2. ✓ Time progression monotonic (no gaps or backwards time)
3. ✓ No windows extend beyond data bounds
4. ✓ Overlap fraction matches configuration
5. ✓ Sample counts consistent with fs and duration

---

## Potential Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Fewer windows than expected** | Data too short, or high % marked invalid | Check data duration and quality |
| **Inconsistent sample counts** | Resampling artifacts | Check preprocessing step |
| **Time gaps between windows** | Stride calculation error | Verify overlap parameter |
| **Modalities not aligned** | Different start times | Alignment step handles this |

---

## Next Step: Feature Extraction

After windowing, each window definition is used to extract features from the corresponding data range. See [03_FEATURE_EXTRACTION.md](03_FEATURE_EXTRACTION.md).

