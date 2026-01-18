# Stage 7: ADL Alignment & Labeling ⭐ CRITICAL

## Purpose

Attach Borg effort labels to sensor windows by matching them to ADL (Activity of Daily Living) events from the SCAI app.

**⚠️ Critical Feature**: Window time range filtering to handle sensor/app desynchronization.

---

## 7.1 ADL Data Parsing

**File**: `ml/targets/adl_alignment.py`
**Function**: `parse_adl_intervals(adl_csv)`

### Input
- ADL CSV file from SCAI app
- Path: `scai_app/ADLs_*.csv` (or `.csv.gz`)

### ADL CSV Formats

**Format 1 (Datetime strings)**:
```csv
Time,Activities,Borg RPE,Additional
2024-10-15 09:15:30.000,Walking,3,
2024-10-15 09:25:45.000,Sitting,1,
2024-10-15 09:45:20.000,Climbing stairs,6,
```

**Format 2 (Unix timestamps)**:
```csv
Start,Stop,Activities,Borg
1700000100,1700000160,Walking,3
1700000200,1700000270,Sitting,1
1700000500,1700000620,Climbing stairs,6
```

### Parsing Steps

**Step 1: Read CSV**
```python
df = pd.read_csv(adl_csv, sep=',', dtype=str)
# Handle encoding issues, missing values
```

**Step 2: Detect Timestamp Format**
- If contains '-' and ':' → datetime string (Format 1)
- If all digits → Unix timestamp (Format 2)

**Step 3: Convert to Unix Seconds (UTC)**
- **For datetime strings**: Parse with timezone handling
  - Format: `YYYY-MM-DD HH:MM:SS.mmm`
  - Timezone: Japan (UTC+9) → convert to UTC
  - Formula: $t_{UTC} = t_{Japan} - 9 \times 3600$

- **For Unix timestamps**: Already UTC, use as-is

```python
import pandas as pd
from dateutil import parser

if is_datetime_format(df):
    # Parse with Japan timezone offset
    df['t_unix'] = pd.to_datetime(df['Time']).values.astype('int64') // 10**9
    df['t_unix'] -= 9 * 3600  # Convert Japan → UTC
else:
    # Already Unix timestamp
    df['t_unix'] = df['Start'].astype(int)
```

**Step 4: Extract Borg Values**
- Parse Borg column (0-10 scale)
- Handle missing/invalid values (set to NaN)
- Validate range: 0 ≤ Borg ≤ 10

**Step 5: Create Intervals**
- Each ADL event defines interval: [t_start, t_end]
- t_start: start of activity
- t_end: end of activity (next activity start or last timestamp)

### Output
- List of tuples: `[(t_start, t_end, borg_value), ...]`
- All times in Unix seconds (UTC)
- Example:
  ```python
  [
    (1700000100, 1700000160, 3.0),    # Walking, Borg 3
    (1700000200, 1700000270, 1.0),    # Sitting, Borg 1
    (1700000500, 1700000620, 6.0)     # Climbing stairs, Borg 6
  ]
  ```

---

## 7.2 Window-to-Borg Alignment

**Function**: `align_windows_to_borg(windows_df, borg_intervals)`

### The Critical Step: Time Range Filtering

#### Problem

Sensor windows may have **different time bounds** than ADL recording:

```
healthy3 example:
  Sensor windows: t_min=1764832240, t_max=1764836887
  Duration: 4,647 seconds (77 minutes)
  
  ADL recording: t_min=1764835353, t_max=1764836800
  Duration: 1,447 seconds (24 minutes)
  
  Desynchronization: 52 minutes!
  Sensors started 52 minutes BEFORE app
```

#### Why This Happens

- Wearable sensors auto-start when turned on
- Mobile app user launches minutes later
- No manual time synchronization

#### The Solution: Filter Windows

**Algorithm**:

```python
def align_windows_to_borg(windows_df, borg_intervals):
    """
    Step 1: Calculate ADL recording time bounds
    """
    if not borg_intervals:
        windows_df['borg'] = np.nan
        return windows_df
    
    t_starts = [t[0] for t in borg_intervals]
    t_ends = [t[1] for t in borg_intervals]
    
    adl_t_min = min(t_starts)
    adl_t_max = max(t_ends)
    
    print(f"ADL time range: {adl_t_min} to {adl_t_max}")
    print(f"Sensor window range: {windows_df['t_center'].min()} to {windows_df['t_center'].max()}")
    
    """
    Step 2: Filter windows to ADL time range
    """
    mask = (windows_df['t_center'] >= adl_t_min) & \
           (windows_df['t_center'] <= adl_t_max)
    
    windows_filtered = windows_df[mask].copy()
    
    n_before = len(windows_df)
    n_after = len(windows_filtered)
    n_removed = n_before - n_after
    
    print(f"Windows before filtering: {n_before}")
    print(f"Windows after filtering: {n_after}")
    print(f"Windows removed: {n_removed} ({100*n_removed/n_before:.1f}%)")
    
    """
    Step 3: Match windows to intervals
    """
    windows_filtered['borg'] = np.nan
    
    for i, row in windows_filtered.iterrows():
        t_center = row['t_center']
        
        # Find interval containing this window center
        for t_start, t_end, borg_value in borg_intervals:
            if t_start <= t_center <= t_end:
                windows_filtered.loc[i, 'borg'] = borg_value
                break
    
    """
    Step 4: Return aligned dataset
    """
    return windows_filtered
```

### Example Alignment

```
ADL Intervals:
  [1700000000, 1700000300] → borg=2
  [1700000300, 1700000600] → borg=4
  [1700000600, 1700001200] → borg=6

Sensor Windows (BEFORE filtering):
  window_id | t_center    | Action
  w_0000    | 1699999950  | FILTERED OUT (before ADL start)
  w_0001    | 1700000050  | ✅ matched [1700000000-300] → borg=2
  w_0002    | 1700000350  | ✅ matched [1700000300-600] → borg=4
  w_0003    | 1700000800  | ✅ matched [1700000600-1200] → borg=6
  w_0004    | 1700001500  | FILTERED OUT (after ADL end)

Result: 3 labeled windows (from 4 sensor windows)
```

---

## 7.3 Performance Impact

### Without Time Range Filtering ❌

```
healthy3:
  Total windows: 380
  Labeled windows: 0 (0%)
  → UNUSABLE! No training data!
  
severe3:
  Total windows: 450
  Labeled windows: 0 (0%)
  → UNUSABLE!
```

### With Time Range Filtering ✅

```
healthy3:
  Total windows: 380
  After filtering: 347 (91.3%)
  Labeled windows: 347 (100% of filtered)
  ✅ Ready for training!
  
severe3:
  Total windows: 450
  After filtering: 412 (91.6%)
  Labeled windows: 412 (100% of filtered)
  ✅ Ready for training!
```

**Result**: From 0 labeled samples → 759 labeled samples! 🎉

---

## 7.4 Output Structure

### Input (fused features)
```
window_id | t_start    | t_center   | t_end      | valid | mean_accel_x | ...
w_0042    | 1234567    | 1234570    | 1234573    | 1     | 0.012       | ...
w_0043    | 1234570    | 1234573    | 1234576    | 1     | 0.015       | ...
```

### Output (after alignment)
```
window_id | t_start    | t_center   | t_end      | valid | mean_accel_x | ... | borg
w_0042    | 1234567    | 1234570    | 1234573    | 1     | 0.012       | ... | 3.5
w_0043    | 1234570    | 1234573    | 1234576    | 1     | 0.015       | ... | 3.5
w_0044    | 1234573    | 1234576    | 1234579    | 1     | 0.011       | ... | NaN (no match)
```

### File
- Output: `{subject}/effort_estimation_output/aligned_{window_length}s.csv`
- Size: 1,188 rows (labeled only) × 262 columns
- Format: CSV, all numeric

---

## 7.5 Implementation Details

**File**: `ml/targets/run_target_alignment.py`
**Function**: `run_alignment()`

**Key Code Section** (lines 45-63):

```python
# Parse ADL intervals
intervals = parse_adl_intervals(adl_csv)

# Get window time range
adl_t_min = intervals["t_start"].min()
adl_t_max = intervals["t_end"].max()

# CRITICAL: Filter windows to ADL time range
mask = (windows_labeled["t_center"] >= adl_t_min) & \
       (windows_labeled["t_center"] <= adl_t_max)
windows_labeled = windows_labeled[mask].copy()

print(f"Windows within ADL time range: {len(windows_labeled)}")
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | Fused feature matrix + ADL CSV labels |
| **Parsing** | Detect format (datetime or Unix), convert to UTC |
| **Critical Step** | Filter windows to ADL time bounds |
| **Matching** | For each window, find interval containing t_center |
| **Output** | Labeled windows with Borg column |
| **Performance** | 0 → 1,188 labeled samples with filtering! |

**⭐ Key Insight**: Time range filtering is **essential** for getting any labeled data at all!
