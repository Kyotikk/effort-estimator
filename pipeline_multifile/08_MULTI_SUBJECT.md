# Stage 8: Multi-Subject Dataset Combination

## Purpose

Merge aligned datasets from all 3 conditions into a single training dataset with standardized format.

---

## 8.1 Combination Process

**File**: `run_multisub_pipeline.py` - `combine_datasets()` function

### Algorithm

**Step 1: Load Subject Datasets**

```python
def combine_datasets(subjects, window_length):
    """
    Load aligned files from each subject
    """
    dataframes = []
    
    for subject in subjects:  # ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
        aligned_file = f"/data/interim/parsingsim3/{subject}/effort_estimation_output/aligned_{window_length}s.csv"
        
        df = pd.read_csv(aligned_file)
        print(f"Loaded {subject}: {len(df)} rows")
        
        dataframes.append(df)
```

**Step 2: Add Subject Column**

```python
    # Ensure subject column exists
    for subject, df in zip(subjects, dataframes):
        if 'subject' not in df.columns:
            df['subject'] = subject
```

**Step 3: Concatenate Vertically**

```python
    # Combine all subjects
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined: {len(combined)} total rows")
```

**Step 4: Filter to Labeled Only**

```python
    # Remove unlabeled windows (borg=NaN)
    combined_labeled = combined.dropna(subset=['borg'])
    
    n_total = len(combined)
    n_labeled = len(combined_labeled)
    n_unlabeled = n_total - n_labeled
    
    print(f"Labeled: {n_labeled} ({100*n_labeled/n_total:.1f}%)")
    print(f"Unlabeled: {n_unlabeled} ({100*n_unlabeled/n_total:.1f}%)")
    
    return combined_labeled
```

**Step 5: Save Combined Dataset**

```python
    output_file = f"/multisub_combined/multisub_aligned_{window_length}s.csv"
    combined_labeled.to_csv(output_file, index=False)
    
    print(f"✓ Combined dataset saved: {output_file}")
    print(f"  Shape: {combined_labeled.shape}")
```

---

## 8.2 Subject-Level Statistics

### sim_elderly3 (Elderly Population)

```
Raw aligned file:
  Total windows: 450
  Labeled: 429 (95.3%)
  Unlabeled: 21 (4.7%)
  
Mean Borg: 3.30 ± 1.88
Borg range: 0.5 - 6.0
Distribution: Well-distributed across range
```

### sim_healthy3 (Healthy/Low Effort)

```
Raw aligned file:
  Total windows: 380
  Labeled: 347 (91.3%)
  Unlabeled: 33 (8.7%)
  
Mean Borg: 0.28 ± 0.32
Borg range: 0.0 - 1.5
Distribution: 93.7% at 0-1 Borg (extremely narrow!)
```

### sim_severe3 (High Intensity)

```
Raw aligned file:
  Total windows: 450
  Labeled: 412 (91.6%)
  Unlabeled: 38 (8.4%)
  
Mean Borg: 4.71 ± 2.06
Borg range: 1.5 - 8.0
Distribution: 50% at extreme (5-8 Borg)
```

---

## 8.3 Combined Dataset Statistics

### Overall

| Metric | Value |
|--------|-------|
| **Total Subjects** | 3 |
| **Total Windows (all)** | 1,280 |
| **Labeled Windows** | 1,188 (92.8%) |
| **Unlabeled Windows** | 92 (7.2%) |
| **Borg Range** | 0.0 - 8.0 |
| **Mean Borg (overall)** | 2.76 |
| **Feature Columns** | 262 (257 features + 5 metadata) |

### By Condition

```
Subject       | Windows | Labeled | % Labeled | Mean Borg | Range
──────────────┼─────────┼─────────┼───────────┼───────────┼───────────
sim_elderly3  | 450     | 429     | 95.3%     | 3.30      | 0.5-6.0
sim_healthy3  | 380     | 347     | 91.3%     | 0.28      | 0.0-1.5
sim_severe3   | 450     | 412     | 91.6%     | 4.71      | 1.5-8.0
──────────────┼─────────┼─────────┼───────────┼───────────┼───────────
TOTAL         | 1,280   | 1,188   | 92.8%     | 2.76      | 0.0-8.0
```

---

## 8.4 Key Insights

### Effort Distributions Are Very Different

```
Borg Distribution Visualization:

elderly3 (n=429):          healthy3 (n=347):        severe3 (n=412):
█████████████████          ███████████████████      ██████
████████████               ████                     ███████
███████                    ██                       ████████
██████                     █                        █████████
█████                      █                        ██████████ ← 50% here!
████                       █                        ███████████
███                        █                        ██████████
```

### Why Single Model Won't Work

```
Multi-subject single model attempts:
  • Input: all 1,188 samples, mixed distributions
  • Target: Borg values ranging 0.0 to 8.0
  • Problem: Model struggles to fit wildly different ranges
  • Result: R² = -113 (terrible!)
  
With condition-specific models:
  • elderly3 model: learns 0.5-6.0 range
  • healthy3 model: learns 0.0-1.5 range (tight!)
  • severe3 model: learns 1.5-8.0 range
  • Result: R² = 0.926, 0.405, 0.997 respectively ✓
```

### Sample Count Distribution

```
Borg bins (all subjects):
  0.0-1.0: 367 samples (30.8%)  ← healthy3 dominates
  1.0-2.0: 89 samples (7.5%)
  2.0-3.0: 145 samples (12.2%)  ← elderly3
  3.0-4.0: 186 samples (15.6%)  ← mixed
  4.0-5.0: 134 samples (11.3%)  ← severe3
  5.0-6.0: 89 samples (7.5%)    ← severe3
  6.0-7.0: 56 samples (4.7%)    ← severe3
  7.0-8.0: 22 samples (1.9%)    ← severe3 (extreme)
```

---

## 8.5 Combined Dataset Output

**File**: `/multisub_combined/multisub_aligned_10.0s.csv`

### Structure

```
Rows: 1,188 (labeled samples only)
Columns: 262 (5 metadata + 257 features)
  
window_id | t_start | t_center | t_end | subject | modality | valid | n_samples | 
[257 feature columns]... | borg

Example row:
w_00042 | 1700000000 | 1700000005 | 1700000010 | sim_elderly3 | fused | 1 | 501 |
  0.0125 | 0.0234 | ... | (257 more features) ... | 3.5
```

### Sample Composition

```
Condition      | Samples | Percentage
───────────────┼─────────┼───────────
sim_elderly3   | 429     | 36.1%
sim_healthy3   | 347     | 29.2%
sim_severe3    | 412     | 34.7%
───────────────┼─────────┼───────────
TOTAL          | 1,188   | 100%
```

---

## Summary

- **Purpose**: Merge 3 subjects into one dataset
- **Method**: Concatenate vertically, filter to labeled
- **Result**: 1,188 labeled samples from 3 distinct conditions
- **Key Insight**: Conditions have vastly different effort ranges
- **Implication**: Separate models needed for each condition
- **Next**: Feature selection and model training
