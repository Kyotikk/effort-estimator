# Stage 6: Quality Checks

## Purpose

Validate feature quality and document coverage issues before alignment with ADL labels.

---

## 6.1 Quality Metrics

**File**: `windowing/feature_quality_check_any.py`

### Metrics Computed

For each feature in the fused matrix:

| Metric | Computation | Purpose |
|--------|-----------|---------|
| **Validity Rate** | % of rows with `valid=1` | Data coverage |
| **NaN Rate** | % of NaN values per feature | Missing data |
| **Outlier Rate** | % beyond mean ± 5×std | Anomalies |
| **Mean** | Average value | Baseline |
| **Std** | Standard deviation | Variability |
| **Min/Max** | Minimum and maximum | Range |
| **Skewness** | Distribution asymmetry | Normality |
| **Kurtosis** | Distribution tailedness | Outlier presence |

### Example Calculation

```python
for feature_name in fused_df.columns:
    feature_values = fused_df[feature_name].dropna()
    
    # NaN rate
    nan_rate = fused_df[feature_name].isna().sum() / len(fused_df)
    
    # Outliers (> mean ± 5*std)
    mean = np.mean(feature_values)
    std = np.std(feature_values)
    outliers = np.sum(np.abs(feature_values - mean) > 5 * std)
    outlier_rate = outliers / len(feature_values)
    
    # Validity
    validity = (fused_df['valid'] == 1).sum() / len(fused_df)
    
    metrics[feature_name] = {
        'nan_rate': nan_rate,
        'outlier_rate': outlier_rate,
        'validity': validity,
        'mean': mean,
        'std': std
    }
```

---

## 6.2 Validity Thresholds

| Metric | Pass | Warning | Fail |
|--------|------|---------|------|
| **Valid Windows** | >95% | 80-95% | <80% |
| **Feature NaN Rate** | <5% | 5-20% | >20% |
| **Outlier Rate** | <2% | 2-5% | >5% |

### Decision Logic

```python
def classify_feature_quality(feature_metrics):
    nan_rate = feature_metrics['nan_rate']
    outlier_rate = feature_metrics['outlier_rate']
    
    if nan_rate < 0.05 and outlier_rate < 0.02:
        return "✓ PASS"
    elif nan_rate < 0.20 and outlier_rate < 0.05:
        return "⚠ WARNING"
    else:
        return "❌ FAIL - consider excluding"
```

---

## 6.3 Quality Check Output

**File**: `feature_quality_check_{window_length}s.txt` and `.png`

### Text Report Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature Quality Report
Subject: sim_elderly3, Window: 10.0s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Overview:
  Total windows: 3,524
  Valid windows (valid=1): 3,412 (96.8%) ✓
  Invalid windows: 112 (3.2%)

Feature-Level Analysis:
  Total features: 257

✓ PASS (NaN<5%, Outliers<2%):
  mean_accel_x: NaN=0.0%, Outliers=0.1%
  std_accel_y: NaN=0.0%, Outliers=0.3%
  heart_rate: NaN=0.2%, Outliers=0.0%
  ... (254 more PASS)

⚠ WARNING (5-20% NaN or 2-5% Outliers):
  scl_level: NaN=8.2%, Outliers=1.5%
  rr_variability: NaN=12.4%, Outliers=2.1%

❌ FAIL (>20% NaN or >5% Outliers):
  (none)

Summary:
  PASS: 254 features (98.8%)
  WARNING: 3 features (1.2%)
  FAIL: 0 features (0.0%)

Recommendation: Proceed with caution
  → All features usable, 3 have some NaN (not critical)
  → Will be handled in fusion/selection stages
```

---

## 6.4 Visualization

### Plots Generated

**1. Feature NaN Rate Histogram**

```
Histogram of NaN rates across 257 features
  Most features: 0-2% NaN (good!)
  Few features: 5-15% NaN (acceptable)
  None: >20% NaN
```

**2. Valid Window Count Over Time**

```
Time series showing:
  • Number of valid windows per 1-hour bin
  • Detection of data gaps
  • Coverage uniformity
```

**3. Feature Distribution Heatmap**

```
Color-coded heatmap showing:
  • Feature name (rows)
  • Statistics: mean, std, skewness, kurtosis (columns)
  • Color intensity = feature magnitude
```

**4. Outlier Rate by Feature**

```
Scatter plot:
  X-axis: Feature index (0-257)
  Y-axis: Outlier percentage (0-5%)
  Points above 2%: Investigate
```

---

## 6.5 Quality Check Workflow

```python
def run_quality_checks(fused_df, output_dir):
    """
    1. Compute all quality metrics
    2. Classify features (PASS/WARNING/FAIL)
    3. Generate text report
    4. Generate visualizations
    """
    
    # Compute metrics
    metrics = compute_quality_metrics(fused_df)
    
    # Classify
    classifications = classify_features(metrics)
    
    # Report
    write_text_report(metrics, classifications, output_dir)
    
    # Visualize
    create_histograms(metrics, output_dir)
    create_heatmap(fused_df, output_dir)
    create_outlier_plot(metrics, output_dir)
    
    # Summary
    n_pass = sum(1 for c in classifications.values() if c == 'PASS')
    n_warn = sum(1 for c in classifications.values() if c == 'WARNING')
    n_fail = sum(1 for c in classifications.values() if c == 'FAIL')
    
    print(f"Quality Check Complete:")
    print(f"  PASS: {n_pass}")
    print(f"  WARNING: {n_warn}")
    print(f"  FAIL: {n_fail}")
```

---

## Summary

- **Purpose**: Validate data quality before ML
- **Metrics**: NaN rate, validity, outliers, distribution
- **Thresholds**: Pass (<5% NaN), Warning (5-20%), Fail (>20%)
- **Output**: Text report + visualizations
- **Decision**: Proceed if >95% valid, can handle warnings
- **Next**: Label attachment (ADL alignment)
