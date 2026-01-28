# Borg CR10 Effort Estimation Pipeline
## Presentation Slides

---

# Slide 1: Objective

**Goal:** Predict perceived exertion (Borg CR10, 0-10) from wearable sensors

**Input Signals:**
- PPG (photoplethysmography) → Heart rate, HRV
- EDA (electrodermal activity) → Sympathetic arousal
- IMU (accelerometer + gyroscope) → Movement intensity

**Output:** Continuous effort score (0-10)

---

# Slide 2: Data Collection

| Sensor | Signal | Rate | Device |
|--------|--------|------|--------|
| PPG | Green, Infrared, Red | 64 Hz | Empatica E4 |
| EDA | Skin conductance | 4 Hz | Empatica E4 |
| IMU | Accel + Gyro (6-axis) | 100 Hz | Chest sensor |

**Ground Truth:** Borg CR10 ratings collected during ADL tasks

**Subjects:** 3 participants (elderly, healthy, clinical)

---

# Slide 3: Pipeline Overview

```
Raw Signals → Preprocess → Window → Extract Features → Fuse → Align Labels → Train
```

| Step | What happens |
|------|--------------|
| **Preprocess** | Filter noise, normalize |
| **Window** | 10s segments, 0% overlap |
| **Extract** | ~300 features per window |
| **Fuse** | Combine all modalities |
| **Align** | Match Borg labels to windows |

---

# Slide 4: Preprocessing

**PPG:**
- Bandpass filter: 0.5–4.0 Hz
- Removes DC offset + high-frequency noise
- Preserves cardiac frequencies (30-240 BPM)

**EDA:**
- Lowpass filter: 1.0 Hz
- Decompose into tonic (baseline) + phasic (responses)

**IMU:**
- Separate gravity from dynamic acceleration
- Compute magnitude: $\sqrt{x^2 + y^2 + z^2}$

---

# Slide 5: Windowing

**Parameters:**
- Window size: **10 seconds**
- Overlap: **0%** (critical!)

**Why 0% overlap?**

With 70% overlap:
```
Window 1: [0-10s]
Window 2: [3-13s]  ← 7s shared data!
```
→ **Temporal leakage** if split into train/test

With 0% overlap:
```
Window 1: [0-10s]
Window 2: [10-20s]  ← No shared data
```
→ **No leakage** ✓

---

# Slide 6: Feature Extraction - PPG

**Time-domain (per wavelength):**
- Mean, std, min, max
- Percentiles: p5, p25, p50, p75, p95
- Skewness, kurtosis
- RMS, trim mean

**Derivative features:**
- First derivative: mean, std
- Second derivative: std

**~40 features per wavelength × 3 wavelengths = 120 PPG features**

---

# Slide 7: Feature Extraction - HRV

**From PPG peak detection:**

| Feature | Meaning |
|---------|---------|
| **hr_mean** | Mean heart rate (BPM) |
| **mean_ibi** | Mean inter-beat interval (ms) |
| **sdnn** | Std of IBI → overall HRV |
| **rmssd** | Root mean square of successive differences → parasympathetic activity |

**Expected correlations:**
- HR ↑ when effort ↑ ✓
- IBI ↓ when effort ↑ ✓
- RMSSD ↓ when effort ↑ ✓

---

# Slide 8: Feature Extraction - EDA & IMU

**EDA Features:**
- Mean, std, slope
- SCR count (skin conductance responses)
- Tonic level (baseline arousal)

**IMU Features (per axis + magnitude):**
- Mean, std, variance
- Energy, entropy
- Zero crossings
- Peak count

**Total: ~150 IMU features**

---

# Slide 9: Feature Fusion

**Merge all modalities on window center timestamp:**

```
PPG features  ─┐
HRV features  ─┼─→ Merge on t_center ─→ Combined
EDA features  ─┤
IMU features  ─┘
```

**Result:** ~300 features per 10-second window

---

# Slide 10: Label Alignment

**ADL Labels format:**
```
t_start | t_end | activity | borg
0       | 60    | walking  | 3
60      | 120   | stairs   | 7
...
```

**Alignment:**
- For each window with center time `t_center`
- Find ADL where: `t_start ≤ t_center ≤ t_end`
- Assign that Borg rating

**Result:** 1199 labeled windows from 3 subjects

---

# Slide 11: Data Sanitization

**Remove metadata (prevents leakage):**
- ❌ t_start, t_end (time information)
- ❌ window_id, *_idx (identifiers)
- ❌ subject, activity (categorical)

**Handle missing values:**
- Drop columns with >50% NaN
- Impute remaining with median

**Final:** 299 usable features

---

# Slide 12: Validation Approach

**Leave-One-Subject-Out Cross-Validation (LOSO):**

```
Fold 1: Train on {healthy, severe} → Test on elderly
Fold 2: Train on {elderly, severe} → Test on healthy
Fold 3: Train on {elderly, healthy} → Test on severe
```

**Why LOSO?**
- Tests generalization to **new individuals**
- Prevents subject-specific pattern leakage
- Realistic deployment scenario

---

# Slide 13: Physiological Validation

**Correlations with Borg (before training):**

| Feature | Expected | Observed | ✓/✗ |
|---------|----------|----------|-----|
| HR | + | r = +0.43 | ✓ |
| IBI | − | r = −0.46 | ✓ |
| RMSSD | − | r = −0.23 | ✓ |
| EDA | + | r = +0.15 | ✓ |
| ACC mag | + | r = +0.38 | ✓ |

**All correlations match physiological expectations!**

---

# Slide 14: Dataset Summary

| Metric | Value |
|--------|-------|
| **Subjects** | 3 |
| **Total windows** | ~3000+ |
| **Labeled windows** | 1199 |
| **Features** | 299 |
| **Window size** | 10 seconds |
| **Overlap** | 0% |

**Borg distribution:** 0-10 scale, mean ~4.5

---

# Slide 15: Pipeline Flow Diagram

```
┌────────────────────────────────────────┐
│           RAW SENSOR DATA              │
│   PPG (64Hz) + EDA (4Hz) + IMU (100Hz) │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│            PREPROCESSING               │
│   Filter → Normalize → Clean           │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│             WINDOWING                  │
│      10s windows, 0% overlap           │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│         FEATURE EXTRACTION             │
│   PPG + HRV + EDA + IMU = 300 features │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│      LABEL ALIGNMENT + FUSION          │
│     Match Borg ratings to windows      │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│              OUTPUT                    │
│    1199 samples × 299 features         │
└────────────────────────────────────────┘
```

---

# Slide 16: Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Window size | 10s | Standard for HRV; captures 10-15 heartbeats |
| Overlap | 0% | Prevents temporal leakage |
| NaN threshold | 50% | Balance data retention vs quality |
| Imputation | Median | Robust to outliers |
| Validation | LOSO | Tests cross-subject generalization |

---

# Slide 17: Current Limitations

1. **Small sample size** (n=3 subjects)
   - Model overfits to individual patterns
   - LOSO CV shows poor generalization

2. **Sparse labels**
   - One Borg rating per ADL task
   - Not continuous during activity

3. **HRV coverage**
   - ~16% of windows lack HRV features
   - Insufficient detected heartbeats

---

# Slide 18: Next Steps

1. **Add more subjects** → Improve generalization
2. **Continuous Borg collection** → Finer temporal resolution
3. **Feature selection** → Reduce from 300 to most predictive
4. **Simpler models** → Less overfitting with small n
5. **Domain adaptation** → Transfer learning across subjects

---

# Slide 19: Summary

✅ **Methodology is correct:**
- No temporal leakage (0% overlap)
- No metadata leakage (sanitized)
- HRV features included
- Physiologically valid correlations

⚠️ **Model generalization limited:**
- Need more subjects for robust cross-subject prediction
- Current data good for proof-of-concept

---

# Appendix: Feature List (Top 20 by Importance)

1. ppg_green_p99
2. ppg_infra_diff_mean_abs
3. ppg_green_p95
4. ppg_green_p90
5. eda_cc_max
6. ppg_red_min
7. ppg_infra_ddx_std
8. acc_z_dyn_variance
9. ppg_green_tke_std
10. ppg_red_p99_p1
11. ppg_green_median
12. ppg_green_ddx_std
13. ppg_green_hr_min
14. ppg_green_diff_std
15. **ppg_green_mean_ibi** ← HRV!
16. ppg_red_std
17. acc_z_dyn_lower_moment
18. acc_z_dyn_sample_entropy
19. acc_x_dyn_max
20. eda_cc_min
