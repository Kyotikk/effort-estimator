# Effort Estimation Formula - Scientific Summary

## Goal
Develop a **training load / effort intensity proxy** from wearable sensor data (chest-worn HR + accelerometer) that correlates with perceived exertion (Borg CR10) in **neuropatients performing activities of daily living (ADLs)**.

## Constraints
- No lab-grade equipment (VO2, lactate)
- Short activity bouts (10s - 3min), not sustained exercise
- Neuropatients have autonomic dysfunction → unreliable HRV
- Need a simple, interpretable formula (no black-box ML)

---

## Approach: Weighted Linear Combination

**Final Formula:**
```
Effort = 0.8 × z(HR_load) + 0.2 × z(IMU_load)
```

Where:
- `HR_load = (HR_mean - HR_rest) × √duration`
- `IMU_load = MAD × √duration`
- `MAD = Mean Amplitude Deviation of acceleration magnitude`
- `z()` = z-score normalization

---

## Why These Metrics?

| Component | Rationale | Reference |
|-----------|-----------|-----------|
| **HR_mean - HR_rest** | HR reserve correlates with %VO2max | Karvonen (1957) |
| **MAD** | Movement intensity, gravity-independent | Mathie (2004), van Hees (2013) |
| **√duration** | Stevens' Power Law: perceived effort scales sublinearly with time | Stevens (1957) |
| **80/20 weighting** | HR dominates effort perception; IMU captures mechanical work when HR unreliable | Empirically optimized on n=5 subjects |

---

## Validation Results (n=27 ADL bouts, 1 subject)

| Metric | Correlation with Borg CR10 |
|--------|---------------------------|
| HR_load alone | r = 0.82 |
| IMU_load alone | r = 0.48 |
| Combined (0.8/0.2) | r = 0.83 |

---

## Alternatives Considered but Rejected

| Method | Issue |
|--------|-------|
| **Banister TRIMP** | Designed for sustained exercise, exponential weighting overshoots for short ADLs |
| **Edwards TRIMP** | Requires HR zones; zones poorly defined in neuropatients |
| **RMSSD (HRV)** | Weak correlation in neuropatients due to autonomic dysfunction |
| **Raw RMS acceleration** | Includes gravity, sensitive to sensor orientation |

---

## Implementation

### HR Load
```python
hr_delta = hr_mean - hr_rest  # HR elevation above baseline
hr_load = hr_delta * np.sqrt(duration_seconds)
```

### IMU Load (MAD)
```python
mag = np.sqrt(x**2 + y**2 + z**2)  # Acceleration magnitude in g
mad = np.mean(np.abs(mag - np.mean(mag)))  # Mean Amplitude Deviation
imu_load = mad * np.sqrt(duration_seconds)
```

### Combined Effort Score
```python
z_hr = (hr_load - mu_hr) / sigma_hr
z_imu = (imu_load - mu_imu) / sigma_imu
effort = 0.8 * z_hr + 0.2 * z_imu
```

---

## Questions for Scientific Review

1. Is `HR_delta × √duration` a reasonable proxy for internal training load in short-bout ADLs?
2. Is MAD the best accelerometer-derived intensity metric, or would ENMO, SMA, or activity counts be better?
3. Are there better ways to combine HR and IMU signals (e.g., multiplicative, nonlinear)?
4. Any concerns with using Stevens' Power Law (√duration) for physical effort scaling?
5. Literature on effort estimation in neuropatients specifically?

---

*Created: 2026-01-29*
*Branch: experiment/test-new-idea*
