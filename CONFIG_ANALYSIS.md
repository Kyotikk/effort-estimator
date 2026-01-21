# Config Files Analysis - Currently Used vs Unused

## Summary

**Total config files:** 19 YAML files  
**Actually used in current pipeline:** 2  
**Unused/obsolete:** 17

---

## ACTIVE CONFIG FILES (Used in Current Pipeline)

### 1. ✅ `config/pipeline.yaml` - PRIMARY & ACTIVE

**Used by:**
- `run_pipeline.py` (line 528: default argument)
- `phases/phase5_alignment/run_target_alignment.py` (line 254)

**Purpose:** Main orchestration config for full end-to-end pipeline

**What it defines:**
- Project metadata (name, output_dir)
- Raw dataset paths (IMU, PPG, ECG, EDA)
- Preprocessing parameters (noise cutoff, HPF, etc.)
- Windowing config (overlap, window lengths)
- Feature extraction settings
- Target alignment config (ADL path, HRV recovery params)
- Fusion output directory

**Status:** ✅ **CRITICAL - Used every pipeline run**

**Current datasets configured:**
- parsingsim3_sim_elderly3 (only one - others need HRV labels)

---

### 2. ⚠️ `config/ecg_processing.yaml` - NEVER USED

**Status:** ❌ **NOT REFERENCED anywhere**

**Why not used:**
- ECG processing happens inside `run_pipeline.py` Phase 1
- All ECG params are in `config/pipeline.yaml` preprocessing.ecg section
- This file appears to be a forgotten stub

---

## INACTIVE CONFIG FILES (Never used in current pipeline)

### 3-19. ❌ Dataset-Specific Configs (NEVER USED)

All variations below are **alternative dataset configurations** but are NOT loaded by the pipeline:

```
config/pipeline_sim_elderly3.yaml      - Same as pipeline.yaml
config/pipeline_sim_elderly4.yaml      - For sim_elderly4
config/pipeline_sim_elderly5.yaml      - For sim_elderly5
config/pipeline_sim_healthy3.yaml      - For sim_healthy3
config/pipeline_sim_healthy4.yaml      - For sim_healthy4
config/pipeline_sim_healthy5.yaml      - For sim_healthy5
config/pipeline_sim_severe3.yaml       - For sim_severe3
config/pipeline_sim_severe4.yaml       - For sim_severe4
config/pipeline_sim_severe5.yaml       - For sim_severe5
config/pipeline_elderly4.yaml          - Variant for sim_elderly4
config/pipeline_elderly4_full.yaml     - Full variant
config/pipeline_healthy3.yaml          - For sim_healthy3
config/pipeline_severe3.yaml           - For sim_severe3
config/pipeline_severe5.yaml           - For sim_severe5
config/pipeline_severe5_full.yaml      - Full variant
```

**Why not used:**
- Pipeline defaults to `config/pipeline.yaml`
- To use alternatives, would need to explicitly pass them:
  ```bash
  python run_pipeline.py config/pipeline_sim_elderly4.yaml
  ```
- **Currently not done** - only `config/pipeline.yaml` is used

**Status:** ❌ **Obsolete unless multi-dataset batch processing**

---

### 20. ❌ `config/training.yaml` - NEVER USED

**Status:** ❌ **NOT REFERENCED anywhere**

**Problems:**
- References Windows paths (C:\Users\...)
- Not integrated with `train_hrv_recovery_clean.py`
- Hardcoded paths from old Windows development

---

### 21. ❌ `config/hrv_pipeline_example.yaml` - NEVER USED

**Status:** ❌ **NOT REFERENCED anywhere**

---

## How Pipeline ACTUALLY Works (Current Implementation)

```
User runs:
  python run_pipeline.py [config_path]

If config_path not provided:
  defaults to "config/pipeline.yaml"

Load config → Parse YAML → Extract settings

Run phases:
  1. Preprocessing (uses config settings)
  2. Windowing (uses config settings)
  3. Feature extraction (uses config settings)
  4. Fusion (hardcoded in phase4)
  5. Alignment (uses config settings from phase5 section)
  6. Feature selection (hardcoded in phase6)

Train model:
  python train_hrv_recovery_clean.py
  (no config - hardcoded paths, no config load)
```

**Key insight:** Only `config/pipeline.yaml` is actually read and used. All other config files are dead code.

---

## What WOULD Need to Change for Multi-Dataset

To use other configs, would need:

### Option 1: Manual specification
```bash
python run_pipeline.py config/pipeline_sim_elderly4.yaml
python run_pipeline.py config/pipeline_sim_healthy3.yaml
```

### Option 2: Batch script
```bash
#!/bin/bash
for config in config/pipeline_sim_*.yaml; do
  python run_pipeline.py "$config"
done
```

### Option 3: Config selection in code
```python
import sys
config = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
run_pipeline(config)
```

**Current state:** None of these are implemented. Only sim_elderly3 data flows through the pipeline.

---

## Recommendations

### Clean Up Unused Configs

**Delete (safe - never referenced):**
```bash
rm config/pipeline_sim_*.yaml
rm config/pipeline_*elderly*.yaml
rm config/pipeline_*healthy*.yaml
rm config/pipeline_*severe*.yaml
rm config/pipeline_*_full.yaml
rm config/ecg_processing.yaml
rm config/training.yaml
rm config/hrv_pipeline_example.yaml
```

**Keep (actually used):**
```bash
# Only keep:
config/pipeline.yaml
```

**Result:** Single source of truth, eliminates confusion

---

## Current Config Structure (`config/pipeline.yaml`)

```yaml
project:
  - name
  - output_dir

datasets: [ARRAY]
  - dataset 0: parsingsim3_sim_elderly3
    - imu_bioz, imu_wrist, ppg_green, ppg_infra, ppg_red, eda, ecg
    - path, fs_out for each

preprocessing: [DICT]
  - imu_bioz, imu_wrist
  - ppg_green, ppg_infra, ppg_red
  - ecg
  - eda
  - Each has: time_col, signal_col, processing params

windowing:
  - overlap: 0.7
  - window_lengths_sec: [10.0, 5.0, 2.0]

features: [DICT]
  - imu_bioz, imu_wrist, ppg_*, eda, rr
  - Each has: modality, prefix, signal_col, etc.

targets:
  - target_type: "hrv_recovery_rate"
  - imu:
    - adl_path: ADL annotations
    - rr_path: RR intervals (from ECG preprocessing)
    - recovery windows (rec_start, rec_end)

fusion:
  - output_dir
  - window_lengths_sec
```

---

## Suggested Cleanup

### Keep Only Active Config

```bash
# Backup old configs
mkdir -p config/archive
mv config/pipeline_*.yaml config/archive/
mv config/training.yaml config/archive/
mv config/ecg_processing.yaml config/archive/
mv config/hrv_pipeline_example.yaml config/archive/
```

**Result:**
- `config/` directory stays clean
- Only `config/pipeline.yaml` remains active
- Old configs preserved in archive if needed

### Future: Multi-Dataset Support

If batch processing is needed later:

```python
# Enhanced run_pipeline.py
datasets_to_process = [
    "config/pipeline_sim_elderly3.yaml",
    "config/pipeline_sim_healthy3.yaml",
    "config/pipeline_sim_severe3.yaml",
]

for config in datasets_to_process:
    run_pipeline(config)
```

---

## Summary Table

| File | Used | Reference | Status |
|------|------|-----------|--------|
| `config/pipeline.yaml` | ✅ YES | run_pipeline.py line 528 | **ACTIVE** |
| `config/ecg_processing.yaml` | ❌ NO | None | Dead code |
| `config/training.yaml` | ❌ NO | None | Dead code |
| `config/hrv_pipeline_example.yaml` | ❌ NO | None | Dead code |
| `config/pipeline_sim_*.yaml` (10 files) | ❌ NO | None | Dead code |
| `config/pipeline_*_full.yaml` (2 files) | ❌ NO | None | Dead code |
| `config/pipeline_elderly*.yaml` (2 files) | ❌ NO | None | Dead code |
| `config/pipeline_healthy*.yaml` (1 file) | ❌ NO | None | Dead code |
| `config/pipeline_severe*.yaml` (2 files) | ❌ NO | None | Dead code |

---

## Conclusion

**Current pipeline uses: 1 config file (pipeline.yaml)**
**Unused configs: 18 files**

For clean pipeline maintenance: **Delete all unused configs or move to archive**.
