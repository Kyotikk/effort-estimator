# 🎯 Quick Start: Your Next 3 Days

## Day 1: Understand the Plan (1-2 hours)

### Read These Documents (In Order)

1. **MULTI_SUBJECT_EXPANSION_PLAN.md** ← START HERE
   - 10 min overview of 4 phases
   - What you'll build and when
   - Expected improvements

2. **10_IMU_MULTI_INPUT_GUIDE.md**
   - Implementation details for Phase 1
   - Code examples you'll use
   - Testing strategy

3. **09_MULTI_SUBJECT_ARCHITECTURE.md** (optional today)
   - Technical deep-dive for Phase 3
   - Multi-subject training strategy
   - Data organization patterns

### Verify Your Data Exists

```bash
# Check that 3 IMU sources are available
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_acc/
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_acc/
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_acceleration/

# Check sim_severe3 exists for Phase 2
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_severe3/
```

---

## Day 2: Implement Phase 1 (3-4 hours)

### Step 1: Update Configuration

**File**: `config/pipeline.yaml`

Find this section (around line 16):
```yaml
imu_bioz:
  path: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_acc/2025-12-04.csv.gz
  fs_out: 32
```

Add below it:
```yaml
imu_wrist:
  path: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_acc/2025-12-04.csv.gz
  fs_out: 32

imu_chest:
  path: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/vivalnk_vv330_acceleration/2025-12-04.csv.gz
  fs_out: 32
```

Also find the fusion section (around line 100):
```yaml
fusion:
  modalities:
    imu_bioz: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/imu_bioz/imu_bioz_features_{window_length}.csv
```

Add:
```yaml
    imu_wrist: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/imu_wrist/imu_wrist_features_{window_length}.csv
    imu_chest: /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/imu_chest/imu_chest_features_{window_length}.csv
```

### Step 2: Modify Pipeline Script

**File**: `run_pipeline.py`

Find the IMU processing section (around line 66):
```python
# ---------- MODALITY: IMU ----------
print("▶ IMU: preprocessing")
imu_cfg = cfg["preprocessing"]["imu"]
feat_cfg_imu = cfg["features"]["imu"]

imu_path = dataset["imu_bioz"]["path"]
# ... rest of single IMU code ...
```

Replace with multi-IMU loop. See **10_IMU_MULTI_INPUT_GUIDE.md** for exact code.

### Step 3: Test the Pipeline

```bash
cd /Users/pascalschlegel/effort-estimator
python run_pipeline.py 2>&1 | head -50
```

**Expected output**:
```
▶ IMU: preprocessing
  ▶ imu_bioz: preprocessing
    ✓ Saved to .../imu_bioz_preprocessed.csv
  ▶ imu_wrist: preprocessing
    ✓ Saved to .../imu_wrist_preprocessed.csv
  ▶ imu_chest: preprocessing
    ✓ Saved to .../imu_chest_preprocessed.csv

▶ Windowing
  ...
```

### Step 4: Verify Results

```bash
# Check that 90 IMU features were created (was 30)
head -1 /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/imu_bioz/imu_bioz_features_10.0s.csv | tr ',' '\n' | wc -l
# Should output: 30

head -1 /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/imu_wrist/imu_wrist_features_10.0s.csv | tr ',' '\n' | wc -l
# Should output: 30

# Total features in fused file (should be 278 instead of 188)
head -1 /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_features_10.0s.csv | tr ',' '\n' | wc -l
# Should output: 278 or close
```

### Step 5: Train & Check Performance

```bash
python train_xgboost_borg.py
```

**Expected**:
- Model trains without errors
- R² ≥ 0.85 (may be slightly lower than 0.92 due to more features)
- 3 IMU modalities all contributing to model

---

## Day 3: Prepare for Phase 2 (1-2 hours)

### Option A: Process sim_severe3 (Recommended)

1. Create `config/pipeline_severe.yaml` (copy of pipeline.yaml but point to sim_severe3 data)
2. Run pipeline on sim_severe3:
   ```bash
   python run_pipeline.py config/pipeline_severe.yaml
   ```
3. Expected: ~1,200 windows generated

### Option B: Document Phase 1 Results

1. Save model performance metrics
2. Analyze: Which IMU features matter most?
3. Document findings

---

## Your Documentation

**Read in this order**:
1. ✅ **MULTI_SUBJECT_EXPANSION_PLAN.md** - Overview (today)
2. ✅ **10_IMU_MULTI_INPUT_GUIDE.md** - Implementation (today)
3. ⏳ **09_MULTI_SUBJECT_ARCHITECTURE.md** - Multi-subject (Week 3)
4. 📚 **README.md** - Full context (anytime)
5. 📚 **07_PERFORMANCE_METRICS.md** - Current performance (anytime)

**All in**: `/Users/pascalschlegel/effort-estimator/PIPELINE_DOCUMENTATION/`

---

## Common Issues & Solutions

### Issue: "File not found" for IMU inputs

**Solution**: Check paths in config/pipeline.yaml match exactly:
```bash
ls -la /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_acc/
```

### Issue: Model R² drops significantly (< 0.80)

**Solution**: 
- Check that 90 IMU features are being extracted
- Verify no NaN values in new IMU data
- This is OK - SelectKBest will drop weak features

### Issue: Run takes very long

**Solution**:
- Processing 3 IMU types takes ~3× longer
- Normal! Should complete in 10-20 minutes total
- Use `python run_pipeline.py 2>&1 | tail -20` to see progress

---

## Success Criteria

✅ **Phase 1 Complete When**:
- [ ] 90 IMU features extracted (30 per IMU type)
- [ ] 278 total features in fused data (was 188)
- [ ] Model trains and achieves R² ≥ 0.85
- [ ] No error messages

✅ **Ready for Phase 2 When**:
- [ ] Phase 1 working on sim_elderly3
- [ ] You understand the 4-IMU concept
- [ ] You can modify config files
- [ ] You're ready to process sim_severe3

---

## Next: After Phase 1 Complete

### Before Phase 2:
1. Commit your changes to git:
   ```bash
   cd /Users/pascalschlegel/effort-estimator
   git add config/pipeline.yaml run_pipeline.py
   git commit -m "Implement 4-IMU integration: 188 → 278 features"
   ```

2. Document results:
   - Model performance with 4-IMU
   - Which IMU features are most important
   - Any issues encountered

3. Read: 09_MULTI_SUBJECT_ARCHITECTURE.md

### Phase 2 Plan:
- Process sim_severe3 with same 4-IMU setup
- Expected: Another 1,200+ windows
- Timeline: 1 week

---

## Questions?

All answers are in the PIPELINE_DOCUMENTATION folder:

| Question | Document |
|----------|----------|
| How does preprocessing work? | 01_PREPROCESSING.md |
| What are the 4 IMU inputs? | 10_IMU_MULTI_INPUT_GUIDE.md |
| How do I integrate them? | 10_IMU_MULTI_INPUT_GUIDE.md (Step-by-step) |
| What should I expect? | MULTI_SUBJECT_EXPANSION_PLAN.md |
| How do I train multi-subject? | 09_MULTI_SUBJECT_ARCHITECTURE.md |
| Current performance? | 07_PERFORMANCE_METRICS.md |
| Navigation? | INDEX.md |

---

**Status**: ✅ Ready to Start

Your pipeline is saved, documented, and ready to expand. Start with Day 1 (reading), then Day 2 (implementation), then Day 3 (testing).

Good luck! 🚀
