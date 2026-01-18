# Pipeline Documentation Index

## Quick Navigation

Welcome to the comprehensive Effort Estimation Pipeline documentation. This folder contains detailed explanations of every stage from raw sensor data to model predictions.

---

## 📚 Documentation Files

### 🎯 [README.md](README.md) - START HERE
**Complete pipeline overview (15 min read)**

Best first resource. Contains:
- 30,000 ft executive summary
- Pipeline architecture diagram
- Quick statistics & key metrics
- Multi-subject expansion roadmap summary
- What's uploaded, what features are used, how they're selected
- Performance metrics at a glance
- All abbreviations explained

**Read this if:** You want the big picture overview

---

### 🔧 Stage-by-Stage Detailed Docs

#### [01_PREPROCESSING.md](01_PREPROCESSING.md)
**Raw sensor data → Clean signals (10 min read)**

Covers:
- IMU acceleration preprocessing (gravity removal, noise filtering)
- PPG wrist sensor preprocessing (3 variants, HPF for weak signals)
- EDA electrodermal preprocessing (stress response isolation)
- RR respiratory rate infrastructure (future work)
- Signal quality analysis (why RED/INFRA need HPF)
- Why each preprocessing step was chosen

**Key metrics:**
- Green PPG: 8,614 units (baseline)
- Infra PPG: 5,024 units (-42%)
- Red PPG: 2,731 units (-68%)
- HPF cutoff: 0.5 Hz (removes baseline drift)

---

#### [02_WINDOWING.md](02_WINDOWING.md)
**Continuous signals → Fixed-duration windows (8 min read)**

Covers:
- Windowing algorithm (sliding windows with 70% overlap)
- Window metadata structure
- Three window lengths: 10s (primary), 5s, 2s
- Why 70% overlap chosen
- Edge case handling

**Key metrics:**
- 10s windows: 429 labeled
- 5s windows: ~800
- 2s windows: ~2000
- Stride (30% of window length)

---

#### [03_FEATURE_EXTRACTION.md](03_FEATURE_EXTRACTION.md)
**Windowed signals → 188 numerical features (12 min read)**

Covers:
- 30 IMU features (acceleration, jerk, frequency spectrum)
- 44 PPG features per variant (HR, HRV, spectral, morphology)
- 26 EDA features (tonic/phasic conductance statistics)
- Why these feature categories
- Feature definitions with examples

**Key breakdown:**
- Total: 188 features
- IMU: 30 (acceleration + dynamics)
- PPG: 132 (44 × 3 variants)
- EDA: 26 (conductance statistics)

---

#### [04_ALIGNMENT_AND_FUSION.md](04_ALIGNMENT_AND_FUSION.md)
**Features + Labels → Training dataset (10 min read)**

Covers:
- How Borg effort labels are aligned with windows
- Time-based matching from ADL annotations
- Feature fusion across modalities
- Why windows are dropped
- Final fused table structure

**Key metrics:**
- Labeled windows: 429 (from ~2000 generated)
- Train/test split: 80/20 (343 train, 86 test)
- Feature matrix: 429 × 188

---

#### [05_FEATURE_SELECTION.md](05_FEATURE_SELECTION.md)
**188 features → 100 selected (10 min read)**

Covers:
- The curse of dimensionality problem
- Why 188 features caused overfitting
- SelectKBest with f_regression method
- How overfitting was eliminated
- Which features were selected (and which rejected)

**Key results:**
- Before selection: 188 features, overfitting gap = 0.061
- After selection: 100 features, overfitting gap = 0.0001
- EDA dominates (top 6 features all EDA)
- Red PPG severely downweighted (only 8/44 features)

---

#### [06_TRAINING.md](06_TRAINING.md)
**Selected features → Trained XGBoost model (12 min read)**

Covers:
- Data preparation & cleaning
- Feature scaling (StandardScaler)
- XGBoost hyperparameters & rationale
- Training process
- Performance metrics (train, test, cross-validation)
- Feature importance interpretation
- Model reproducibility

**Key model:**
- Algorithm: XGBoost regressor
- Hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=200
- Training data: 343 samples
- Test data: 86 samples

---

#### [07_PERFORMANCE_METRICS.md](07_PERFORMANCE_METRICS.md)
**Complete results & interpretation (15 min read)**

Covers:
- Test R² = 0.9225 (explains 92.25% of variance)
- Test RMSE = 0.5171 Borg points (±0.52 typical error)
- Test MAE = 0.3540 Borg points
- Cross-validation analysis (5-fold, R² = 0.8689 ± 0.036)
- Overfitting analysis (gap reduced 99.8%)
- Feature importance breakdown
- Error distribution analysis
- Per-subject performance (if multi-subject)
- Confidence intervals & outlier detection
- Comparison to commercial devices

**Clinical utility:**
- Sensitivity/specificity for high effort detection
- Real-time performance (<1ms inference)

---

#### [08_MULTISUB_ROADMAP.md](08_MULTISUB_ROADMAP.md)
**Path from single-subject to production (10 min read)**

Covers:
- Current limitations (single subject, n=429)
- Phase 2: Multi-subject data collection (5 subjects, 3000+ windows)
- Phase 3: Data harmonization
- Phase 4: Model retraining with larger dataset
- Phase 5: Cross-subject generalization validation
- Phase 6: Feature stability analysis
- Phase 7: Real-world deployment
- Phase 8: Continuous improvement pipeline
- Timeline, budget, risk mitigation

**Success criteria:**
- Multi-subject cross-validation R² ≥ 0.85
- RMSE < 0.6 Borg points
- No systematic bias across cohorts
- Deployment-ready by month 7-9

---

## 📊 Quick Reference: What's Where?

### To understand...

| Question | See | File |
|----------|-----|------|
| ...the full pipeline in 5 min | Overview section | README.md |
| ...raw data cleaning | Preprocessing | 01_PREPROCESSING.md |
| ...why we use 70% overlap | Window design | 02_WINDOWING.md |
| ...what 188 features are | Feature definitions | 03_FEATURE_EXTRACTION.md |
| ...how labels are assigned | Label alignment | 04_ALIGNMENT_AND_FUSION.md |
| ...why 100 not 188 features | Curse of dimensionality | 05_FEATURE_SELECTION.md |
| ...the model hyperparameters | Model configuration | 06_TRAINING.md |
| ...R² = 0.9225 exactly | Metric interpretation | 07_PERFORMANCE_METRICS.md |
| ...multi-subject expansion | Future roadmap | 08_MULTISUB_ROADMAP.md |
| ...PPG signal quality | Signal analysis | 01_PREPROCESSING.md |
| ...why EDA matters most | Feature importance | 07_PERFORMANCE_METRICS.md |
| ...how to generalize to new patients | Multi-subject plan | 08_MULTISUB_ROADMAP.md |

---

## 🎯 Reading Recommendations

### For Managers / PMs
1. README.md (overview)
2. 07_PERFORMANCE_METRICS.md (results)
3. 08_MULTISUB_ROADMAP.md (next steps)

**Time: ~20 minutes**

---

### For Data Scientists / ML Engineers
1. README.md (context)
2. 03_FEATURE_EXTRACTION.md (features)
3. 05_FEATURE_SELECTION.md (selection method)
4. 06_TRAINING.md (model details)
5. 07_PERFORMANCE_METRICS.md (evaluation)

**Time: ~1 hour**

---

### For Sensor / Signal Processing Engineers
1. README.md (context)
2. 01_PREPROCESSING.md (detailed signal processing)
3. 02_WINDOWING.md (windowing)
4. 03_FEATURE_EXTRACTION.md (feature extraction)

**Time: ~45 minutes**

---

### For Clinical / Domain Experts
1. README.md (overview)
2. 08_MULTISUB_ROADMAP.md (multi-subject validation plan)
3. 07_PERFORMANCE_METRICS.md (clinical metrics)

**Time: ~30 minutes**

---

### For System Architects / DevOps
1. README.md (pipeline overview)
2. 06_TRAINING.md (model training)
3. 08_MULTISUB_ROADMAP.md (continuous improvement)

**Time: ~30 minutes**

---

## 📈 Key Takeaways (One-Pagers)

### Model Performance
```
✅ Test R²: 0.9225 (92% variance explained)
✅ Test RMSE: 0.5171 Borg points (±0.52 error)
✅ No overfitting: train-test gap = 0.0001
✅ Stable: CV R² = 0.8689 ± 0.036

Status: PRODUCTION-READY for single subject
```

### Data & Features
```
Sensors: 3×PPG + IMU + EDA + RR infrastructure
Features: 188 total → 100 selected
Top features: EDA (52.8%), PPG Green (22.5%), IMU (10.4%)
Data: 429 labeled windows from 1 elderly patient
```

### Next Steps
```
Phase 1 (Now): Single-subject validation ✅
Phase 2 (3-6 mo): Multi-subject data collection
Phase 3 (7-9 mo): Cross-subject validation & deployment
Goal: R² ≥ 0.85 on held-out subjects
```

---

## 🔗 How to Use This Documentation

### For Implementation
1. Read relevant stage docs in order
2. Reference the pseudocode and algorithms
3. Check configuration examples
4. Follow quality checks

### For Debugging
1. Find your stage in the index
2. Look for "Troubleshooting" section
3. Check "Common Issues" table
4. Verify against expected outputs

### For Extension
1. Read full stage doc for context
2. Check "Future Work" sections
3. Understand constraints & decisions
4. Propose changes aligned with rationale

---

## 📞 Document Metadata

| Property | Value |
|----------|-------|
| **Version** | 2.0 (Multi-modal, Feature-Selected) |
| **Updated** | 2026-01-18 |
| **Subject** | sim_elderly3 (elderly patient) |
| **Modalities** | 6 (PPG×3, IMU, EDA, RR-infra) |
| **Features** | 100 selected from 188 |
| **Data Points** | 429 labeled windows |
| **Model** | XGBoost v200 trees |
| **Primary Metric** | R² = 0.9225 |

---

## ✅ Checklist: What's Covered?

### Preprocessing ✅
- [x] IMU acceleration (gravity removal, noise filtering)
- [x] PPG wrist sensors (3 variants, selective HPF)
- [x] EDA electrodermal (stress response isolation)
- [x] RR respiratory (infrastructure, non-uniform sampling noted)
- [x] Signal quality analysis
- [x] Preprocessing configuration

### Windowing ✅
- [x] Windowing algorithm (70% overlap)
- [x] Window metadata structure
- [x] Multiple window lengths (10s, 5s, 2s)
- [x] Edge case handling

### Features ✅
- [x] 30 IMU features explained
- [x] 44 PPG features × 3 variants
- [x] 26 EDA features
- [x] Feature extraction pseudocode
- [x] Quality checks

### Alignment & Fusion ✅
- [x] Label alignment process
- [x] Time-based matching
- [x] Multi-modality fusion
- [x] Output structure

### Feature Selection ✅
- [x] SelectKBest method explained
- [x] Overfitting problem & solution
- [x] Top features ranked
- [x] Feature importance by modality

### Training ✅
- [x] Data preparation
- [x] Feature scaling
- [x] XGBoost hyperparameters
- [x] Cross-validation setup
- [x] Reproducibility & seeding

### Metrics ✅
- [x] R² interpretation (0.9225)
- [x] RMSE & MAE calculation
- [x] Overfitting analysis
- [x] Cross-validation breakdown
- [x] Error distribution
- [x] Feature importance interpretation
- [x] Clinical utility metrics

### Multi-Subject ✅
- [x] Current limitations identified
- [x] Phased roadmap (8 phases)
- [x] Subject recruitment strategy
- [x] Cross-subject validation plan
- [x] Timeline & budget
- [x] Risk mitigation

---

## 📞 Questions?

| Topic | File | Section |
|-------|------|---------|
| "Why does X matter?" | README.md | Rationale sections |
| "How exactly does Y work?" | Stage-specific docs | Algorithm sections |
| "What's the next step?" | 08_MULTISUB_ROADMAP.md | Phase descriptions |
| "Is the model good?" | 07_PERFORMANCE_METRICS.md | Performance summary |
| "Can this work for new patients?" | 08_MULTISUB_ROADMAP.md | Validation section |

---

**Total Reading Time Estimates:**
- Quick overview: 15 min (README only)
- Manager briefing: 30 min
- Technical deep-dive: 2 hours
- Complete mastery: 4-6 hours

