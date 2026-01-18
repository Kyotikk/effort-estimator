# Multi-Subject Expansion Roadmap

## Executive Summary

Current pipeline is production-ready for single-subject monitoring but requires multi-subject validation and data expansion for clinical deployment. This document outlines the path to a generalizable, robust effort estimation model.

---

## Phase 1: Current State (v2 Complete) ✅

### Completed
- ✅ Single-patient data processing pipeline
- ✅ 6-modality sensor integration (3×PPG + IMU + EDA + RR infra)
- ✅ 188 features extracted, 100 selected via f_regression
- ✅ XGBoost model with 92.25% test R²
- ✅ Cross-validation stable (5-fold, CV R² = 0.8689 ± 0.036)
- ✅ Overfitting eliminated (train-test gap = 0.0001)

### Dataset
```
Subject:     sim_elderly3 (elderly, single)
Labeled windows (10s): 429
Sample duration: ~70 minutes
Borg range: 0-20
Conditions: Mixed activities (rest, light walk, moderate/heavy exercise)
```

### Model Performance
```
Test R²:     0.9225
Test RMSE:   0.5171 Borg points
Test MAE:    0.3540 Borg points
CV R²:       0.8689 ± 0.0360
```

### Limitations
- ⚠️ **Single subject:** Cannot claim generalization
- ⚠️ **Limited sample size:** 429 windows is on the threshold
- ⚠️ **One age group:** Only elderly data
- ⚠️ **No health diversity:** Only healthy elderly
- ⚠️ **Internal test set:** Test from same patient

---

## Phase 2: Multi-Subject Data Collection (3-6 months)

### Objectives

1. **Expand to 3-5 subjects** across different profiles
2. **Collect 1000+ labeled windows** total
3. **Represent full Borg range** (0-20)
4. **Diverse health states** (healthy, severe, elderly)

### Subject Recruitment Strategy

```
Cohort A: Healthy Young (2-3 subjects)
  Age: 20-35
  Health: No cardiovascular/respiratory disease
  Purpose: Baseline cardiac/EDA response to effort
  Expected characteristics:
    - High HR response to effort (180+ bpm max)
    - Good HRV at baseline
    - Minimal resting EDA (1-2 µS)
    - Strong PPG signals

Cohort B: Healthy Elderly (1-2 subjects)
  Age: 65-75
  Health: No disease, normal aging
  Purpose: Age-matched comparison
  Expected characteristics:
    - Moderate HR response (150-160 bpm max)
    - Reduced HRV compared to young
    - Slightly elevated resting EDA
    - Variable PPG signals

Cohort C: Severe/Medical (1-2 subjects)
  Status: Post-cardiac event, COPD, diabetes
  Age: Variable
  Purpose: Understand effort capacity limitations
  Expected characteristics:
    - Limited HR response (120-130 bpm max)
    - Elevated resting HR
    - High phasic EDA (stress response exaggerated)
    - Potentially weak PPG signals
```

### Data Collection Protocol

**Per subject:**
```
Session length: 45-60 minutes
Borg labeling: Every 30-60 seconds (ADL annotations)
Activities:
  1. Rest (baseline, 10 min)
  2. Light activity (5-10 min, walk, yoga)
  3. Moderate activity (10-15 min, brisk walk, light jog)
  4. High effort (5-10 min, stairs, running if tolerated)
  5. Recovery (5 min)

Sensors active: All 6 modalities (3×PPG + IMU + EDA + RR)
Recording frequency: 32 Hz
```

### Expected Data Yield

```
Per subject (60 min recording):
  - Windows (10s, 70% overlap): ~720
  - Labeled windows: ~500-600 (accounting for unlabeled transitions)

Total (5 subjects × 550 avg):
  - New windows: 2,750
  - Combined with current (429): 3,179 total
```

---

## Phase 3: Data Harmonization & Preprocessing

### Objectives
- Standardize data formats across subjects
- Handle inter-subject variation in sensor readings
- Prepare for cross-subject validation

### Standardization Steps

**1. Timestamp Alignment**
```
Challenge: Each subject's recording starts at different absolute time
Solution: Relative timestamps (0s = session start)
Verification: All data within [0, session_duration] seconds
```

**2. Sensor Calibration**
```
PPG baseline shifts between subjects:
  - Baseline EDA varies by skin properties (1-10 µS)
  - PPG signal amplitude depends on skin pigmentation
  - HR varies at baseline (50-80 bpm healthy)
  
Approach: Feature normalization (StandardScaler) handles this
Alternative: Per-subject baseline normalization if needed
```

**3. Borg Label Consistency**
```
Subjective scale, can drift between subjects/raters
Solution:
  - Train raters on standard Borg scale
  - Use same rater across subjects if possible
  - Validate by comparing with physiological expectations
    (Borg ↑ should correlate with HR ↑, EDA ↑)
```

**4. Signal Quality Validation**
```
Per-window checks:
  ✓ PPG heart rate detectable (50-200 bpm)
  ✓ EDA range reasonable (0-20 µS)
  ✓ IMU not saturated (±16 g range)
  ✓ RR intervals reasonable (0.2-10 sec)
  
Drop windows failing checks.
```

---

## Phase 4: Model Retraining

### Objectives
- Train new model on multi-subject data
- Validate generalization capability
- Optimize hyperparameters with larger dataset

### Retraining Strategy

**Approach 1: Naive Pooling (Simple)**
```python
# Combine all subjects' data
X_all = pd.concat([X_s1, X_s2, X_s3, X_s4, X_s5])
y_all = pd.concat([y_s1, y_s2, y_s3, y_s4, y_s5])

# 80/20 split (random)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# Train
model_v3 = xgb.XGBRegressor(**params).fit(X_train, y_train)
```

**Expected improvement:**
- R² test: 0.9225 → 0.94-0.95
- RMSE: 0.5171 → 0.40-0.45
- No overfitting expected (larger dataset)

**Approach 2: Cross-Subject Validation (Rigorous)**
```python
# Leave-one-subject-out cross-validation
for test_subject in [S1, S2, S3, S4, S5]:
    X_train = pool(subjects - test_subject)
    y_train = pool(targets - test_subject)
    X_test = test_subject features
    y_test = test_subject targets
    
    model_k = xgb.XGBRegressor(**params).fit(X_train, y_train)
    cv_scores_k = evaluate(model_k, X_test, y_test)
    
report_mean_cv_score()
```

**Expected result:**
- Shows true generalization capability
- Identifies which subjects are outliers
- Informs model robustness

### Hyperparameter Tuning

**With 3000+ samples, can optimize:**

```python
param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(
    xgb.XGBRegressor(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best R²: {grid_search.best_score_:.4f}")
```

**Expected tuning improvements:**
- Depth: Increase from 6 to 7-8 (more data to fit)
- Learning rate: Could decrease to 0.05 (slower, better fit)
- N_estimators: Increase from 200 to 300+
- Expected R² gain: +0.01-0.02

---

## Phase 5: Validation & Testing

### Cross-Subject Generalization Test

**Protocol:**
```
Train on: Subjects 1-4 (80% of combined data)
Test on: Subject 5 (held out, ~600 windows)

Evaluate:
  - Test R²: Expected 0.88-0.92 (drop from 0.94-0.95 is normal)
  - Subject-specific RMSE variations
  - Are errors larger for different age/health?
```

**Success criteria:**
- ✅ Test R² > 0.85 (generalizes reasonably)
- ✅ RMSE < 0.6 Borg points
- ✅ No systematic bias (predictions not systematically too high/low)

### Per-Subject Performance Analysis

```
Model trained on subjects 1-4, tested on each:
  Subject 1: R² = 0.92 (seen in training)
  Subject 2: R² = 0.89 (seen in training)
  Subject 3: R² = 0.91 (seen in training)
  Subject 4: R² = 0.88 (seen in training)
  Subject 5: R² = 0.87 (held out, true generalization)
  
Interpretation:
  - Small drop from training to held-out (0.90 avg → 0.87)
  - Consistent across subjects (no outliers)
  - Acceptable generalization
```

### Domain Shift Analysis

**Question:** Does model performance degrade for different conditions?

```python
# Segment test data by subject health status
young_test = X_test[subjects == 'young']
elderly_test = X_test[subjects == 'elderly']
cardiac_test = X_test[subjects == 'cardiac_event']

# Evaluate on each segment
r2_young = r2_score(y_test[young], model.predict(young_test))
r2_elderly = r2_score(y_test[elderly], model.predict(elderly_test))
r2_cardiac = r2_score(y_test[cardiac], model.predict(cardiac_test))

print(f"Young R²: {r2_young:.4f}")
print(f"Elderly R²: {r2_elderly:.4f}")
print(f"Cardiac R²: {r2_cardiac:.4f}")
```

**Expected:**
- Young/Elderly: Similar R² (0.88-0.91 range)
- Cardiac: Possibly lower (0.82-0.87 range) due to atypical physiology

**Mitigation:** If cardiac shows large degradation, may need:
- Cardiac-specific model (separate training)
- OR additional cardiac-event data in main model

---

## Phase 6: Feature Analysis Across Subjects

### Are Features Stable Across Subjects?

**Question:** Do the top features (EDA stress, PPG stats) matter equally for all subjects?

```python
# Train separate models on each subject
for subject in [S1, S2, S3, S4, S5]:
    model_s = xgb.XGBRegressor().fit(X_s[subject], y_s[subject])
    importance_s = model_s.feature_importances_
    
# Compare feature importance distributions
plot_importance_per_subject()
```

**Expected findings:**
- ✅ EDA still dominates (expected, universal stress response)
- ✅ PPG metrics consistent across age
- ⚠️ IMU might vary (young more active, elderly less)
- ⚠️ RR might be different (cardiac subjects have RR abnormalities)

**Implications:**
- Robustness: Features are universal (good for generalization)
- Personalization: Could adjust weights per subject if needed

---

## Phase 7: Real-World Deployment

### Once Multi-Subject Model is Validated

**Deliverables:**

1. **Unified Model**
   ```
   xgboost_borg_multi_subject_v3.json
   - Trained on 5 subjects, 3000+ windows
   - Cross-validated generalization R² = 0.87+
   ```

2. **Deployment Package**
   ```
   - Model weights (JSON)
   - Feature names & scaling parameters
   - Prediction API (inference code)
   - Documentation (input format, output format)
   ```

3. **Real-Time Inference**
   ```python
   # On wearable device or phone
   raw_sensors = read_sensors(window_length=10.0s)
   features = preprocess_and_extract(raw_sensors)
   borg_pred = model.predict(features)
   confidence = estimate_confidence(borg_pred)
   
   display_to_user(f"Effort: {borg_pred:.1f} ± {confidence:.1f}")
   ```

4. **Continuous Monitoring**
   ```
   - Stream Borg predictions at 0.1 Hz (every 10s)
   - Alert if Borg > threshold (e.g., > 15 for cardiac patient)
   - Log predictions for review
   ```

### Use Cases

**Medical:**
- Cardiac rehabilitation: Monitor effort during exercise
- Pulmonary disease: Avoid over-exertion
- Diabetes: Prescribe activity levels

**Consumer:**
- Fitness apps: Alternative to manual Borg input
- Smartwatch: Continuous effort tracking
- Coach feedback: "You're working too hard, slow down"

**Research:**
- Effort-physiology studies
- Autonomic nervous system investigations
- Wearable validation studies

---

## Phase 8: Continuous Improvement Pipeline

### Ongoing Updates

**Monthly:**
- Collect new subject data
- Retrain model on expanded dataset
- Monitor performance metrics
- Flag data quality issues

**Quarterly:**
- Feature engineering review (add new features?)
- Hyperparameter re-optimization
- Error analysis (why are predictions wrong?)
- User feedback incorporation

**Annually:**
- Full cross-subject validation
- Comparison with other effort estimation methods
- Published results / patent filing
- Model versioning (v3.1, v3.2, etc.)

### A/B Testing Framework

```
Production model: v3.0 (current best)
Experimental model: v3.1 (new features or data)

Deploy v3.1 to 10% of users
Monitor:
  - Prediction accuracy
  - User satisfaction
  - Error distribution
  - Edge case handling

If v3.1 better: Promote to 100%
If equal: Keep v3.0
If worse: Debug v3.1
```

---

## Timeline & Resource Estimate

| Phase | Task | Duration | FTE | Cost Est. |
|-------|------|----------|-----|-----------|
| **1** | Current pipeline | Complete | — | — |
| **2** | Data collection (5 subjects, 3000 windows) | 3-6 mo | 0.5 | $5k-10k |
| **3** | Data harmonization & preprocessing | 1-2 mo | 1.0 | $3k-5k |
| **4** | Model retraining & tuning | 1 mo | 1.0 | $2k-3k |
| **5** | Validation & analysis | 1 mo | 1.0 | $2k-3k |
| **6** | Feature analysis | 0.5 mo | 0.5 | $1k-2k |
| **7** | Deployment package | 1 mo | 1.0 | $3k-5k |
| **8** | Continuous improvement infrastructure | 1 mo | 1.0 | $3k-5k |
| | **TOTAL** | **7-9 months** | **5.5** | **$22k-33k** |

---

## Success Metrics

### Phase 2-3 (Data Collection)
- [ ] ≥ 5 subjects enrolled
- [ ] ≥ 3000 labeled windows collected
- [ ] No systematic data quality issues
- [ ] Borg coverage: 0-20 range present

### Phase 4-5 (Retraining)
- [ ] Multi-subject model R² test ≥ 0.88
- [ ] RMSE < 0.6 Borg points
- [ ] Cross-subject validation R² ≥ 0.85
- [ ] No systematic bias across subjects

### Phase 6 (Feature Analysis)
- [ ] Top features consistent across subjects
- [ ] EDA still dominates (>40% importance)
- [ ] No subject is statistical outlier

### Phase 7-8 (Deployment)
- [ ] Model deployed on test device
- [ ] Real-time inference <1ms
- [ ] Documentation complete
- [ ] CI/CD pipeline for retraining

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Recruitment difficulty** | Delay phase 2 | Broad recruitment, incentives |
| **Poor subject generalization** | Invalid phase 5 | Start with similar demographics |
| **Data quality issues** | Phase 3 delays | Strict QC protocol during collection |
| **Model overfits to new data** | Poor phase 5 results | Aggressive regularization, early stopping |
| **Computational constraints** | Phase 4 delays | Use cloud GPUs if needed |
| **Regulatory compliance** | Deployment blockers | Plan IRB approval early |

---

## Future Research Directions

### Beyond V3 (Year 2+)

1. **Personalization**
   - Per-user adaptation (subject-specific models)
   - Online learning (update model from user feedback)

2. **Fusion with Other Signals**
   - Blood pressure waveforms (if available)
   - Respiratory rate (solve non-uniform sampling)
   - Core temperature (deeper physiology)

3. **Advanced Modeling**
   - LSTM/temporal models (use time dynamics)
   - Ensemble methods (combine multiple models)
   - Transfer learning (pre-train on general population)

4. **Interpretability**
   - SHAP values (which features matter per prediction?)
   - Attention mechanisms (visualize model reasoning)
   - Clinical validation (cardiologists review predictions)

5. **Multi-Task Learning**
   - Predict both Borg AND heart rate
   - Shared features, specific heads
   - Improved generalization

---

## Conclusion

**Current Status:** Single-subject v2 model with excellent performance (R²=0.9225) but limited generalizability.

**Next 9 months:** Multi-subject data collection, retraining, and validation to create a robust v3 model ready for real-world deployment.

**Success Criteria:** Cross-subject validation R² ≥ 0.85 on held-out subjects.

**Timeline:** 7-9 months with proper resourcing.

**Impact:** Wearable-based effort estimation tool enabling cardiac rehab, fitness apps, and research applications.

