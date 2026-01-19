# Pipeline Script Analysis

## ✅ ACTIVE (Currently Used)

### Core Pipeline Scripts
- **`run_multisub_pipeline.py`** - Main orchestrator for multi-subject pipeline
  - Imports: fusion, alignment, quality_check, feature_selection modules
  - Combines subjects, runs feature selection, generates outputs
  - Status: **ACTIVELY USED** ✓

- **`run_pipeline.py`** - Single-subject full pipeline
  - Preprocessing → windowing → feature extraction → fusion → alignment
  - Status: **ACTIVELY USED** ✓
  - Called by run_multisub_pipeline.py via subprocess

- **`train_multisub_xgboost.py`** - Multi-subject model training
  - Trains XGBoost, generates 7 diagnostic plots
  - Status: **ACTIVELY USED** ✓

### Modular ML Components (NEW - Now Used)
- **`ml/alignment.py`** - Temporal alignment module
  - Functions: `align_fused_data()`, `save_aligned_data()`
  - Status: **IMPORTED & USED** in run_multisub_pipeline.py ✓

- **`ml/fusion.py`** - Feature fusion module
  - Functions: `fuse_modalities()`, `save_fused_data()`
  - Status: **IMPORTED & USED** in run_multisub_pipeline.py ✓
  - Also used by run_pipeline.py via ml.run_fusion

- **`ml/quality_check.py`** - Data quality validation
  - Functions: `check_data_quality()`, `print_qc_results()`
  - Status: **IMPORTED & USED** in run_multisub_pipeline.py ✓

- **`ml/feature_selection.py`** - Feature selection orchestration
  - Functions: `select_features()`, `save_feature_selection_outputs()`
  - Status: **IMPORTED & USED** in run_multisub_pipeline.py ✓

- **`ml/feature_selection_and_qc.py`** - Feature selection backend (LEGACY)
  - Functions: `select_and_prune_features()`, `perform_pca_analysis()`, etc.
  - Status: **ACTIVELY USED** (called by feature_selection.py) ✓

### Dependency Modules (Always Needed)
- **`ml/targets/`** - Target alignment utilities
  - Used by: run_pipeline.py
  - Status: **REQUIRED** ✓

- **`ml/fusion/`** - Fusion utilities
  - Used by: ml.run_fusion (called by run_pipeline.py)
  - Status: **REQUIRED** ✓

- **`ml/features/`** - Feature sanitization
  - Used by: ml.run_fusion (called by run_pipeline.py)
  - Status: **REQUIRED** ✓

- **`ml/scalers/`** - Scaling utilities
  - Status: **REQUIRED** (dependency) ✓

- **`ml/time/`** - Time utilities
  - Status: **REQUIRED** (dependency) ✓

### Legacy Execution Scripts
- **`ml/run_fusion.py`** - Direct fusion runner
  - Status: **LEGACY** - Still used by run_pipeline.py but wrapped by new ml.fusion module
  - Can stay for backward compatibility but not needed for main pipeline

- **`ml/train_and_save_all.py`** - Single-subject training alternative
  - Status: **ALTERNATIVE** - Not used in current pipeline but available for single-subject work
  - Can keep for reference

---

## ❌ UNUSED (Can Remove or Archive)

### Placeholder/Empty Scripts
- **`ml/feature_extraction.py`** - Feature extraction orchestration
  - Status: **PLACEHOLDER** - Created but not actively used (preprocessing is in run_pipeline.py)
  - Functions are no-ops
  - **RECOMMENDATION: REMOVE** (adds noise, no functionality)

### What's NOT in Use
- Any test files in ml/ directory (if they exist)
- Any debug/analysis scripts (if they exist)

---

## Summary Table

| Script | Location | Status | Used By | Keep? |
|--------|----------|--------|---------|-------|
| run_multisub_pipeline.py | Root | **ACTIVE** | Main entry point | ✓ YES |
| run_pipeline.py | Root | **ACTIVE** | run_multisub_pipeline.py | ✓ YES |
| train_multisub_xgboost.py | Root | **ACTIVE** | User entry point | ✓ YES |
| ml/alignment.py | ml/ | **ACTIVE** | run_multisub_pipeline.py | ✓ YES |
| ml/fusion.py | ml/ | **ACTIVE** | run_multisub_pipeline.py, run_pipeline.py | ✓ YES |
| ml/quality_check.py | ml/ | **ACTIVE** | run_multisub_pipeline.py | ✓ YES |
| ml/feature_selection.py | ml/ | **ACTIVE** | run_multisub_pipeline.py | ✓ YES |
| ml/feature_selection_and_qc.py | ml/ | **ACTIVE** | ml/feature_selection.py | ✓ YES |
| ml/run_fusion.py | ml/ | **LEGACY** | run_pipeline.py (via import) | ~ KEEP |
| ml/train_and_save_all.py | ml/ | **ALTERNATIVE** | None (backup option) | ~ KEEP |
| ml/feature_extraction.py | ml/ | **PLACEHOLDER** | None | ✗ REMOVE |
| ml/targets/ | ml/ | **REQUIRED** | run_pipeline.py | ✓ YES |
| ml/fusion/ | ml/ | **REQUIRED** | ml/run_fusion.py | ✓ YES |
| ml/features/ | ml/ | **REQUIRED** | ml/run_fusion.py | ✓ YES |
| ml/scalers/ | ml/ | **REQUIRED** | Dependency | ✓ YES |
| ml/time/ | ml/ | **REQUIRED** | Dependency | ✓ YES |

---

## Actual Data Flow (What's Really Used)

```
┌────────────────────────────────────────────┐
│ User runs: python run_multisub_pipeline.py  │
└───────────────┬────────────────────────────┘
                │
    ┌───────────┴──────────┬──────────────────┐
    │                      │                  │
    ▼                      ▼                  ▼
Per Subject: run_pipeline.py (3 times)
    │
    ├─→ ml.targets (alignment)
    ├─→ ml.run_fusion
    │   ├─→ ml.fusion.fuse_windows
    │   └─→ ml.features.sanitise
    └─→ ml.feature_selection_and_qc
         (only if run as standalone)
         
[Outputs: fused_aligned_10.0s.csv per subject]
         │
         ▼
    Combine subjects (concat CSV)
         │
         ▼
    ml.quality_check.check_data_quality()
         │
         ▼
    ml.feature_selection.select_features()
         │
         ├─→ ml.feature_selection_and_qc (backend)
         └─→ PCA analysis + save outputs
         
[Outputs: features_selected_pruned.csv]
         │
         ▼
    python train_multisub_xgboost.py
         │
         └─→ Train final model + generate plots
```

---

## Recommendations

### Keep (Active)
✓ All files in the flow diagram above  
✓ All in ml/targets, ml/fusion, ml/features, ml/scalers, ml/time

### Remove (Not Used)
✗ `ml/feature_extraction.py` - Placeholder with no functionality

### Archive (Legacy but Keep)
~ `ml/run_fusion.py` - Still used by run_pipeline.py, keep for compatibility
~ `ml/train_and_save_all.py` - Backup single-subject trainer, keep for reference

---

## Clean Code Overview

**Current system is lean:**
- 3 main scripts (run_multisub_pipeline, run_pipeline, train_multisub_xgboost)
- 4 modular components (fusion, alignment, quality_check, feature_selection)
- 5 dependency folders (targets, fusion, features, scalers, time)
- 1 backend module (feature_selection_and_qc)
- 1 legacy runner (run_fusion)
- 1 alternative trainer (train_and_save_all)

**Only truly unused:**
- `ml/feature_extraction.py` (placeholder)
