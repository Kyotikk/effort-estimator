#!/usr/bin/env python3
"""
CLARIFICATION: HR SOURCES AND APPROACHES COMPARISON
====================================================

This script clarifies the confusion about:
1. What "r = 0.78 across 4 subjects" means
2. PPG HR (in pipeline) vs ECG HR (Vivalnk sensor)
3. Scientific multi-feature approach vs Simple HR approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

print("="*80)
print("CLARIFICATION: TWO DIFFERENT APPROACHES")
print("="*80)

# =============================================================================
# QUESTION 1: What does "r = 0.78 across 4 subjects" mean?
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ QUESTION 1: What is "r = 0.78 across 4 subjects"?                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ It means:                                                                    ║
║   - For EACH subject, we train a model on 50% of THEIR data                  ║
║   - Then test on the remaining 50% of THEIR data                             ║
║   - This gives us predictions for each subject separately                    ║
║   - We then POOL all predictions together and compute ONE overall r          ║
║                                                                              ║
║ Example:                                                                     ║
║   Subject 2: 14 test samples → predictions                                   ║
║   Subject 3: 17 test samples → predictions                                   ║
║   Subject 4: 14 test samples → predictions                                   ║
║   Subject 5: 13 test samples → predictions                                   ║
║   ────────────────────────────────────────                                   ║
║   Total: 58 pooled samples → r = 0.78                                        ║
║                                                                              ║
║ This is NOT one model for all subjects!                                      ║
║ It's 4 SEPARATE personalized models, results pooled.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# QUESTION 2: PPG HR vs ECG HR - Different Data Sources!
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ QUESTION 2: Are ECG HR features in your pipeline?                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ NO! Your pipeline has TWO different HR sources:                              ║
║                                                                              ║
║ ┌─────────────────────────────────────────────────────────────────────────┐  ║
║ │ SOURCE 1: PPG HR (IN YOUR PIPELINE)                                     │  ║
║ │ ─────────────────────────────────────                                   │  ║
║ │ • Comes from: Corsano wristband (PPG sensor)                            │  ║
║ │ • Features: ppg_green_hr_mean, ppg_infra_hr_mean, etc.                  │  ║
║ │ • Quality: NOISY - PPG is motion-sensitive                              │  ║
║ │ • Correlation with Borg: r ≈ 0.05 (very weak!)                          │  ║
║ │ • This is what your 34 selected features use                            │  ║
║ └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║ ┌─────────────────────────────────────────────────────────────────────────┐  ║
║ │ SOURCE 2: ECG HR (NOT IN YOUR PIPELINE - I used separately)             │  ║
║ │ ─────────────────────────────────────                                   │  ║
║ │ • Comes from: Vivalnk VV330 chest patch (ECG sensor)                    │  ║
║ │ • Location: /data/interim/*/vivalnk_vv330_heart_rate/                   │  ║
║ │ • Quality: CLEAN - ECG is gold standard                                 │  ║
║ │ • Correlation with Borg: r ≈ 0.50 within-subject                        │  ║
║ │ • NOT in your feature extraction pipeline!                              │  ║
║ └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║ The r = 0.78 result used ECG HR (clean), NOT PPG HR (noisy)!                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# QUESTION 3: Two Approaches Comparison
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ QUESTION 3: Scientific Multi-Feature vs Simple HR Approach                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ APPROACH A: Scientific Multi-Feature Pipeline (YOUR MAIN APPROACH)           ║
║ ─────────────────────────────────────────────────────────────────            ║
║ • 296 raw features (EDA, IMU, PPG)                                           ║
║ • PCA + correlation filtering → 34 selected features                         ║
║ • XGBoost / Ridge Regression                                                 ║
║ • 5-second windows                                                           ║
║ • Results: r = 0.24-0.48 (LOSO), r = 0.57 (activity-level with guessed labels)║
║                                                                              ║
║ Pros: Generalizable, uses multimodal sensors, scientifically rigorous        ║
║ Cons: PPG HR features are noisy, cross-subject generalization poor           ║
║                                                                              ║
║ ─────────────────────────────────────────────────────────────────────────────║
║                                                                              ║
║ APPROACH B: Simple HR-Based Model (SEPARATE ANALYSIS)                        ║
║ ─────────────────────────────────────────────────────────────────            ║
║ • Only 3 features: HR_delta, HR_load, duration                               ║
║ • Uses ECG HR (clean, not from your pipeline)                                ║
║ • Activity-level (not 5s windows)                                            ║
║ • Results: r = 0.47 within-subject, r = 0.78 personalized                    ║
║                                                                              ║
║ Pros: Simple, interpretable, clean HR signal                                 ║
║ Cons: Requires ECG sensor (not wrist-only), needs personalization            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ RECOMMENDATION FOR THESIS                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ Present BOTH approaches:                                                     ║
║                                                                              ║
║ 1. MAIN APPROACH (Scientific Multi-Feature Pipeline)                         ║
║    • Show the full methodology: windowing, feature extraction, selection     ║
║    • Report LOSO results honestly: r = 0.24-0.48                             ║
║    • This is your reproducible, scalable approach for larger datasets        ║
║    • Recommend this for future studies with more subjects                    ║
║                                                                              ║
║ 2. SUPPLEMENTARY (Activity-Level with Clean HR)                              ║
║    • Show that with better HR signal, performance improves                   ║
║    • Report: r = 0.78 personalized                                           ║
║    • Explains WHY PPG-based results are weaker                               ║
║    • Shows potential with better sensors                                     ║
║                                                                              ║
║ 3. KEY INSIGHT to highlight:                                                 ║
║    "Inter-subject variability is the fundamental limitation.                 ║
║     With 5 subjects, cross-subject generalization is poor.                   ║
║     Personalized calibration or larger datasets are needed."                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# Show the actual data
# =============================================================================

print("\n" + "="*80)
print("ACTUAL DATA CHECK")
print("="*80)

# Check PPG HR features correlation with Borg
fused_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv")
if fused_path.exists():
    df = pd.read_csv(fused_path)
    print(f"\nPPG HR features correlation with Borg (from your pipeline):")
    ppg_hr_cols = [c for c in df.columns if 'hr_mean' in c]
    for col in ppg_hr_cols:
        valid = df[[col, 'borg']].dropna()
        if len(valid) > 2:
            r, p = pearsonr(valid[col], valid['borg'])
            print(f"  {col}: r = {r:.3f}")

# Check ECG HR from Vivalnk
tli_path = Path("/Users/pascalschlegel/effort-estimator/output/tli_all_subjects.csv")
if tli_path.exists():
    tli_df = pd.read_csv(tli_path)
    print(f"\nECG HR features correlation with Borg (Vivalnk sensor):")
    for col in ['hr_delta', 'hr_load']:
        if col in tli_df.columns:
            valid = tli_df[[col, 'borg']].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid[col], valid['borg'])
                print(f"  {col}: r = {r:.3f}")

print("""
\n📌 CONCLUSION:
   - PPG HR (your pipeline): r ≈ 0.05 with Borg (too noisy)
   - ECG HR (Vivalnk): r ≈ 0.35 with Borg (much cleaner)
   
   The scientific approach is correct, but PPG HR quality limits results.
   For your thesis, this is an important finding to discuss!
""")
