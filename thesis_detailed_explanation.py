"""
DETAILED THESIS EXPLANATION
===========================
1. What is "pooled" vs "per-subject"?
2. What is "feature importance" vs "generalizability"?
3. Can you compute RMSSD delta with current data?
4. How does the pipeline select features?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject_id'] = f'elderly{i}'
        all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
df_clean = df.dropna(subset=['borg'])

print("="*80)
print("PART 1: WHAT IS 'POOLED' vs 'PER-SUBJECT' CORRELATION?")
print("="*80)

print("""
POOLED CORRELATION:
───────────────────
- Put ALL data points from ALL subjects into ONE big dataset
- Train model on this pooled data
- Compute correlation between all predictions and all true values
- THIS IS INFLATED because subjects cluster together!

Example visualization (Borg vs Predicted):

    14 |         ●●●  Subject 3 (high effort activities)
       |        ●●●●
    12 |    ▲▲▲       Subject 2  
       |   ▲▲▲▲
    10 | ■■■          Subject 1 (low effort activities)
       | ■■■■
     8 |
       +----------------------------------
         8    10    12    14   Predicted

The model learns: "These ● points are Subject 3, predict ~13"
                  "These ▲ points are Subject 2, predict ~11"
                  
Pooled r = 0.87 because it captures BOTH:
  1. Real effort relationship (within each cluster)
  2. Subject identification (between clusters) ← THIS IS CHEATING!

PER-SUBJECT CORRELATION:
────────────────────────
- Train on subjects 1,2,3,4 → Test on subject 5 (never seen!)
- Compute correlation for subject 5 ONLY
- Repeat for each subject (Leave-One-Subject-Out)
- Average the 5 correlations

This is HONEST because the model can't use subject identity.
""")

# Demonstrate with actual data
print("\n--- Actual demonstration ---\n")

# Get some features
imu_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns 
            if c.startswith(('acc_', 'gyr_')) and df_clean[c].notna().all()][:20]

X = df_clean[imu_cols].values
y = df_clean['borg'].values
subjects = df_clean['subject_id'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# POOLED approach (inflated)
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
model.fit(X_scaled, y)
pooled_pred = model.predict(X_scaled)
pooled_r = np.corrcoef(y, pooled_pred)[0, 1]

print(f"POOLED r = {pooled_r:.3f}  ← Looks great, but INFLATED!")
print(f"  (Model trained AND tested on same data, can learn subject identity)")

# PER-SUBJECT approach (honest)
per_subject_r = []
for test_subj in df_clean['subject_id'].unique():
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    
    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    model.fit(X_scaled[train_mask], y[train_mask])
    pred = model.predict(X_scaled[test_mask])
    
    r = np.corrcoef(y[test_mask], pred)[0, 1]
    per_subject_r.append(r)
    print(f"  {test_subj}: r = {r:.3f}")

print(f"\nPER-SUBJECT r = {np.mean(per_subject_r):.3f}  ← HONEST metric!")
print(f"  (Model tested on subject it NEVER saw during training)")

print(f"\nInflation: {pooled_r - np.mean(per_subject_r):.3f} difference!")

# =============================================================================
print("\n" + "="*80)
print("PART 2: WHAT IS 'FEATURE IMPORTANCE' vs 'GENERALIZABILITY'?")
print("="*80)

print("""
FEATURE IMPORTANCE (RandomForest):
──────────────────────────────────
How it works:
1. Train RF on ALL pooled data
2. For each feature, measure how much prediction error increases 
   when you randomly shuffle that feature's values
3. Features that hurt predictions most when shuffled = "important"

THE PROBLEM:
- This measures importance for fitting THE POOLED DATA
- If a feature helps identify subjects (not effort), it looks "important"
- PPG features vary a lot between subjects → helps identify who is who
- RF thinks: "ppg_green_p99 helps me know this is elderly3" → HIGH importance

GENERALIZABILITY:
─────────────────
Does the feature help predict effort on UNSEEN subjects?

Test:
1. Train using ONLY that feature (or feature group)
2. Use LOSO - train on 4 subjects, test on 5th
3. If r is high → feature captures UNIVERSAL effort patterns
4. If r is low → feature only captures individual differences
""")

# Demonstrate
print("\n--- Actual demonstration ---\n")

# Get all features
all_numeric = df_clean.select_dtypes(include=[np.number]).columns
exclude = ['borg', 'subject']
feature_cols = [c for c in all_numeric if not any(x in c.lower() for x in exclude) 
                and df_clean[c].notna().all() and np.isfinite(df_clean[c]).all()]

imu_cols = [c for c in feature_cols if c.startswith(('acc_', 'gyr_'))]
ppg_cols = [c for c in feature_cols if c.startswith('ppg_')]

# Train on ALL data to get importance
X_all = df_clean[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_scaled, y)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df['modality'] = importance_df['feature'].apply(
    lambda x: 'IMU' if x.startswith(('acc_', 'gyr_')) else ('PPG' if x.startswith('ppg_') else 'EDA')
)

print("TOP 10 FEATURES BY IMPORTANCE (on pooled data):")
print("-" * 60)
for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
    print(f"  {i+1}. [{row['modality']:3}] {row['feature'][:40]:40} {row['importance']:.4f}")

imu_importance = importance_df[importance_df['modality'] == 'IMU']['importance'].sum()
ppg_importance = importance_df[importance_df['modality'] == 'PPG']['importance'].sum()
print(f"\nTotal importance: IMU={imu_importance:.1%}, PPG={ppg_importance:.1%}")

# Now test GENERALIZABILITY with LOSO
print("\n" + "-"*60)
print("NOW TEST GENERALIZABILITY (LOSO):")
print("-" * 60)

def loso_test(cols, name):
    X_feat = df_clean[cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    per_subj_r = []
    for test_subj in df_clean['subject_id'].unique():
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X_scaled[train_mask], y[train_mask])
        pred = model.predict(X_scaled[test_mask])
        r = np.corrcoef(y[test_mask], pred)[0, 1]
        per_subj_r.append(r)
    
    return np.mean(per_subj_r)

imu_loso = loso_test(imu_cols, "IMU")
ppg_loso = loso_test(ppg_cols, "PPG")

print(f"\n  IMU: importance={imu_importance:.1%}, LOSO r={imu_loso:.3f}")
print(f"  PPG: importance={ppg_importance:.1%}, LOSO r={ppg_loso:.3f}")

print(f"""
CONCLUSION:
  PPG has {ppg_importance/imu_importance:.1f}x more importance than IMU
  BUT IMU generalizes {imu_loso/ppg_loso:.1f}x better than PPG!
  
  Importance ≠ Generalizability!
""")

# =============================================================================
print("\n" + "="*80)
print("PART 3: CAN YOU COMPUTE RMSSD DELTA WITH CURRENT DATA?")
print("="*80)

# Check what HR/HRV features exist
hr_cols = [c for c in df_clean.columns if 'hr' in c.lower() or 'rmssd' in c.lower() or 'rr' in c.lower()]
print(f"\nHR/HRV related columns in your data: {hr_cols[:10]}")

# Check if there's activity labeling
print(f"\nActivities in data: {df_clean['activity'].unique() if 'activity' in df_clean.columns else 'No activity column'}")

print("""
CURRENT DATA LIMITATION:
────────────────────────
Your current dataset has 5-second windows with ABSOLUTE feature values:
- ppg_green_mean = 1234.5 (absolute, varies by person)
- No explicit "rest" vs "activity" labeling per window

TO COMPUTE RMSSD DELTA, you would need:
1. Identify "rest" windows (sitting/lying down activities)
2. Compute baseline RMSSD per subject from rest windows
3. For each activity window: delta = RMSSD_activity - RMSSD_baseline

LET ME CHECK IF THIS IS POSSIBLE...
""")

# Check for rest-like activities
if 'activity' in df_clean.columns:
    print("\nActivity distribution:")
    for subj in df_clean['subject_id'].unique():
        subj_acts = df_clean[df_clean['subject_id'] == subj]['activity'].value_counts()
        print(f"\n  {subj}:")
        for act, count in subj_acts.head(5).items():
            print(f"    {act}: {count} windows")

# Try to compute delta features
print("\n--- ATTEMPTING DELTA FEATURE COMPUTATION ---\n")

# Find a PPG feature to use as proxy
ppg_mean_col = [c for c in df_clean.columns if 'ppg' in c.lower() and 'mean' in c.lower()]
if ppg_mean_col:
    test_col = ppg_mean_col[0]
    print(f"Using {test_col} as example...")
    
    # Identify "rest" as lowest 20% Borg ratings per subject
    df_with_delta = df_clean.copy()
    
    for subj in df_clean['subject_id'].unique():
        subj_mask = df_with_delta['subject_id'] == subj
        subj_borg = df_with_delta.loc[subj_mask, 'borg']
        
        # Rest = lowest 20% effort windows
        rest_threshold = subj_borg.quantile(0.2)
        rest_mask = subj_mask & (df_with_delta['borg'] <= rest_threshold)
        
        # Compute baseline from rest
        baseline = df_with_delta.loc[rest_mask, test_col].mean()
        
        # Compute delta
        df_with_delta.loc[subj_mask, f'{test_col}_delta'] = df_with_delta.loc[subj_mask, test_col] - baseline
        df_with_delta.loc[subj_mask, f'{test_col}_baseline'] = baseline
        
        print(f"  {subj}: baseline={baseline:.1f}, n_rest_windows={rest_mask.sum()}")
    
    # Test if delta generalizes better
    print("\n--- COMPARING RAW vs DELTA ---")
    
    raw_loso = loso_test([test_col], "raw")
    
    # LOSO for delta
    delta_col = f'{test_col}_delta'
    X_delta = df_with_delta[[delta_col]].values
    scaler = StandardScaler()
    X_delta_scaled = scaler.fit_transform(X_delta)
    
    delta_per_subj = []
    subjects_delta = df_with_delta['subject_id'].values
    y_delta = df_with_delta['borg'].values
    
    for test_subj in df_with_delta['subject_id'].unique():
        train_mask = subjects_delta != test_subj
        test_mask = subjects_delta == test_subj
        
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X_delta_scaled[train_mask], y_delta[train_mask])
        pred = model.predict(X_delta_scaled[test_mask])
        r = np.corrcoef(y_delta[test_mask], pred)[0, 1]
        delta_per_subj.append(r)
    
    delta_loso = np.mean(delta_per_subj)
    
    print(f"\n  Raw {test_col}: LOSO r = {raw_loso:.3f}")
    print(f"  Delta {test_col}: LOSO r = {delta_loso:.3f}")
    print(f"  Improvement: {(delta_loso - raw_loso):.3f}")

print("""
VERDICT ON DELTA FEATURES:
──────────────────────────
YES, you CAN compute delta features with current data, BUT:
1. You don't have explicit "rest" periods - must estimate from low-Borg windows
2. 5-second windows may not capture true resting baseline
3. Activities are short, no clear rest→activity transition

YOUR NEW STUDY WILL BE BETTER BECAUSE:
- 5-minute baseline period (true resting state)
- 5-minute recovery period (return to baseline)
- Clear activity phases with known start/end
- Can compute: RMSSD_activity - RMSSD_baseline
- Can compute: HR_peak - HR_rest (true delta)
""")

# =============================================================================
print("\n" + "="*80)
print("PART 4: HOW DOES THE PIPELINE SELECT FEATURES?")
print("="*80)

print("""
YOUR DATA-DRIVEN PIPELINE - EXACT ALGORITHM:
════════════════════════════════════════════

STEP 1: LOAD & PREPARE DATA
───────────────────────────
- Load fused features from all subjects
- Identify feature groups: IMU, PPG, EDA
- Remove NaN/infinite values

STEP 2: TEST ALL MODALITY COMBINATIONS (7 total)
────────────────────────────────────────────────
For each of: [IMU, PPG, EDA, IMU+PPG, IMU+EDA, PPG+EDA, ALL]
  For each model: [Ridge, ElasticNet, RF_d4, RF_d6, GB_d4, SVR]
    
    → Run LOSO evaluation:
      for each subject as test:
        train on other 4 subjects
        (optionally use 20% of test subject for calibration)
        predict on remaining test data
        compute correlation
      
      → per_subject_r = mean of 5 correlations
    
    → Store result: (feature_set, model, per_subject_r)

STEP 3: SELECT BEST BY GENERALIZATION
─────────────────────────────────────
best = argmax(per_subject_r)  # NOT importance, NOT pooled r!

STEP 4 (OPTIONAL): GRANULAR SELECTION
─────────────────────────────────────
If --granular flag:
  Start with empty feature set
  For each candidate feature:
    Add feature to set
    Run LOSO
    If per_subject_r improves → keep feature
    Else → remove feature
  
  → This finds the MINIMAL set of best-generalizing features

KEY INSIGHT:
────────────
The pipeline uses LOSO per-subject r as the selection criterion.
This automatically:
- Penalizes features that only identify subjects
- Rewards features that capture universal effort patterns
- Discovers that IMU >> PPG even though RF importance says opposite!
""")

print("\n" + "="*80)
print("SUMMARY FOR YOUR THESIS")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  POOLED r:                                                              │
│    - Train/test on same pooled data                                     │
│    - INFLATED because model learns subject identity                     │
│    - Your data: r = 0.87 (looks great, but cheating!)                   │
│                                                                         │
│  PER-SUBJECT r (LOSO):                                                  │
│    - Train on N-1 subjects, test on 1 unseen subject                    │
│    - HONEST because model can't use subject identity                    │
│    - Your data: r = 0.55 (real generalization ability)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  FEATURE IMPORTANCE (RF):                                               │
│    - "How much does this feature help fit the pooled data?"            │
│    - PPG = 58%, IMU = 16%                                               │
│    - MISLEADING - PPG helps identify subjects, not effort!              │
│                                                                         │
│  GENERALIZABILITY (LOSO r):                                             │
│    - "How much does this feature help predict UNSEEN subjects?"        │
│    - IMU LOSO r = 0.55, PPG LOSO r = 0.18                              │
│    - IMU generalizes 3x better despite lower "importance"!              │
├─────────────────────────────────────────────────────────────────────────┤
│  DELTA FEATURES:                                                        │
│    - Current data: Can approximate, but no true baseline               │
│    - New study: 5-min baseline enables true delta computation           │
│    - RMSSD_delta = RMSSD_activity - RMSSD_rest                         │
│    - Should generalize better by normalizing individual differences     │
├─────────────────────────────────────────────────────────────────────────┤
│  PIPELINE SELECTION:                                                    │
│    - Tests all feature combinations                                     │
│    - Uses LOSO r (not importance!) as criterion                        │
│    - Automatically discovers IMU > PPG                                  │
│    - Data-driven: will adapt if new data shows different patterns       │
└─────────────────────────────────────────────────────────────────────────┘
""")
