#!/usr/bin/env python3
"""
Run pipeline for ALL 5 elderly subjects with proper LOSO evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from pathlib import Path
import sys
sys.path.insert(0, '/Users/pascalschlegel/effort-estimator')

from windowing.windows import extract_windows_for_subject
from ml.feature_selection_and_qc import run_feature_selection_and_qc

# =============================================================================
# CONFIG
# =============================================================================
SUBJECTS = [
    ('parsingsim1', 'sim_elderly1'),
    ('parsingsim2', 'sim_elderly2'),
    ('parsingsim3', 'sim_elderly3'),
    ('parsingsim4', 'sim_elderly4'),
    ('parsingsim5', 'sim_elderly5'),
]

WINDOW_SEC = 5.0
OVERLAP = 0.7
DATA_ROOT = Path("/Users/pascalschlegel/data/interim")
OUTPUT_DIR = DATA_ROOT / "all_elderly_combined"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# STEP 1: Extract windows for each subject
# =============================================================================
print("="*70)
print(f"PIPELINE FOR ALL 5 ELDERLY SUBJECTS")
print(f"Window: {WINDOW_SEC}s, Overlap: {OVERLAP*100:.0f}%")
print("="*70)

all_dfs = []

for parsing_dir, subject_id in SUBJECTS:
    print(f"\n📦 Processing {subject_id}...")
    subject_path = DATA_ROOT / parsing_dir / subject_id
    
    if not subject_path.exists():
        print(f"  ⚠️ Path not found: {subject_path}")
        continue
    
    try:
        df = extract_windows_for_subject(
            subject_path=str(subject_path),
            window_sec=WINDOW_SEC,
            overlap=OVERLAP,
            modalities=['eda', 'imu', 'ppg'],
        )
        
        if df is not None and len(df) > 0:
            df['subject'] = subject_id
            df['label'] = subject_id
            all_dfs.append(df)
            print(f"  ✓ Extracted {len(df)} windows")
        else:
            print(f"  ⚠️ No windows extracted")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if not all_dfs:
    print("\n❌ No data extracted!")
    sys.exit(1)

# Combine all subjects
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Combined: {len(combined_df)} total windows from {combined_df['subject'].nunique()} subjects")

# Save combined raw features
combined_path = OUTPUT_DIR / f"all_elderly_features_{WINDOW_SEC}s.csv"
combined_df.to_csv(combined_path, index=False)
print(f"Saved: {combined_path}")

# =============================================================================
# STEP 2: Align with Borg labels
# =============================================================================
print("\n" + "="*70)
print("ALIGNING WITH BORG LABELS")
print("="*70)

# Load Borg labels for each subject
borg_dfs = []
for parsing_dir, subject_id in SUBJECTS:
    # Try different possible label file locations
    label_paths = [
        DATA_ROOT / parsing_dir / subject_id / "labels.csv",
        DATA_ROOT / parsing_dir / "labels.csv",
        Path(f"/Users/pascalschlegel/data/raw/{parsing_dir}/{subject_id}/labels.csv"),
    ]
    
    for lp in label_paths:
        if lp.exists():
            ldf = pd.read_csv(lp)
            ldf['subject'] = subject_id
            borg_dfs.append(ldf)
            print(f"  ✓ Loaded labels for {subject_id}: {len(ldf)} rows")
            break
    else:
        print(f"  ⚠️ No labels found for {subject_id}")

if borg_dfs:
    borg_df = pd.concat(borg_dfs, ignore_index=True)
    print(f"\nTotal label rows: {len(borg_df)}")
    
    # Check for borg column
    borg_col = None
    for col in ['borg', 'borg_cr10', 'effort', 'Borg']:
        if col in borg_df.columns:
            borg_col = col
            break
    
    if borg_col:
        print(f"Using Borg column: {borg_col}")

# =============================================================================
# STEP 3: Feature selection and QC
# =============================================================================
print("\n" + "="*70)
print("FEATURE SELECTION AND QC")
print("="*70)

# Get feature columns (exclude metadata)
meta_cols = ['t_center', 'subject', 'label', 'borg', 'activity', 'modality', 
             'valid', 'n_samples', 'win_sec', 'valid_r', 'n_samples_r', 'win_sec_r']
feature_cols = [c for c in combined_df.columns if c not in meta_cols and not c.endswith('_r')]

print(f"Feature columns: {len(feature_cols)}")

# For now, let's check what Borg data we have in the combined df
if 'borg' in combined_df.columns:
    non_null_borg = combined_df['borg'].notna().sum()
    print(f"Windows with Borg labels: {non_null_borg}")
    
    # Filter to labeled windows
    labeled_df = combined_df.dropna(subset=['borg'])
    print(f"Labeled windows: {len(labeled_df)}")
    
    for subj in labeled_df['subject'].unique():
        subj_data = labeled_df[labeled_df['subject'] == subj]
        print(f"  {subj}: {len(subj_data)} windows, Borg range: {subj_data['borg'].min():.1f}-{subj_data['borg'].max():.1f}")
else:
    print("⚠️ No 'borg' column in combined data - need to align with labels")

# =============================================================================
# STEP 4: Run LOSO evaluation if we have labeled data
# =============================================================================
if 'borg' in combined_df.columns:
    labeled_df = combined_df.dropna(subset=['borg'])
    
    if len(labeled_df) > 0 and labeled_df['subject'].nunique() >= 3:
        print("\n" + "="*70)
        print("LOSO CROSS-VALIDATION")
        print("="*70)
        
        # Simple feature selection - remove constant/nan columns
        X_cols = [c for c in feature_cols if c in labeled_df.columns]
        X_df = labeled_df[X_cols].copy()
        
        # Drop columns with any NaN or constant values
        X_df = X_df.dropna(axis=1)
        X_df = X_df.loc[:, X_df.std() > 1e-6]
        
        X = X_df.values
        y = labeled_df['borg'].values
        subjects = labeled_df['subject'].values
        
        print(f"\nFeatures after cleaning: {X.shape[1]}")
        print(f"Samples: {len(y)}")
        print(f"Subjects: {np.unique(subjects)}")
        
        # LOSO CV
        print("\nLOSO Results:")
        print("-"*50)
        
        results = []
        for test_subj in np.unique(subjects):
            train_mask = subjects != test_subj
            test_mask = subjects == test_subj
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(X_test) < 10:
                print(f"  {test_subj}: skipped (only {len(X_test)} samples)")
                continue
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_s, y_train)
            y_pred_ridge = ridge.predict(X_test_s)
            r_ridge, _ = pearsonr(y_test, y_pred_ridge)
            mae_ridge = np.abs(y_test - y_pred_ridge).mean()
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
            xgb_model.fit(X_train_s, y_train)
            y_pred_xgb = xgb_model.predict(X_test_s)
            r_xgb, _ = pearsonr(y_test, y_pred_xgb)
            mae_xgb = np.abs(y_test - y_pred_xgb).mean()
            
            results.append({
                'subject': test_subj,
                'n_test': len(y_test),
                'ridge_r': r_ridge,
                'ridge_mae': mae_ridge,
                'xgb_r': r_xgb,
                'xgb_mae': mae_xgb,
            })
            
            print(f"  {test_subj}: Ridge r={r_ridge:.3f}, XGB r={r_xgb:.3f} (n={len(y_test)})")
        
        if results:
            results_df = pd.DataFrame(results)
            print("\n" + "="*70)
            print("SUMMARY (LOSO AVERAGE)")
            print("="*70)
            print(f"Ridge:   r={results_df['ridge_r'].mean():.3f}, MAE={results_df['ridge_mae'].mean():.2f}")
            print(f"XGBoost: r={results_df['xgb_r'].mean():.3f}, MAE={results_df['xgb_mae'].mean():.2f}")
            
            results_df.to_csv(OUTPUT_DIR / "loso_results_5subjects.csv", index=False)
            print(f"\nSaved: {OUTPUT_DIR}/loso_results_5subjects.csv")

print("\n✅ Done!")
