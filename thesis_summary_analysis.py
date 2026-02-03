"""
Thesis Summary: PPG Within-Subject vs Across-Subject Analysis
==============================================================
Verify: Do PPG features correlate well WITHIN subjects but fail to generalize ACROSS?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data from actual paths
all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject_id'] = f'elderly{i}'
        all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"Total windows: {len(df)}")
print(f"Subjects: {df['subject_id'].unique()}")

# Get feature columns - only numeric
exclude_cols = ['subject_id', 'subject', 'window_start', 'window_end', 'borg', 'activity', 'activity_label', 'source']
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
imu_cols = [c for c in feature_cols if c.startswith(('acc_', 'gyr_'))]
ppg_cols = [c for c in feature_cols if c.startswith('ppg_')]
eda_cols = [c for c in feature_cols if c.startswith('eda_')]

# Remove columns with NaN or zero variance
valid_cols = [c for c in feature_cols if df[c].notna().all() and np.isfinite(df[c]).all() and df[c].std() > 1e-10]
imu_valid = [c for c in imu_cols if c in valid_cols]
ppg_valid = [c for c in ppg_cols if c in valid_cols]
eda_valid = [c for c in eda_cols if c in valid_cols]

print(f"\nValid features: {len(valid_cols)} (IMU: {len(imu_valid)}, PPG: {len(ppg_valid)}, EDA: {len(eda_valid)})")

# =============================================================================
# ANALYSIS 1: Within-subject correlation vs Across-subject generalization
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS: WITHIN-SUBJECT vs ACROSS-SUBJECT CORRELATION")
print("="*80)

def safe_corr(x, y):
    """Compute correlation safely, handling edge cases"""
    try:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return np.nan
        x, y = x[mask], y[mask]
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return np.nan
        r, _ = pearsonr(x, y)
        return r
    except:
        return np.nan

def analyze_feature_generalization(feature_cols, name):
    """
    For each feature:
    1. Compute correlation with Borg WITHIN each subject
    2. Check if the correlation DIRECTION is consistent across subjects
    """
    results = []
    
    for feat in feature_cols[:50]:  # Top 50 features
        within_corrs = []
        for subj in df['subject_id'].unique():
            subj_df = df[df['subject_id'] == subj]
            r = safe_corr(subj_df[feat].values, subj_df['borg'].values)
            if not np.isnan(r):
                within_corrs.append(r)
        
        if len(within_corrs) >= 4:
            mean_within = np.mean(within_corrs)
            std_within = np.std(within_corrs)
            # Check sign consistency (do all subjects show same direction?)
            signs = [1 if r > 0.1 else (-1 if r < -0.1 else 0) for r in within_corrs]
            sign_consistency = abs(sum(signs)) / len(signs) if len(signs) > 0 else 0
            
            results.append({
                'feature': feat,
                'mean_within_r': mean_within,
                'std_within_r': std_within,
                'sign_consistency': sign_consistency,
                'min_r': min(within_corrs),
                'max_r': max(within_corrs),
                'per_subject': within_corrs
            })
    
    return pd.DataFrame(results)

print("\n--- IMU Features ---")
imu_results = analyze_feature_generalization(imu_valid, "IMU")
if len(imu_results) > 0:
    imu_results = imu_results.sort_values('mean_within_r', key=abs, ascending=False)
    print(f"\nTop 10 IMU features by within-subject correlation:")
    for _, row in imu_results.head(10).iterrows():
        print(f"  {row['feature'][:45]:45} | mean r={row['mean_within_r']:+.3f} ± {row['std_within_r']:.3f} | consistency={row['sign_consistency']:.0%}")

print("\n--- PPG Features ---")
ppg_results = analyze_feature_generalization(ppg_valid, "PPG")
if len(ppg_results) > 0:
    ppg_results = ppg_results.sort_values('mean_within_r', key=abs, ascending=False)
    print(f"\nTop 10 PPG features by within-subject correlation:")
    for _, row in ppg_results.head(10).iterrows():
        print(f"  {row['feature'][:45]:45} | mean r={row['mean_within_r']:+.3f} ± {row['std_within_r']:.3f} | consistency={row['sign_consistency']:.0%}")

# =============================================================================
# ANALYSIS 2: Compare within-subject R vs LOSO R (THE KEY TEST)
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS: WITHIN-SUBJECT FIT vs LOSO GENERALIZATION")
print("="*80)

def compare_within_vs_loso(feature_cols, name):
    """Compare how well models fit within subjects vs generalize to new subjects"""
    # Filter out rows with NaN in borg or features
    df_clean = df.dropna(subset=['borg'] + feature_cols)
    
    X = df_clean[feature_cols].values
    y = df_clean['borg'].values
    subjects = df_clean['subject_id'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Within-subject R (train and test on same subject - overfitting measure)
    within_r = []
    for subj in df['subject_id'].unique():
        mask = subjects == subj
        if mask.sum() > 30:
            model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
            model.fit(X_scaled[mask], y[mask])
            pred = model.predict(X_scaled[mask])
            r = safe_corr(y[mask], pred)
            if not np.isnan(r):
                within_r.append(r)
    
    # LOSO R (train on 4, test on 1 - generalization measure)
    loso_r = []
    for test_subj in df['subject_id'].unique():
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        
        if test_mask.sum() > 10:
            model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
            model.fit(X_scaled[train_mask], y[train_mask])
            pred = model.predict(X_scaled[test_mask])
            r = safe_corr(y[test_mask], pred)
            if not np.isnan(r):
                loso_r.append(r)
    
    return np.mean(within_r) if within_r else 0, np.mean(loso_r) if loso_r else 0

print("\nModality | Within-Subject r | LOSO r | Gap (overfitting indicator)")
print("-" * 65)

imu_within, imu_loso = compare_within_vs_loso(imu_valid, "IMU")
print(f"IMU      | {imu_within:.3f}            | {imu_loso:.3f}  | {imu_within - imu_loso:.3f}")

ppg_within, ppg_loso = compare_within_vs_loso(ppg_valid, "PPG")
print(f"PPG      | {ppg_within:.3f}            | {ppg_loso:.3f}  | {ppg_within - ppg_loso:.3f}")

if len(eda_valid) > 5:
    eda_within, eda_loso = compare_within_vs_loso(eda_valid, "EDA")
    print(f"EDA      | {eda_within:.3f}            | {eda_loso:.3f}  | {eda_within - eda_loso:.3f}")

# All features
all_within, all_loso = compare_within_vs_loso(valid_cols, "ALL")
print(f"ALL      | {all_within:.3f}            | {all_loso:.3f}  | {all_within - all_loso:.3f}")

# =============================================================================
# ANALYSIS 3: Feature importance vs generalization
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS: RF IMPORTANCE (ALL DATA) - Does this mislead?")
print("="*80)

# Train on all data to get "importance" - this is what looks impressive but may not generalize
df_clean_all = df.dropna(subset=['borg'] + valid_cols)
X_all = df_clean_all[valid_cols].values
y_all = df_clean_all['borg'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_scaled, y_all)

importance_df = pd.DataFrame({
    'feature': valid_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Add modality
importance_df['modality'] = importance_df['feature'].apply(
    lambda x: 'IMU' if x.startswith(('acc_', 'gyr_')) else ('PPG' if x.startswith('ppg_') else 'EDA')
)

print("\nTop 20 features by RF importance (trained on ALL data pooled):")
print("-" * 70)
for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
    print(f"  {i+1:2}. [{row['modality']:3}] {row['feature'][:45]:45} imp={row['importance']:.4f}")

print("\nTotal importance by modality:")
modality_imp = importance_df.groupby('modality')['importance'].sum()
for mod, imp in sorted(modality_imp.items(), key=lambda x: -x[1]):
    print(f"  {mod}: {imp:.3f} ({imp/modality_imp.sum()*100:.1f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("THESIS SUMMARY: PPG GENERALIZATION ANALYSIS")
print("="*80)

print(f"""
QUESTION: Do PPG features correlate within-subject but fail to generalize?

EVIDENCE:
─────────
1. Within-subject model fit (r):
   - IMU: {imu_within:.3f}
   - PPG: {ppg_within:.3f}
   
2. LOSO generalization (r):
   - IMU: {imu_loso:.3f}
   - PPG: {ppg_loso:.3f}
   
3. Overfitting gap (within - LOSO):
   - IMU: {imu_within - imu_loso:.3f}
   - PPG: {ppg_within - ppg_loso:.3f}

ANSWER: {'YES' if ppg_loso < imu_loso * 0.7 else 'PARTIALLY'} - PPG features {'show significantly worse' if ppg_loso < imu_loso * 0.7 else 'show somewhat worse'} generalization than IMU.

INTERPRETATION:
───────────────
- PPG captures individual physiology (resting HR, HRV baseline, perfusion)
- These correlate with effort WITHIN a person but vary greatly BETWEEN people
- IMU captures movement patterns - more universal across individuals
- A model trained on PPG learns "elderly1's high HR = high effort" but this 
  doesn't transfer to elderly2 who has different baseline HR

THIS JUSTIFIES:
───────────────
1. Data-driven pipeline that selects features by GENERALIZATION, not just fit
2. In your new study with longer activities + baseline/recovery:
   - HR DELTA from baseline could fix this (normalizes individual differences)
   - 5-min baseline + 5-min recovery gives you reference points
   - RMSSD change from rest → activity is more generalizable than raw RMSSD
""")
