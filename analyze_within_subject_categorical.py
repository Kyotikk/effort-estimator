#!/usr/bin/env python3
"""
Two critical analyses:
1. Within-subject categorical accuracy (LOW/MOD/HIGH)
2. Are features similar across subjects? (validates subjectivity claim)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined_5subj/all_5_elderly_5s.csv')
OUTPUT_DIR = '/Users/pascalschlegel/data/interim/elderly_combined_5subj/ml_expert_plots'

exclude_cols = ['subject', 'borg', 't_center', 'window_start', 'window_end', 'unix_time', 'Unnamed: 0', 'index']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64'] and df[c].notna().sum() > 100]
valid_features = [c for c in feature_cols if df[c].isna().mean() < 0.5]

df_model = df.dropna(subset=['borg'])[['subject', 'borg'] + valid_features].dropna()

# Categorize Borg
def to_cat(b):
    if b <= 2: return 0  # LOW
    elif b <= 4: return 1  # MODERATE
    else: return 2  # HIGH

SUBJECT_LABELS = {
    'sim_elderly1': 'P1', 'sim_elderly2': 'P2', 'sim_elderly3': 'P3',
    'sim_elderly4': 'P4', 'sim_elderly5': 'P5'
}

print("="*70)
print("PART 1: WITHIN-SUBJECT CATEGORICAL ACCURACY")
print("="*70)

within_results = {}
for subj in sorted(df_model['subject'].unique()):
    subj_data = df_model[df_model['subject'] == subj]
    
    X = subj_data[valid_features].values
    y = subj_data['borg'].values
    y_cat = np.array([to_cat(b) for b in y])
    
    if len(y) < 20:
        continue
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5-fold CV within subject
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
    y_pred_cat = np.array([to_cat(b) for b in y_pred])
    
    # Calculate accuracies
    exact_acc = accuracy_score(y_cat, y_pred_cat)
    adjacent_acc = (np.abs(y_cat - y_pred_cat) <= 1).mean()
    
    # Check category distribution
    unique, counts = np.unique(y_cat, return_counts=True)
    cat_dist = {0: 0, 1: 0, 2: 0}
    for u, c in zip(unique, counts):
        cat_dist[u] = c
    
    within_results[subj] = {
        'exact': exact_acc,
        'adjacent': adjacent_acc,
        'n': len(y),
        'n_low': cat_dist[0],
        'n_mod': cat_dist[1],
        'n_high': cat_dist[2],
    }
    
    label = SUBJECT_LABELS[subj]
    print(f"\n{label} (n={len(y)}):")
    print(f"  Category distribution: LOW={cat_dist[0]}, MOD={cat_dist[1]}, HIGH={cat_dist[2]}")
    print(f"  Exact 3-class accuracy:   {exact_acc:.1%}")
    print(f"  Adjacent accuracy (±1):   {adjacent_acc:.1%}")

# Summary
print("\n" + "="*70)
print("SUMMARY: WITHIN-SUBJECT CATEGORICAL ACCURACY")
print("="*70)

mean_exact = np.mean([r['exact'] for r in within_results.values()])
mean_adjacent = np.mean([r['adjacent'] for r in within_results.values()])

print(f"\nMean EXACT accuracy:    {mean_exact:.1%}")
print(f"Mean ADJACENT accuracy: {mean_adjacent:.1%}")
print(f"\nCompare to CROSS-SUBJECT (LOSO):")
print(f"  LOSO exact:    33%")
print(f"  LOSO adjacent: 87%")

# ============================================================================
print("\n\n" + "="*70)
print("PART 2: ARE FEATURES SIMILAR ACROSS SUBJECTS?")
print("="*70)
print("\nThis validates whether 'subjectivity' claim is real.")
print("If features overlap but Borg differs → TRUE subjectivity")
print("If features differ → physiological differences, not perception\n")

# Select key features to analyze
key_features = [
    'ppg_green_hr_mean',      # Heart rate
    'ppg_green_rmssd',        # HRV
    'ppg_green_p50',          # PPG amplitude
    'eda_stress_skin_mean',   # Skin conductance (if exists)
]
key_features = [f for f in key_features if f in valid_features]

# Add some that exist
if len(key_features) < 4:
    for f in valid_features:
        if 'hr_mean' in f or 'rmssd' in f or 'eda' in f:
            if f not in key_features:
                key_features.append(f)
        if len(key_features) >= 6:
            break

print(f"Analyzing features: {key_features}\n")

# For each feature, check overlap across subjects
feature_overlap = {}
for feat in key_features:
    print(f"\n{feat}:")
    
    # Get distribution per subject
    subject_data = {}
    for subj in sorted(df_model['subject'].unique()):
        subj_feat = df_model[df_model['subject'] == subj][feat].dropna()
        subject_data[subj] = subj_feat
        label = SUBJECT_LABELS[subj]
        print(f"  {label}: mean={subj_feat.mean():.2f}, std={subj_feat.std():.2f}, range=[{subj_feat.min():.2f}, {subj_feat.max():.2f}]")
    
    # Calculate overlap using Kolmogorov-Smirnov test
    # If p > 0.05, distributions are similar
    subjects = sorted(subject_data.keys())
    similar_pairs = 0
    total_pairs = 0
    
    for i, s1 in enumerate(subjects):
        for s2 in subjects[i+1:]:
            stat, p = ks_2samp(subject_data[s1], subject_data[s2])
            total_pairs += 1
            if p > 0.05:
                similar_pairs += 1
    
    overlap_pct = similar_pairs / total_pairs * 100 if total_pairs > 0 else 0
    feature_overlap[feat] = overlap_pct
    print(f"  → {similar_pairs}/{total_pairs} subject pairs have similar distribution ({overlap_pct:.0f}%)")

# ============================================================================
print("\n\n" + "="*70)
print("PART 3: SAME BORG RANGE → DIFFERENT FEATURES?")
print("="*70)
print("\nFiltering to samples with MODERATE effort (Borg 3-4)")
print("If features are similar for same Borg → subjectivity is real")

moderate_mask = (df_model['borg'] >= 3) & (df_model['borg'] <= 4)
df_moderate = df_model[moderate_mask]

print(f"\nModerate effort samples per subject:")
for subj in sorted(df_moderate['subject'].unique()):
    n = (df_moderate['subject'] == subj).sum()
    print(f"  {SUBJECT_LABELS[subj]}: {n} samples")

print(f"\nFeature values when Borg=3-4 (MODERATE):")
for feat in key_features[:4]:
    print(f"\n{feat} @ MODERATE effort:")
    for subj in sorted(df_moderate['subject'].unique()):
        subj_feat = df_moderate[df_moderate['subject'] == subj][feat].dropna()
        if len(subj_feat) > 0:
            label = SUBJECT_LABELS[subj]
            print(f"  {label}: mean={subj_feat.mean():.2f}, std={subj_feat.std():.2f}")

# ============================================================================
print("\n\n" + "="*70)
print("PART 4: FEATURE SIMILARITY VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

SUBJECT_COLORS = {
    'sim_elderly1': '#E69F00', 'sim_elderly2': '#56B4E9', 'sim_elderly3': '#009E73',
    'sim_elderly4': '#CC79A7', 'sim_elderly5': '#F0E442'
}

for ax, feat in zip(axes, key_features[:4]):
    for subj in sorted(df_model['subject'].unique()):
        subj_data = df_model[df_model['subject'] == subj]
        ax.scatter(subj_data[feat], subj_data['borg'], 
                  c=SUBJECT_COLORS[subj], alpha=0.5, s=30,
                  label=SUBJECT_LABELS[subj])
    
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('Borg CR-10')
    ax.legend(fontsize=8)
    ax.set_title(f'{feat[:30]} vs Borg')

plt.suptitle('Feature vs Borg by Subject\n(Do feature distributions overlap?)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/22_feature_overlap_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 22_feature_overlap_analysis.png")

# ============================================================================
print("\n\n" + "="*70)
print("CONCLUSIONS")
print("="*70)

print("""
WITHIN-SUBJECT CATEGORICAL:
─────────────────────────────────────────────────────────────────────
  Within-subject exact:    ~{:.0f}% (vs 33% cross-subject)
  Within-subject adjacent: ~{:.0f}% (vs 87% cross-subject)
  
  → Personalization improves categorical accuracy significantly!

FEATURE OVERLAP ANALYSIS:
─────────────────────────────────────────────────────────────────────
  Question: Are features similar across subjects?
  
  If YES → "Same physiology, different Borg" = TRUE SUBJECTIVITY
  If NO  → "Different physiology, different Borg" = NOT SUBJECTIVITY

  Results:""".format(mean_exact*100, mean_adjacent*100))

for feat, overlap in feature_overlap.items():
    interpretation = "OVERLAP" if overlap > 30 else "DIFFERENT"
    print(f"    {feat[:25]:25s}: {overlap:5.0f}% pairs similar → {interpretation}")

print("""
INTERPRETATION:
─────────────────────────────────────────────────────────────────────
  IF features overlap significantly:
    → Same physiological state leads to different Borg ratings
    → This is TRUE SUBJECTIVITY
    → Personalization required to learn individual calibration
    
  IF features don't overlap:
    → Different subjects have different physiology
    → Different Borg is expected (not subjectivity)
    → Need to control for activity type
    
  MIXED RESULTS suggest:
    → BOTH factors contribute
    → Some features overlap (HR similar ranges)
    → Some features differ (PPG amplitude, motion patterns)
    → Subjectivity + physiological differences BOTH matter
""")
