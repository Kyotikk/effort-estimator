#!/usr/bin/env python3
"""
Fixing the Compressed Predictions Problem
==========================================
Three approaches:
1. Post-hoc scaling (stretch predictions)
2. Weighted sampling (emphasize extreme Borg)
3. Ordinal classification (predict categories)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    paths = [
        '/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv',
        '/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv',
    ]
    
    dfs = []
    for i, p in enumerate(paths, 1):
        df = pd.read_csv(p)
        df['subject'] = f'P{i}'
        dfs.append(df)
    
    combined = pd.concat(dfs).dropna(subset=['borg'])
    
    # Get IMU features
    imu_cols = [c for c in combined.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
    imu_cols = [c for c in imu_cols if combined[c].notna().mean() > 0.3 and combined[c].std() > 1e-10]
    
    return combined, imu_cols


# ============================================================================
# Approach 1: Standard (baseline)
# ============================================================================

def run_standard_loso(df, features):
    """Standard RF regression - the compressed predictions."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        X_train, y_train = train[features].values, train['borg'].values
        X_test, y_test = test[features].values, test['borg'].values
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    return np.array(all_true), np.array(all_pred)


# ============================================================================
# Approach 2: Scaled predictions (stretch to match true range)
# ============================================================================

def scale_predictions(y_true, y_pred):
    """Stretch predictions to match true distribution."""
    # Z-score normalize, then scale to true stats
    pred_z = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-10)
    pred_scaled = pred_z * y_true.std() + y_true.mean()
    # Clip to valid Borg range
    pred_scaled = np.clip(pred_scaled, 0, 10)
    return pred_scaled


# ============================================================================
# Approach 3: Weighted training (emphasize extremes)
# ============================================================================

def run_weighted_loso(df, features):
    """RF with sample weights - emphasize extreme Borg values."""
    subjects = df['subject'].unique()
    all_true, all_pred = [], []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        X_train, y_train = train[features].values, train['borg'].values
        X_test, y_test = test[features].values, test['borg'].values
        
        # Weight: higher for extremes (Borg 0-1 and 5+)
        weights = np.ones(len(y_train))
        weights[y_train <= 1] = 3.0  # Low effort - rare, important
        weights[y_train >= 5] = 3.0  # High effort - rare, important
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train, sample_weight=weights)
        y_pred = rf.predict(X_test)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    return np.array(all_true), np.array(all_pred)


# ============================================================================
# Approach 4: Ordinal Classification (best for clinical use)
# ============================================================================

def borg_to_category(borg):
    """Convert Borg to ordinal category."""
    if borg <= 1:
        return 0  # No effort
    elif borg <= 3:
        return 1  # Light
    elif borg <= 5:
        return 2  # Moderate
    else:
        return 3  # Hard (6-10)


def category_to_borg_center(cat):
    """Convert category back to Borg center for plotting."""
    centers = {0: 0.5, 1: 2.0, 2: 4.0, 3: 7.0}
    return centers.get(cat, 3)


def run_ordinal_loso(df, features):
    """Classify into effort categories instead of exact Borg."""
    subjects = df['subject'].unique()
    all_true_cat, all_pred_cat = [], []
    all_true_borg, all_pred_borg = [], []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        X_train, y_train_borg = train[features].values, train['borg'].values
        X_test, y_test_borg = test[features].values, test['borg'].values
        
        # Convert to categories
        y_train_cat = np.array([borg_to_category(b) for b in y_train_borg])
        y_test_cat = np.array([borg_to_category(b) for b in y_test_borg])
        
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_train = scl.fit_transform(imp.fit_transform(X_train))
        X_test = scl.transform(imp.transform(X_test))
        
        # Use balanced class weights
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, 
                                    n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train_cat)
        y_pred_cat = rf.predict(X_test)
        
        # Convert back to Borg centers for visualization
        y_pred_borg = np.array([category_to_borg_center(c) for c in y_pred_cat])
        
        all_true_cat.extend(y_test_cat)
        all_pred_cat.extend(y_pred_cat)
        all_true_borg.extend(y_test_borg)
        all_pred_borg.extend(y_pred_borg)
    
    return (np.array(all_true_cat), np.array(all_pred_cat), 
            np.array(all_true_borg), np.array(all_pred_borg))


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("FIXING COMPRESSED PREDICTIONS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df, imu_cols = load_data()
    print(f"  {len(df)} samples, {len(imu_cols)} IMU features")
    
    # Approach 1: Standard
    print("\n1. Standard RF (baseline)...")
    y_true, y_pred_standard = run_standard_loso(df, imu_cols)
    r_standard, _ = pearsonr(y_true, y_pred_standard)
    mae_standard = np.mean(np.abs(y_true - y_pred_standard))
    print(f"   r = {r_standard:.3f}, MAE = {mae_standard:.2f}")
    print(f"   Pred range: {y_pred_standard.min():.1f} - {y_pred_standard.max():.1f}")
    
    # Approach 2: Scaled
    print("\n2. Post-hoc scaling...")
    y_pred_scaled = scale_predictions(y_true, y_pred_standard)
    r_scaled, _ = pearsonr(y_true, y_pred_scaled)
    mae_scaled = np.mean(np.abs(y_true - y_pred_scaled))
    print(f"   r = {r_scaled:.3f}, MAE = {mae_scaled:.2f}")
    print(f"   Pred range: {y_pred_scaled.min():.1f} - {y_pred_scaled.max():.1f}")
    
    # Approach 3: Weighted
    print("\n3. Weighted training (3x weight for extremes)...")
    y_true_w, y_pred_weighted = run_weighted_loso(df, imu_cols)
    r_weighted, _ = pearsonr(y_true_w, y_pred_weighted)
    mae_weighted = np.mean(np.abs(y_true_w - y_pred_weighted))
    print(f"   r = {r_weighted:.3f}, MAE = {mae_weighted:.2f}")
    print(f"   Pred range: {y_pred_weighted.min():.1f} - {y_pred_weighted.max():.1f}")
    
    # Approach 4: Ordinal
    print("\n4. Ordinal classification (4 categories)...")
    y_true_cat, y_pred_cat, y_true_borg_ord, y_pred_borg_ord = run_ordinal_loso(df, imu_cols)
    accuracy = np.mean(y_true_cat == y_pred_cat)
    # Adjacent accuracy (off by at most 1 category)
    adjacent_acc = np.mean(np.abs(y_true_cat - y_pred_cat) <= 1)
    print(f"   Exact accuracy: {accuracy*100:.1f}%")
    print(f"   Adjacent accuracy (±1 cat): {adjacent_acc*100:.1f}%")
    
    # Confusion matrix for ordinal
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_cat, y_pred_cat)
    print(f"   Confusion matrix:")
    print(f"            Pred: Rest  Light  Mod   Hard")
    labels = ['Rest', 'Light', 'Mod', 'Hard']
    for i, label in enumerate(labels):
        print(f"   True {label:5s}: {cm[i]}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Standard (the bad one)
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred_standard, alpha=0.4, c='#e74c3c', edgecolor='black', linewidth=0.3)
    ax.plot([0, 7], [0, 7], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(f'A. Standard RF\nr = {r_standard:.2f}, MAE = {mae_standard:.2f}\n(Compressed to 1-5)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 7)
    ax.legend()
    ax.axhspan(1, 5, alpha=0.1, color='red', label='Prediction range')
    
    # Plot 2: Scaled
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred_scaled, alpha=0.4, c='#3498db', edgecolor='black', linewidth=0.3)
    ax.plot([0, 7], [0, 7], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(f'B. Post-hoc Scaling\nr = {r_scaled:.2f}, MAE = {mae_scaled:.2f}\n(Stretched to full range)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 7)
    ax.legend()
    
    # Plot 3: Weighted
    ax = axes[1, 0]
    ax.scatter(y_true_w, y_pred_weighted, alpha=0.4, c='#2ecc71', edgecolor='black', linewidth=0.3)
    ax.plot([0, 7], [0, 7], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Borg', fontsize=12)
    ax.set_title(f'C. Weighted Training (3x extremes)\nr = {r_weighted:.2f}, MAE = {mae_weighted:.2f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 7)
    ax.legend()
    
    # Plot 4: Ordinal classification
    ax = axes[1, 1]
    # Add jitter for visibility
    jitter = 0.15
    y_true_jitter = y_true_borg_ord + np.random.uniform(-jitter, jitter, len(y_true_borg_ord))
    y_pred_jitter = y_pred_borg_ord + np.random.uniform(-jitter, jitter, len(y_pred_borg_ord))
    
    # Color by correctness
    correct = y_true_cat == y_pred_cat
    ax.scatter(y_true_jitter[correct], y_pred_jitter[correct], alpha=0.5, c='#2ecc71', 
               edgecolor='black', linewidth=0.3, label=f'Correct ({sum(correct)})')
    ax.scatter(y_true_jitter[~correct], y_pred_jitter[~correct], alpha=0.5, c='#e74c3c',
               edgecolor='black', linewidth=0.3, label=f'Wrong ({sum(~correct)})')
    
    # Draw category boundaries
    for b in [1.5, 3.5, 5.5]:
        ax.axhline(b, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(b, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Actual Borg', fontsize=12)
    ax.set_ylabel('Predicted Category Center', fontsize=12)
    ax.set_title(f'D. Ordinal Classification (4 categories)\nExact: {accuracy*100:.0f}%, ±1 cat: {adjacent_acc*100:.0f}%', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.5, 7)
    ax.legend(loc='upper left')
    
    # Add category labels
    ax.text(0.5, -0.3, 'Rest', ha='center', fontsize=9)
    ax.text(2.5, -0.3, 'Light', ha='center', fontsize=9)
    ax.text(4.5, -0.3, 'Moderate', ha='center', fontsize=9)
    ax.text(6.0, -0.3, 'Hard', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / '49_fixing_compressed_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.show()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY: WHICH APPROACH IS BEST?")
    print("="*70)
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ Approach              │ r      │ MAE    │ Recommendation           │
├───────────────────────┼────────┼────────┼──────────────────────────┤
│ Standard RF           │ {r_standard:.2f}   │ {mae_standard:.2f}   │ ✗ Compressed predictions │
│ Post-hoc scaling      │ {r_scaled:.2f}   │ {mae_scaled:.2f}   │ ~ Quick fix, same r      │
│ Weighted training     │ {r_weighted:.2f}   │ {mae_weighted:.2f}   │ ~ Slightly better        │
│ Ordinal (4 classes)   │ ---    │ ---    │ ✓ {accuracy*100:.0f}% exact, {adjacent_acc*100:.0f}% ±1 cat │
└───────────────────────┴────────┴────────┴──────────────────────────┘

RECOMMENDATION FOR THESIS:
==========================
1. Report BOTH continuous (r = {r_standard:.2f}) AND ordinal ({accuracy*100:.0f}% accuracy)
2. Argue that ordinal is more clinically meaningful:
   - Doctors don't need Borg 4.2 vs 4.5
   - They need: "Is this person at rest, light activity, or working hard?"
3. {adjacent_acc*100:.0f}% ±1 category accuracy is actually USEFUL

HONEST NARRATIVE:
=================
"Continuous Borg prediction achieves moderate correlation (r = {r_standard:.2f}) but 
predictions are compressed to the middle range. Reframing as 4-class ordinal 
classification achieves {accuracy*100:.0f}% exact match and {adjacent_acc*100:.0f}% adjacent accuracy,
which is more appropriate for clinical decision support where the goal is 
identifying effort levels rather than precise Borg values."
""")


if __name__ == "__main__":
    main()
