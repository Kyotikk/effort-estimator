#!/usr/bin/env python3
"""
Thesis Plot: Data-Driven Feature Selection
===========================================
Start with ALL features (IMU, PPG, EDA), select based on:
1. Correlation to effort (Borg)
2. RF feature importance
3. Sequential forward selection with LOSO

Let the data decide which modalities matter!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

def load_all_subjects():
    """Load fused aligned data for all 5 elderly subjects."""
    base_paths = [
        "/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/fused_aligned_5.0s.csv",
        "/Users/pascalschlegel/data/interim/parsingsim2/sim_elderly2/effort_estimation_output/elderly_sim_elderly2/fused_aligned_5.0s.csv",
        "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv",
        "/Users/pascalschlegel/data/interim/parsingsim4/sim_elderly4/effort_estimation_output/elderly_sim_elderly4/fused_aligned_5.0s.csv",
        "/Users/pascalschlegel/data/interim/parsingsim5/sim_elderly5/effort_estimation_output/elderly_sim_elderly5/fused_aligned_5.0s.csv",
    ]
    
    dfs = []
    for i, path in enumerate(base_paths, 1):
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            df['subject'] = f'P{i}'
            dfs.append(df)
            print(f"  P{i}: {len(df)} windows, {len(df.columns)} columns")
        else:
            print(f"  P{i}: NOT FOUND - {path}")
    
    if not dfs:
        raise FileNotFoundError("No data files found!")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['borg'])
    print(f"\nCombined: {len(combined)} labeled windows")
    return combined


def get_feature_columns(df):
    """Get all valid feature columns, categorized by modality."""
    skip_cols = {'t_center', 't_start', 't_end', 'borg', 'subject', 'activity_label', 
                 'window_id', 'n_samples', 'win_sec', 'modality', 'valid', 'Unnamed: 0'}
    
    feature_cols = []
    for c in df.columns:
        if c in skip_cols or c.startswith('Unnamed'):
            continue
        if df[c].dtype not in ['float64', 'int64', 'float32', 'int32']:
            continue
        if df[c].notna().mean() < 0.5:  # At least 50% non-NaN
            continue
        if df[c].std() < 1e-10:  # Not constant
            continue
        feature_cols.append(c)
    
    # Categorize by modality
    imu_cols = [c for c in feature_cols if 'acc' in c.lower() or 'gyro' in c.lower()]
    ppg_cols = [c for c in feature_cols if 'ppg' in c.lower() or 'hr' in c.lower() or 'rr' in c.lower() or 'ibi' in c.lower() or 'rmssd' in c.lower() or 'sdnn' in c.lower()]
    eda_cols = [c for c in feature_cols if 'eda' in c.lower() or 'scr' in c.lower() or 'scl' in c.lower() or 'gsr' in c.lower()]
    
    # Anything not categorized
    categorized = set(imu_cols) | set(ppg_cols) | set(eda_cols)
    other_cols = [c for c in feature_cols if c not in categorized]
    
    return {
        'all': feature_cols,
        'imu': imu_cols,
        'ppg': ppg_cols,
        'eda': eda_cols,
        'other': other_cols
    }


# ============================================================================
# LOSO Cross-Validation
# ============================================================================

def run_loso(df, feature_cols, model=None):
    """Run LOSO CV, return per-subject metrics."""
    if model is None:
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    
    subjects = df['subject'].unique()
    results = {}
    all_true, all_pred = [], []
    
    for test_subj in subjects:
        train = df[df['subject'] != test_subj]
        test = df[df['subject'] == test_subj]
        
        if len(train) < 10 or len(test) < 5:
            continue
        
        X_train = train[feature_cols].values
        y_train = train['borg'].values
        X_test = test[feature_cols].values
        y_test = test['borg'].values
        
        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))
        
        # Clone model and fit
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        
        # Metrics
        r, _ = pearsonr(y_test, y_pred) if len(y_test) > 2 else (0.0, 1.0)
        mae = np.mean(np.abs(y_test - y_pred))
        
        results[test_subj] = {'r': r, 'mae': mae, 'n': len(y_test)}
        all_true.extend(y_test)
        all_pred.extend(y_pred)
    
    # Overall metrics
    if len(all_true) > 2:
        overall_r, _ = pearsonr(all_true, all_pred)
        overall_mae = np.mean(np.abs(np.array(all_true) - np.array(all_pred)))
    else:
        overall_r, overall_mae = 0.0, 99.0
    
    mean_r = np.mean([m['r'] for m in results.values()])
    
    return {
        'per_subject': results,
        'mean_r': mean_r,
        'overall_r': overall_r,
        'overall_mae': overall_mae,
        'all_true': all_true,
        'all_pred': all_pred
    }


# ============================================================================
# Feature Ranking Methods
# ============================================================================

def rank_by_correlation(df, feature_cols):
    """Rank features by absolute correlation with Borg."""
    correlations = {}
    for col in feature_cols:
        valid = df[['borg', col]].dropna()
        if len(valid) > 10:
            r, _ = pearsonr(valid['borg'], valid[col])
            correlations[col] = abs(r)
        else:
            correlations[col] = 0.0
    
    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)


def rank_by_rf_importance(df, feature_cols):
    """Rank features by RF importance (pooled, just for ranking)."""
    X = df[feature_cols].values
    y = df['borg'].values
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = dict(zip(feature_cols, rf.feature_importances_))
    return sorted(importance.items(), key=lambda x: x[1], reverse=True)


def rank_by_consistent_correlation(df, feature_cols):
    """Rank features by CONSISTENT correlation across subjects."""
    subjects = df['subject'].unique()
    
    consistent_scores = {}
    for col in feature_cols:
        per_subj_r = []
        for subj in subjects:
            subj_df = df[df['subject'] == subj][['borg', col]].dropna()
            if len(subj_df) > 5:
                r, _ = pearsonr(subj_df['borg'], subj_df[col])
                per_subj_r.append(r)
        
        if len(per_subj_r) >= 3:
            # Check if same sign for majority
            signs = [np.sign(r) for r in per_subj_r]
            majority_sign = np.sign(np.sum(signs))
            n_agree = sum(1 for s in signs if s == majority_sign)
            
            # Score = mean absolute r * consistency fraction
            mean_abs_r = np.mean([abs(r) for r in per_subj_r])
            consistency = n_agree / len(per_subj_r)
            consistent_scores[col] = mean_abs_r * consistency
        else:
            consistent_scores[col] = 0.0
    
    return sorted(consistent_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# Sequential Forward Selection
# ============================================================================

def sequential_forward_selection(df, candidate_features, max_features=30, min_improvement=0.005, verbose=True):
    """
    Sequential Forward Selection using LOSO r as criterion.
    
    For each iteration:
    1. Try adding each remaining feature
    2. Keep the one that improves LOSO r the most
    3. Stop when no improvement or max features reached
    """
    selected = []
    current_r = 0.0
    history = []  # Track (n_features, r, added_feature, modality)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    remaining = list(candidate_features)
    
    if verbose:
        print("\n" + "="*80)
        print("SEQUENTIAL FORWARD SELECTION (using LOSO r)")
        print("="*80)
    
    while len(selected) < max_features and remaining:
        best_feature = None
        best_r = current_r
        
        for feat in remaining:
            test_features = selected + [feat]
            result = run_loso(df, test_features, model)
            
            if result['mean_r'] > best_r:
                best_r = result['mean_r']
                best_feature = feat
        
        if best_feature is None or best_r < current_r + min_improvement:
            if verbose:
                print(f"\n  Stopping: no feature improves r by {min_improvement}")
            break
        
        selected.append(best_feature)
        remaining.remove(best_feature)
        current_r = best_r
        
        # Determine modality
        mod = get_modality(best_feature)
        history.append((len(selected), current_r, best_feature, mod))
        
        if verbose:
            print(f"  #{len(selected):2d}: +{best_feature:<50} [{mod:3s}] → r = {current_r:.3f}")
    
    return selected, history


def get_modality(feature):
    """Get modality of a feature."""
    f = feature.lower()
    if 'acc' in f or 'gyro' in f:
        return 'IMU'
    elif 'ppg' in f or 'hr' in f or 'rr' in f or 'ibi' in f or 'rmssd' in f or 'sdnn' in f or 'pnn' in f:
        return 'PPG'
    elif 'eda' in f or 'scr' in f or 'scl' in f or 'gsr' in f:
        return 'EDA'
    else:
        return 'OTH'


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("DATA-DRIVEN FEATURE SELECTION")
    print("Starting with ALL features, letting data decide which modalities matter")
    print("="*80)
    
    # Load data
    print("\n1. Loading all subjects...")
    df = load_all_subjects()
    
    # Get features by modality
    print("\n2. Categorizing features by modality...")
    features = get_feature_columns(df)
    print(f"   IMU: {len(features['imu'])} features")
    print(f"   PPG: {len(features['ppg'])} features")
    print(f"   EDA: {len(features['eda'])} features")
    print(f"   Other: {len(features['other'])} features")
    print(f"   TOTAL: {len(features['all'])} features")
    
    # ========================================================================
    # Step 1: Baseline per modality (no selection)
    # ========================================================================
    print("\n" + "="*80)
    print("3. BASELINE: All features per modality (no selection)")
    print("="*80)
    
    baselines = {}
    for mod, cols in [('IMU', features['imu']), ('PPG', features['ppg']), ('EDA', features['eda'])]:
        if cols:
            result = run_loso(df, cols)
            baselines[mod] = result['mean_r']
            print(f"   {mod}: {len(cols):3d} features → LOSO mean r = {result['mean_r']:.3f}")
        else:
            baselines[mod] = 0.0
            print(f"   {mod}: 0 features")
    
    # All features combined
    if features['all']:
        result = run_loso(df, features['all'])
        baselines['ALL'] = result['mean_r']
        print(f"   ALL: {len(features['all']):3d} features → LOSO mean r = {result['mean_r']:.3f}")
    
    # ========================================================================
    # Step 2: Rank features by different methods
    # ========================================================================
    print("\n" + "="*80)
    print("4. FEATURE RANKING (3 methods)")
    print("="*80)
    
    print("\n   A. By pooled correlation to Borg:")
    corr_ranked = rank_by_correlation(df, features['all'])
    for i, (feat, score) in enumerate(corr_ranked[:10]):
        mod = get_modality(feat)
        print(f"      {i+1:2d}. {feat:<55} [{mod:3s}] r={score:.3f}")
    
    print("\n   B. By RF feature importance:")
    imp_ranked = rank_by_rf_importance(df, features['all'])
    for i, (feat, score) in enumerate(imp_ranked[:10]):
        mod = get_modality(feat)
        print(f"      {i+1:2d}. {feat:<55} [{mod:3s}] imp={score:.4f}")
    
    print("\n   C. By CONSISTENT correlation (across subjects):")
    cons_ranked = rank_by_consistent_correlation(df, features['all'])
    for i, (feat, score) in enumerate(cons_ranked[:10]):
        mod = get_modality(feat)
        print(f"      {i+1:2d}. {feat:<55} [{mod:3s}] score={score:.3f}")
    
    # ========================================================================
    # Step 3: Sequential Forward Selection from top candidates
    # ========================================================================
    print("\n" + "="*80)
    print("5. SEQUENTIAL FORWARD SELECTION")
    print("   Candidates: Top 50 by consistent correlation")
    print("="*80)
    
    # Use top 50 features by consistent correlation as candidates
    top_candidates = [f for f, s in cons_ranked[:50]]
    selected_features, selection_history = sequential_forward_selection(
        df, top_candidates, max_features=20, min_improvement=0.003, verbose=True
    )
    
    # ========================================================================
    # Step 4: Analyze final selection
    # ========================================================================
    print("\n" + "="*80)
    print("6. FINAL FEATURE SET ANALYSIS")
    print("="*80)
    
    # Modality breakdown
    mod_counts = {'IMU': 0, 'PPG': 0, 'EDA': 0, 'OTH': 0}
    for f in selected_features:
        mod_counts[get_modality(f)] += 1
    
    print(f"\n   Selected {len(selected_features)} features:")
    for mod, count in mod_counts.items():
        if count > 0:
            print(f"      {mod}: {count} features ({100*count/len(selected_features):.0f}%)")
    
    # Final performance
    print(f"\n   Final LOSO performance:")
    final_result = run_loso(df, selected_features)
    print(f"      Mean r = {final_result['mean_r']:.3f}")
    print(f"      MAE = {final_result['overall_mae']:.2f}")
    
    print("\n   Per-subject breakdown:")
    for subj, metrics in final_result['per_subject'].items():
        print(f"      {subj}: r = {metrics['r']:.3f}, MAE = {metrics['mae']:.2f}, n = {metrics['n']}")
    
    # ========================================================================
    # Step 5: Create visualization
    # ========================================================================
    print("\n" + "="*80)
    print("7. CREATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    colors = {'IMU': '#2ecc71', 'PPG': '#e74c3c', 'EDA': '#3498db', 'OTH': '#95a5a6', 'ALL': '#9b59b6'}
    
    # --- Plot A: Baseline by modality ---
    ax = axes[0, 0]
    mods = ['IMU', 'PPG', 'EDA', 'ALL']
    vals = [baselines.get(m, 0) for m in mods]
    bars = ax.bar(mods, vals, color=[colors[m] for m in mods], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('LOSO Mean r', fontsize=12)
    ax.set_title('A. Baseline: All Features per Modality', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5 threshold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
                ha='center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    
    # --- Plot B: Feature selection curve ---
    ax = axes[0, 1]
    if selection_history:
        n_feats = [h[0] for h in selection_history]
        rs = [h[1] for h in selection_history]
        mod_colors = [colors[h[3]] for h in selection_history]
        
        ax.plot(n_feats, rs, 'k-', linewidth=2, zorder=1)
        ax.scatter(n_feats, rs, c=mod_colors, s=100, edgecolor='black', linewidth=1.5, zorder=2)
        
        ax.set_xlabel('Number of Features', fontsize=12)
        ax.set_ylabel('LOSO Mean r', fontsize=12)
        ax.set_title('B. Sequential Forward Selection Curve', fontsize=13, fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Legend for modalities
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[m], edgecolor='black', label=m) 
                          for m in ['IMU', 'PPG', 'EDA'] if mod_counts.get(m, 0) > 0]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # --- Plot C: Top 10 selected features importance ---
    ax = axes[1, 0]
    if selected_features:
        # Get RF importance for selected features
        X = df[selected_features].values
        y = df['borg'].values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance = dict(zip(selected_features, rf.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        feats = [f[0][:40] + '...' if len(f[0]) > 40 else f[0] for f in sorted_imp]
        imps = [f[1] for f in sorted_imp]
        feat_colors = [colors[get_modality(f[0])] for f in sorted_imp]
        
        y_pos = np.arange(len(feats))
        ax.barh(y_pos, imps, color=feat_colors, edgecolor='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('RF Feature Importance', fontsize=12)
        ax.set_title('C. Top 10 Selected Features', fontsize=13, fontweight='bold')
    
    # --- Plot D: Modality breakdown pie + final prediction ---
    ax = axes[1, 1]
    # Pie chart of modality mix
    active_mods = {k: v for k, v in mod_counts.items() if v > 0}
    if active_mods:
        wedges, texts, autotexts = ax.pie(
            active_mods.values(), 
            labels=active_mods.keys(),
            colors=[colors[m] for m in active_mods.keys()],
            autopct='%1.0f%%',
            explode=[0.05] * len(active_mods),
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax.set_title(f'D. Final Feature Mix ({len(selected_features)} features)\nLOSO r = {final_result["mean_r"]:.2f}', 
                    fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('/Users/pascalschlegel/effort-estimator/thesis_plots_final')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / '46_datadriven_feature_selection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n   Saved: {output_path}")
    
    plt.show()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
Started with:       {len(features['all'])} features (IMU={len(features['imu'])}, PPG={len(features['ppg'])}, EDA={len(features['eda'])})
After selection:    {len(selected_features)} features
Final modality mix: {mod_counts}
Final LOSO r:       {final_result['mean_r']:.3f}

Top 5 selected features:""")
    
    for i, (n, r, feat, mod) in enumerate(selection_history[:5]):
        print(f"   {i+1}. {feat} [{mod}]")
    
    print(f"""
Conclusion: {'IMU dominates' if mod_counts['IMU'] > mod_counts['PPG'] + mod_counts['EDA'] else 
             'Mixed modalities contribute' if mod_counts['PPG'] > 0 or mod_counts['EDA'] > 0 else 
             'Only IMU generalizes'}
""")

    return selected_features, selection_history, final_result


if __name__ == "__main__":
    selected, history, result = main()
