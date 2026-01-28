#!/usr/bin/env python3
"""
Generate Top Features Analysis Plots.
Shows which features are most predictive for each subject.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df.dropna(subset=["borg"]).copy()

def get_feature_columns(df):
    skip = {"window_id", "start_idx", "end_idx", "valid", "t_start", "t_center", 
            "t_end", "n_samples", "win_sec", "modality", "subject", "subject_id", "borg"}
    return [c for c in df.columns if c not in skip and not c.endswith("_r")]

def select_features(X, y, names, top_n=50, corr_thresh=0.85):
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.asarray(y).flatten()
    
    corrs = []
    for i in range(X.shape[1]):
        xi = X[:,i]
        if np.std(xi) > 1e-10 and len(xi) == len(y):
            try:
                c = np.corrcoef(xi.astype(float), y.astype(float))[0,1]
                corrs.append(abs(c) if np.isfinite(c) else 0)
            except:
                corrs.append(0)
        else:
            corrs.append(0)
    
    top_idx = np.argsort(corrs)[-top_n:][::-1]
    selected = []
    for idx in top_idx:
        redundant = False
        for s in selected:
            if np.std(X[:,idx]) > 1e-10 and np.std(X[:,s]) > 1e-10:
                try:
                    pc = np.corrcoef(X[:,idx].astype(float), X[:,s].astype(float))[0,1]
                    if np.isfinite(pc) and abs(pc) > corr_thresh:
                        redundant = True
                        break
                except:
                    pass
        if not redundant:
            selected.append(idx)
    
    return selected, [names[i] for i in selected], [corrs[i] for i in selected]

def categorize_feature(fname):
    fname_lower = fname.lower()
    if any(x in fname_lower for x in ['eda', 'scr', 'scl', 'phasic', 'stress_skin', 'cc_']):
        return 'EDA', '#3498db'  # Blue
    elif any(x in fname_lower for x in ['ppg', 'hr_', 'rmssd', 'sdnn', 'pnn', 'lf_', 'hf_']):
        return 'PPG/HRV', '#e74c3c'  # Red
    elif any(x in fname_lower for x in ['acc', 'imu', 'gyro']):
        return 'IMU', '#2ecc71'  # Green
    else:
        return 'Other', '#95a5a6'  # Gray

def analyze_subject(df, subject, feature_cols):
    df_sub = df[df["subject"] == subject]
    X = df_sub[feature_cols].values
    y = df_sub["borg"].values
    
    # Feature selection
    sel_idx, sel_names, sel_corrs = select_features(X, y, feature_cols, top_n=50)
    
    X_sel = np.nan_to_num(X[:, sel_idx], nan=0, posinf=0, neginf=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    
    # Train model
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, 
                             reg_lambda=2.0, min_child_weight=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y, verbose=False)
    
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    
    return {
        "subject": subject,
        "borg_range": (y.min(), y.max()),
        "features": [sel_names[i] for i in sorted_idx],
        "importance": [importance[i] for i in sorted_idx],
        "correlation": [sel_corrs[i] for i in sorted_idx],
    }

def plot_top_features(results, output_dir, top_n=15):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_subjects = len(results)
    
    # Plot 1: Bar chart of top features by importance
    fig, axes = plt.subplots(1, n_subjects, figsize=(10*n_subjects, 10))
    if n_subjects == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        features = res["features"][:top_n]
        importance = res["importance"][:top_n]
        
        # Colors by category
        colors = [categorize_feature(f)[1] for f in features]
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (XGBoost Gain)', fontsize=11)
        ax.set_title(f"{res['subject']}\nBorg: {res['borg_range'][0]:.1f} - {res['borg_range'][1]:.1f}", 
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='EDA (Skin Conductance)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='PPG/HRV (Heart)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='IMU (Motion)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / "top_features_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'top_features_importance.png'}")
    
    # Plot 2: Correlation with Borg
    fig, axes = plt.subplots(1, n_subjects, figsize=(10*n_subjects, 10))
    if n_subjects == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        features = res["features"][:top_n]
        correlation = res["correlation"][:top_n]
        colors = [categorize_feature(f)[1] for f in features]
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, correlation, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Correlation with Borg Score', fontsize=11)
        ax.set_title(f"{res['subject']}\nBorg: {res['borg_range'][0]:.1f} - {res['borg_range'][1]:.1f}", 
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(correlation) * 1.1)
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / "top_features_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'top_features_correlation.png'}")
    
    # Plot 3: Feature category breakdown
    fig, axes = plt.subplots(1, n_subjects, figsize=(8*n_subjects, 6))
    if n_subjects == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        # Count importance by category
        cat_importance = {'EDA': 0, 'PPG/HRV': 0, 'IMU': 0, 'Other': 0}
        for feat, imp in zip(res["features"], res["importance"]):
            cat, _ = categorize_feature(feat)
            cat_importance[cat] += imp
        
        # Normalize
        total = sum(cat_importance.values())
        if total > 0:
            cat_importance = {k: v/total*100 for k, v in cat_importance.items()}
        
        categories = list(cat_importance.keys())
        values = list(cat_importance.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('Share of Total Importance (%)', fontsize=11)
        ax.set_title(f"{res['subject']}\nFeature Category Breakdown", fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_category_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'feature_category_breakdown.png'}")
    
    # Save CSV summary
    for res in results:
        df_summary = pd.DataFrame({
            'rank': range(1, len(res["features"]) + 1),
            'feature': res["features"],
            'importance': res["importance"],
            'correlation': res["correlation"],
            'category': [categorize_feature(f)[0] for f in res["features"]],
        })
        csv_path = output_dir / f"top_features_{res['subject']}.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
    
    return output_dir

def main():
    print("="*70)
    print("TOP FEATURES ANALYSIS")
    print("="*70)
    
    filepath = "/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv"
    df = load_data(filepath)
    feature_cols = get_feature_columns(df)
    
    # Check for HRV features
    hrv_features = [c for c in feature_cols if any(x in c.lower() for x in ['rmssd', 'hr_', 'sdnn', 'pnn', 'lf_', 'hf_'])]
    
    print(f"\nDataset: {filepath}")
    print(f"Total features: {len(feature_cols)}")
    print(f"HRV features found: {len(hrv_features)}")
    
    if not hrv_features:
        print("\n⚠️  WARNING: No HRV features (RMSSD, HR, etc.) in dataset!")
        print("   The HRV extraction was not run during pipeline execution.")
        print("   Current features are: EDA, PPG raw, IMU")
    else:
        print(f"   HRV features: {hrv_features[:10]}...")
    
    # Handle column naming (some files use 'subject', others use 'subject_id')
    if 'subject_id' in df.columns and 'subject' not in df.columns:
        df['subject'] = df['subject_id']
    
    # Analyze each subject
    results = []
    for subject in sorted(df["subject"].unique()):
        print(f"\nAnalyzing {subject}...")
        res = analyze_subject(df, subject, feature_cols)
        results.append(res)
    
    # Generate plots
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_top_features")
    plot_top_features(results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for res in results:
        print(f"\n{res['subject']} (Borg {res['borg_range'][0]:.1f}-{res['borg_range'][1]:.1f}):")
        print(f"  Top 5 features:")
        for i in range(min(5, len(res["features"]))):
            cat, _ = categorize_feature(res["features"][i])
            print(f"    {i+1}. [{cat}] {res['features'][i]} (imp={res['importance'][i]:.3f}, corr={res['correlation'][i]:.3f})")
    
    print(f"\n📁 Plots saved to: {output_dir}")
    
    if not hrv_features:
        print("\n" + "="*70)
        print("TO ADD HRV FEATURES (RMSSD, HR, etc.):")
        print("="*70)
        print("The HRV extraction module exists but wasn't run.")
        print("Re-run the full pipeline to include HRV features.")
    
    return 0

if __name__ == "__main__":
    main()
