#!/usr/bin/env python3
"""
Generate all visualization plots for the effort estimation pipeline.

Plots generated:
1. PCA variance explained (scree plot + cumulative)
2. Feature importance bar chart (XGBoost top 15)
3. Predicted vs Actual scatter plot
4. Ridge coefficients bar chart (top 15)
5. Correlation heatmap of top features
6. Train/test performance comparison
7. Residual distribution
8. Per-subject performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

# Configuration
OUTPUT_DIR = Path("/Users/pascalschlegel/data/interim/elderly_combined")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = "5.0s"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'ppg': '#e74c3c',
    'eda': '#3498db', 
    'imu': '#2ecc71',
    'other': '#95a5a6'
}

def get_modality_color(feature_name):
    """Get color based on feature modality."""
    if 'ppg' in feature_name.lower():
        return COLORS['ppg']
    elif 'eda' in feature_name.lower():
        return COLORS['eda']
    elif 'acc' in feature_name.lower() or 'imu' in feature_name.lower():
        return COLORS['imu']
    return COLORS['other']


def plot_pca_variance():
    """Plot PCA explained variance (scree + cumulative)."""
    print("📊 Plotting PCA variance...")
    
    pca_df = pd.read_csv(OUTPUT_DIR / f"qc_{WINDOW}" / "pca_variance_explained.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax1 = axes[0]
    n_components = min(30, len(pca_df))
    ax1.bar(range(1, n_components + 1), pca_df['explained_variance_ratio'][:n_components] * 100, 
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, n_components + 1, 2))
    
    # Cumulative variance
    ax2 = axes[1]
    cumulative = pca_df['cumulative_explained_variance'][:n_components] * 100
    ax2.plot(range(1, n_components + 1), cumulative, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax2.axhline(y=90, color='red', linestyle='--', label='90% threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    
    # Find PCs for thresholds
    pc_90 = (cumulative >= 90).idxmax() + 1 if (cumulative >= 90).any() else n_components
    pc_95 = (cumulative >= 95).idxmax() + 1 if (cumulative >= 95).any() else n_components
    
    ax2.axvline(x=pc_90, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(x=pc_95, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Number of Principal Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 105)
    
    # Add annotation
    ax2.annotate(f'{pc_90} PCs for 90%', xy=(pc_90, 90), xytext=(pc_90 + 3, 85),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_variance_explained.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: pca_variance_explained.png")


def plot_feature_importance():
    """Plot XGBoost feature importance (top 15)."""
    print("📊 Plotting feature importance...")
    
    importance_df = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{WINDOW}" / "feature_importance.csv")
    top_15 = importance_df.nlargest(15, 'importance')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = [get_modality_color(f) for f in top_15['feature']]
    
    bars = ax.barh(range(len(top_15)), top_15['importance'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_15['importance'])):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ppg'], label='PPG'),
        Patch(facecolor=COLORS['eda'], label='EDA'),
        Patch(facecolor=COLORS['imu'], label='IMU'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_xgboost.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: feature_importance_xgboost.png")


def plot_ridge_coefficients():
    """Plot Ridge regression coefficients (top 15 by absolute value)."""
    print("📊 Plotting Ridge coefficients...")
    
    coef_df = pd.read_csv(OUTPUT_DIR / f"ridge_results_{WINDOW}" / "coefficients.csv")
    top_15 = coef_df.nlargest(15, 'abs_coefficient')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#e74c3c' if c > 0 else '#3498db' for c in top_15['coefficient']]
    
    bars = ax.barh(range(len(top_15)), top_15['coefficient'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient (Standardized)', fontsize=12)
    ax.set_title('Ridge Regression Coefficients (Top 15 by |coefficient|)', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Positive (↑ feature → ↑ effort)'),
        Patch(facecolor='#3498db', label='Negative (↑ feature → ↓ effort)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "coefficients_ridge.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: coefficients_ridge.png")


def plot_predictions_scatter():
    """Plot predicted vs actual Borg ratings."""
    print("📊 Plotting predictions scatter...")
    
    # Load XGBoost predictions
    xgb_pred = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{WINDOW}" / "predictions.csv")
    ridge_pred = pd.read_csv(OUTPUT_DIR / f"ridge_results_{WINDOW}" / "predictions.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # XGBoost
    ax1 = axes[0]
    r_xgb, _ = pearsonr(xgb_pred['y_true'], xgb_pred['y_pred'])
    ax1.scatter(xgb_pred['y_true'], xgb_pred['y_pred'], alpha=0.5, s=30, c='steelblue')
    ax1.plot([0, 20], [0, 20], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Borg Rating', fontsize=12)
    ax1.set_ylabel('Predicted Borg Rating', fontsize=12)
    ax1.set_title(f'XGBoost: Predicted vs Actual\n(r = {r_xgb:.3f})', fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 20)
    ax1.set_ylim(-1, 20)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Ridge
    ax2 = axes[1]
    r_ridge, _ = pearsonr(ridge_pred['y_true'], ridge_pred['y_pred'])
    ax2.scatter(ridge_pred['y_true'], ridge_pred['y_pred'], alpha=0.5, s=30, c='darkorange')
    ax2.plot([0, 20], [0, 20], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('Actual Borg Rating', fontsize=12)
    ax2.set_ylabel('Predicted Borg Rating', fontsize=12)
    ax2.set_title(f'Ridge: Predicted vs Actual\n(r = {r_ridge:.3f})', fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 20)
    ax2.set_ylim(-1, 20)
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predictions_scatter.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: predictions_scatter.png")


def plot_residuals():
    """Plot residual distribution."""
    print("📊 Plotting residuals...")
    
    xgb_pred = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{WINDOW}" / "predictions.csv")
    ridge_pred = pd.read_csv(OUTPUT_DIR / f"ridge_results_{WINDOW}" / "predictions.csv")
    
    xgb_residuals = xgb_pred['y_true'] - xgb_pred['y_pred']
    ridge_residuals = ridge_pred['y_true'] - ridge_pred['y_pred']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # XGBoost residuals
    ax1 = axes[0]
    ax1.hist(xgb_residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=xgb_residuals.mean(), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean: {xgb_residuals.mean():.2f}')
    ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'XGBoost Residuals\n(MAE = {np.abs(xgb_residuals).mean():.2f})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Ridge residuals
    ax2 = axes[1]
    ax2.hist(ridge_residuals, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=ridge_residuals.mean(), color='blue', linestyle='-', linewidth=2,
                label=f'Mean: {ridge_residuals.mean():.2f}')
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Ridge Residuals\n(MAE = {np.abs(ridge_residuals).mean():.2f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals_distribution.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: residuals_distribution.png")


def plot_per_subject_performance():
    """Plot per-subject prediction performance."""
    print("📊 Plotting per-subject performance...")
    
    pred_df = pd.read_csv(OUTPUT_DIR / f"ridge_results_{WINDOW}" / "predictions.csv")
    
    subjects = pred_df['subject'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = {'subject': [], 'r': [], 'mae': [], 'n': []}
    
    for i, subject in enumerate(subjects):
        subj_data = pred_df[pred_df['subject'] == subject]
        r, _ = pearsonr(subj_data['y_true'], subj_data['y_pred'])
        mae = np.abs(subj_data['y_true'] - subj_data['y_pred']).mean()
        
        metrics['subject'].append(subject)
        metrics['r'].append(r)
        metrics['mae'].append(mae)
        metrics['n'].append(len(subj_data))
        
        ax = axes[i]
        ax.scatter(subj_data['y_true'], subj_data['y_pred'], alpha=0.6, s=40)
        ax.plot([0, 20], [0, 20], 'r--', linewidth=2)
        ax.set_xlabel('Actual Borg', fontsize=11)
        ax.set_ylabel('Predicted Borg', fontsize=11)
        ax.set_title(f'{subject}\nr = {r:.3f}, MAE = {mae:.2f}, n = {len(subj_data)}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(-1, 18)
        ax.set_ylim(-1, 18)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_subject_performance.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: per_subject_performance.png")
    
    return pd.DataFrame(metrics)


def plot_window_comparison():
    """Plot window size comparison (5s vs 10s)."""
    print("📊 Plotting window comparison...")
    
    # Load data for both window sizes
    results = []
    for window in ['5.0s', '10.0s']:
        try:
            xgb_pred = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{window}" / "predictions.csv")
            ridge_pred = pd.read_csv(OUTPUT_DIR / f"ridge_results_{window}" / "predictions.csv")
            
            r_xgb, _ = pearsonr(xgb_pred['y_true'], xgb_pred['y_pred'])
            r_ridge, _ = pearsonr(ridge_pred['y_true'], ridge_pred['y_pred'])
            mae_xgb = np.abs(xgb_pred['y_true'] - xgb_pred['y_pred']).mean()
            mae_ridge = np.abs(ridge_pred['y_true'] - ridge_pred['y_pred']).mean()
            
            results.append({
                'window': window,
                'n_samples': len(xgb_pred),
                'xgboost_r': r_xgb,
                'ridge_r': r_ridge,
                'xgboost_mae': mae_xgb,
                'ridge_mae': mae_ridge,
            })
        except FileNotFoundError:
            print(f"  ⚠ No results for {window}")
    
    if len(results) < 2:
        print("  ⚠ Need both 5s and 10s results for comparison")
        return
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample count
    ax1 = axes[0]
    bars1 = ax1.bar(results_df['window'], results_df['n_samples'], color=['#2ecc71', '#3498db'], alpha=0.8)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Sample Count by Window Size', fontsize=14, fontweight='bold')
    for bar, val in zip(bars1, results_df['n_samples']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val), 
                ha='center', fontsize=12, fontweight='bold')
    
    # Pearson r comparison
    ax2 = axes[1]
    x = np.arange(len(results_df))
    width = 0.35
    bars2a = ax2.bar(x - width/2, results_df['xgboost_r'], width, label='XGBoost', color='steelblue', alpha=0.8)
    bars2b = ax2.bar(x + width/2, results_df['ridge_r'], width, label='Ridge', color='darkorange', alpha=0.8)
    ax2.set_ylabel('Pearson r', fontsize=12)
    ax2.set_title('Correlation by Window Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['window'])
    ax2.legend()
    ax2.set_ylim(0, 0.8)
    for bar, val in zip(bars2a, results_df['xgboost_r']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', fontsize=10)
    for bar, val in zip(bars2b, results_df['ridge_r']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', fontsize=10)
    
    # MAE comparison
    ax3 = axes[2]
    bars3a = ax3.bar(x - width/2, results_df['xgboost_mae'], width, label='XGBoost', color='steelblue', alpha=0.8)
    bars3b = ax3.bar(x + width/2, results_df['ridge_mae'], width, label='Ridge', color='darkorange', alpha=0.8)
    ax3.set_ylabel('MAE (Borg points)', fontsize=12)
    ax3.set_title('MAE by Window Size', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['window'])
    ax3.legend()
    for bar, val in zip(bars3a, results_df['xgboost_mae']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', fontsize=10)
    for bar, val in zip(bars3b, results_df['ridge_mae']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "window_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: window_comparison.png")


def plot_correlation_heatmap():
    """Plot correlation heatmap of top features."""
    print("📊 Plotting correlation heatmap...")
    
    # Load data and selected features
    data_df = pd.read_csv(OUTPUT_DIR / f"elderly_aligned_{WINDOW}.csv")
    features_df = pd.read_csv(OUTPUT_DIR / f"qc_{WINDOW}" / "features_selected_pruned.csv")
    
    # Get top 20 features by importance
    importance_df = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{WINDOW}" / "feature_importance.csv")
    top_20 = importance_df.nlargest(20, 'importance')['feature'].tolist()
    
    # Filter to available columns
    available_cols = [c for c in top_20 if c in data_df.columns][:15]
    
    if len(available_cols) < 5:
        print("  ⚠ Not enough features for heatmap")
        return
    
    # Compute correlation matrix
    corr_matrix = data_df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Heatmap (Top 15 by Importance)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: correlation_heatmap.png")


def plot_modality_contribution():
    """Plot modality contribution to model."""
    print("📊 Plotting modality contribution...")
    
    importance_df = pd.read_csv(OUTPUT_DIR / f"xgboost_results_{WINDOW}" / "feature_importance.csv")
    
    # Categorize by modality
    def get_modality(feature):
        if 'ppg' in feature.lower():
            return 'PPG'
        elif 'eda' in feature.lower():
            return 'EDA'
        elif 'acc' in feature.lower():
            return 'IMU'
        return 'Other'
    
    importance_df['modality'] = importance_df['feature'].apply(get_modality)
    
    modality_importance = importance_df.groupby('modality')['importance'].sum()
    modality_counts = importance_df.groupby('modality').size()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart - total importance
    ax1 = axes[0]
    colors = [COLORS.get(m.lower(), COLORS['other']) for m in modality_importance.index]
    wedges, texts, autotexts = ax1.pie(modality_importance, labels=modality_importance.index, 
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Total Feature Importance by Modality', fontsize=14, fontweight='bold')
    
    # Bar chart - feature count
    ax2 = axes[1]
    bars = ax2.bar(modality_counts.index, modality_counts.values, 
                   color=[COLORS.get(m.lower(), COLORS['other']) for m in modality_counts.index], alpha=0.8)
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Count by Modality', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, modality_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "modality_contribution.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: modality_contribution.png")


def plot_borg_distribution():
    """Plot Borg rating distribution."""
    print("📊 Plotting Borg distribution...")
    
    data_df = pd.read_csv(OUTPUT_DIR / f"elderly_aligned_{WINDOW}.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    ax1 = axes[0]
    borg_values = data_df['borg'].dropna()
    ax1.hist(borg_values, bins=range(0, 21), color='steelblue', alpha=0.7, edgecolor='black', align='left')
    ax1.axvline(x=borg_values.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {borg_values.mean():.1f}')
    ax1.set_xlabel('Borg Rating', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Borg Rating Distribution (n={len(borg_values)})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(range(0, 21, 2))
    
    # Per-subject distribution
    ax2 = axes[1]
    subjects = data_df['subject'].unique()
    positions = []
    for i, subject in enumerate(subjects):
        subj_borg = data_df[data_df['subject'] == subject]['borg'].dropna()
        bp = ax2.boxplot([subj_borg], positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(plt.cm.Set2(i))
        positions.append(i)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(subjects, rotation=15)
    ax2.set_ylabel('Borg Rating', fontsize=12)
    ax2.set_title('Borg Distribution by Subject', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "borg_distribution.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: borg_distribution.png")


def main():
    print("="*70)
    print("GENERATING PIPELINE PLOTS")
    print(f"Output directory: {PLOTS_DIR}")
    print("="*70)
    
    plot_pca_variance()
    plot_feature_importance()
    plot_ridge_coefficients()
    plot_predictions_scatter()
    plot_residuals()
    plot_per_subject_performance()
    plot_window_comparison()
    plot_correlation_heatmap()
    plot_modality_contribution()
    plot_borg_distribution()
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED")
    print(f"Location: {PLOTS_DIR}")
    print("="*70)
    
    # List all generated plots
    plots = list(PLOTS_DIR.glob("*.png"))
    print(f"\nGenerated {len(plots)} plots:")
    for p in sorted(plots):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
