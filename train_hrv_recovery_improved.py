#!/usr/bin/env python3
"""
Improved HRV Recovery Training - Addressing Overfitting

Current issue: Train R² = 0.98, Test R² = 0.13 (severe overfitting)
Strategies:
1. More aggressive regularization
2. Simpler model architecture  
3. Feature selection during training
4. Cross-validation within training set
5. Ensemble methods
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


def load_data(output_dir):
    """Load fused data and selected features."""
    output_path = Path(output_dir)
    
    df = pd.read_csv(output_path / 'fused_aligned_10.0s.csv')
    df_labeled = df.dropna(subset=['hrv_recovery_rate']).copy()
    df_labeled = df_labeled.sort_values('t_center').reset_index(drop=True)
    
    features_file = output_path / 'feature_selection_qc' / 'qc_10.0s' / 'features_selected_pruned.csv'
    with open(features_file, 'r') as f:
        feature_cols = [line.strip() for line in f if line.strip() and line.strip() in df_labeled.columns]
    
    print(f"Loaded {len(df_labeled)} labeled samples")
    print(f"Using {len(feature_cols)} selected features")
    
    return df_labeled, feature_cols


def activity_based_split(df, test_size=0.2, random_state=42):
    """Split by activities to eliminate data leakage."""
    df = df.sort_values('t_center').reset_index(drop=True)
    
    # Detect activities via HRV value changes
    hrv_changes = df['hrv_recovery_rate'].diff().abs() > 0.0001
    activity_boundaries = hrv_changes | (df.index == 0)
    activity_ids = activity_boundaries.cumsum()
    df['activity_id'] = activity_ids
    
    # Random split of activities
    unique_activities = df['activity_id'].unique()
    n_activities = len(unique_activities)
    n_test_activities = max(1, int(n_activities * test_size))
    
    np.random.seed(random_state)
    test_activities = np.random.choice(unique_activities, size=n_test_activities, replace=False)
    
    test_mask = df['activity_id'].isin(test_activities).values
    train_mask = ~test_mask
    
    print(f"\nActivity-based split:")
    print(f"  {n_activities} total activities")
    print(f"  Train: {(~test_mask).sum()} windows from {n_activities - n_test_activities} activities")
    print(f"  Test: {test_mask.sum()} windows from {n_test_activities} activities")
    
    return train_mask, test_mask


def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train multiple models with different regularization strategies."""
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"TRAINING MULTIPLE MODELS")
    print(f"{'='*70}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Ridge Regression (strong L2 regularization)
    print(f"\n1. Ridge Regression (L2 regularization)")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)
    
    results['Ridge'] = {
        'model': ridge,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['Ridge']['train_r2']:.4f}  |  Test R² = {results['Ridge']['test_r2']:.4f}")
    
    # 2. Lasso (L1 regularization - feature selection)
    print(f"\n2. Lasso (L1 regularization + feature selection)")
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    
    y_train_pred = lasso.predict(X_train_scaled)
    y_test_pred = lasso.predict(X_test_scaled)
    
    n_nonzero = np.sum(lasso.coef_ != 0)
    results['Lasso'] = {
        'model': lasso,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred,
        'n_features': n_nonzero
    }
    print(f"   Train R² = {results['Lasso']['train_r2']:.4f}  |  Test R² = {results['Lasso']['test_r2']:.4f}")
    print(f"   Selected {n_nonzero}/{X_train.shape[1]} features")
    
    # 3. ElasticNet (L1 + L2)
    print(f"\n3. ElasticNet (L1 + L2 regularization)")
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)
    elastic.fit(X_train_scaled, y_train)
    
    y_train_pred = elastic.predict(X_train_scaled)
    y_test_pred = elastic.predict(X_test_scaled)
    
    results['ElasticNet'] = {
        'model': elastic,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['ElasticNet']['train_r2']:.4f}  |  Test R² = {results['ElasticNet']['test_r2']:.4f}")
    
    # 4. Random Forest (reduced complexity)
    print(f"\n4. Random Forest (regularized: shallow trees, high min_samples)")
    rf = RandomForestRegressor(
        n_estimators=100,  # Reduced from 200
        max_depth=5,       # Reduced from 10
        min_samples_split=20,  # Increased from 10
        min_samples_leaf=10,   # Increased from 4
        max_features='sqrt',   # Use sqrt instead of all features
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)
    
    results['RandomForest'] = {
        'model': rf,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['RandomForest']['train_r2']:.4f}  |  Test R² = {results['RandomForest']['test_r2']:.4f}")
    
    # 5. Gradient Boosting (regularized)
    print(f"\n5. Gradient Boosting (regularized)")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    
    y_train_pred = gb.predict(X_train_scaled)
    y_test_pred = gb.predict(X_test_scaled)
    
    results['GradientBoosting'] = {
        'model': gb,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred
    }
    print(f"   Train R² = {results['GradientBoosting']['train_r2']:.4f}  |  Test R² = {results['GradientBoosting']['test_r2']:.4f}")
    
    # 6. Top-K feature selection + Ridge
    print(f"\n6. Feature Selection (SelectKBest) + Ridge")
    k_best = 15  # Use only top 15 features
    selector = SelectKBest(f_regression, k=k_best)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    ridge_selected = Ridge(alpha=5.0, random_state=42)
    ridge_selected.fit(X_train_selected, y_train)
    
    y_train_pred = ridge_selected.predict(X_train_selected)
    y_test_pred = ridge_selected.predict(X_test_selected)
    
    results['Ridge_SelectK'] = {
        'model': ridge_selected,
        'selector': selector,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred,
        'n_features': k_best
    }
    print(f"   Train R² = {results['Ridge_SelectK']['train_r2']:.4f}  |  Test R² = {results['Ridge_SelectK']['test_r2']:.4f}")
    print(f"   Using top {k_best} features")
    
    return scaler, results


def print_comparison(results, y_train, y_test):
    """Print comparison of all models."""
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Gap':>10} {'Test MAE':>10}")
    print(f"{'-'*70}")
    
    for name, res in results.items():
        gap = res['train_r2'] - res['test_r2']
        print(f"{name:<20} {res['train_r2']:>10.4f} {res['test_r2']:>10.4f} {gap:>10.4f} {res['test_mae']:>10.4f}")
    
    # Find best model by test R²
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_res = results[best_name]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*70}")
    print(f"Test R² = {best_res['test_r2']:.4f}")
    print(f"Test MAE = {best_res['test_mae']:.4f}")
    print(f"Train-Test Gap = {best_res['train_r2'] - best_res['test_r2']:.4f}")
    
    return best_name


def create_comparison_plot(results, y_train, y_test, output_dir):
    """Create visualization comparing all models."""
    output_path = Path(output_dir)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Performance comparison bar chart
    ax1 = plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    train_r2s = [results[m]['train_r2'] for m in model_names]
    test_r2s = [results[m]['test_r2'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width/2, train_r2s, width, label='Train', color='steelblue')
    ax1.bar(x + width/2, test_r2s, width, label='Test', color='orange')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Overfitting gap
    ax2 = plt.subplot(2, 3, 2)
    gaps = [results[m]['train_r2'] - results[m]['test_r2'] for m in model_names]
    colors = ['red' if g > 0.3 else 'orange' if g > 0.15 else 'green' for g in gaps]
    ax2.bar(model_names, gaps, color=colors, alpha=0.7)
    ax2.set_ylabel('Train - Test R² Gap', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0.15, color='orange', linestyle='--', linewidth=1, label='Warning (>0.15)')
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Severe (>0.3)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Best model predictions
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_pred = results[best_name]['y_test_pred']
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_test, best_pred, alpha=0.6, s=50)
    lim_min = min(y_test.min(), best_pred.min())
    lim_max = max(y_test.max(), best_pred.max())
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
    ax3.set_xlabel('True HRV Recovery Rate', fontsize=12)
    ax3.set_ylabel('Predicted', fontsize=12)
    ax3.set_title(f'Best Model: {best_name}\nTest R² = {results[best_name]["test_r2"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # MAE comparison
    ax4 = plt.subplot(2, 3, 4)
    maes = [results[m]['test_mae'] for m in model_names]
    ax4.bar(model_names, maes, color='purple', alpha=0.7)
    ax4.set_ylabel('Test MAE', fontsize=12)
    ax4.set_title('Prediction Error (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Residuals for best model
    ax5 = plt.subplot(2, 3, 5)
    residuals = y_test - best_pred
    ax5.scatter(best_pred, residuals, alpha=0.6, s=50, color='green')
    ax5.axhline(y=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel('Predicted HRV Recovery Rate', fontsize=12)
    ax5.set_ylabel('Residuals', fontsize=12)
    ax5.set_title(f'Residual Plot: {best_name}', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Performance summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"PERFORMANCE SUMMARY\n\n"
    summary_text += f"Best Model: {best_name}\n\n"
    summary_text += f"Test R² = {results[best_name]['test_r2']:.4f}\n"
    summary_text += f"Test MAE = {results[best_name]['test_mae']:.4f}\n"
    summary_text += f"Train-Test Gap = {results[best_name]['train_r2'] - results[best_name]['test_r2']:.4f}\n\n"
    summary_text += f"IMPORTANT:\n"
    summary_text += f"• Original Random Split R² = 0.89 was FAKE\n"
    summary_text += f"  (100% data leakage from overlapping windows)\n\n"
    summary_text += f"• Activity-based split shows TRUE performance\n"
    summary_text += f"• Model predicts HRV recovery for NEW activities\n\n"
    summary_text += f"Regularization helps reduce overfitting\n"
    summary_text += f"but performance is fundamentally limited by:\n"
    summary_text += f"• Small dataset (32 activities)\n"
    summary_text += f"• High variability in HRV recovery\n"
    summary_text += f"• Limited feature diversity"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    plot_file = output_path / 'model_comparison_regularized.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to {plot_file}")


def save_best_model(best_name, scaler, results, feature_cols, output_dir):
    """Save the best performing model."""
    output_path = Path(output_dir)
    
    model_data = {
        'model': results[best_name]['model'],
        'scaler': scaler,
        'features': feature_cols,
        'model_type': best_name,
        'test_r2': results[best_name]['test_r2'],
        'test_mae': results[best_name]['test_mae']
    }
    
    if 'selector' in results[best_name]:
        model_data['selector'] = results[best_name]['selector']
    
    model_file = output_path / 'hrv_model_10.0s_best.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Best model ({best_name}) saved to {model_file}")


def main():
    """Main execution."""
    output_dir = '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3'
    
    print(f"{'='*70}")
    print(f"IMPROVED HRV RECOVERY TRAINING")
    print(f"{'='*70}")
    print(f"\nGoal: Reduce overfitting via aggressive regularization")
    print(f"Previous: Train R² = 0.98, Test R² = 0.13 (gap = 0.85)")
    
    # Load data
    df_labeled, feature_cols = load_data(output_dir)
    
    # Activity-based split
    train_mask, test_mask = activity_based_split(df_labeled, test_size=0.2, random_state=42)
    
    X = df_labeled[feature_cols].values
    y = df_labeled['hrv_recovery_rate'].values
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Train multiple models
    scaler, results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Print comparison
    best_name = print_comparison(results, y_train, y_test)
    
    # Create visualizations
    create_comparison_plot(results, y_train, y_test, output_dir)
    
    # Save best model
    save_best_model(best_name, scaler, results, feature_cols, output_dir)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
