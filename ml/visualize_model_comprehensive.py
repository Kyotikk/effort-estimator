#!/usr/bin/env python3
"""
Comprehensive Model Visualization Suite
Creates publication-quality plots showing model performance, feature importance, and predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


def load_model_data():
    """Load trained model, scaler, and data"""
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/xgboost_models")
    
    # Load model and scaler
    model = xgb.XGBRegressor()
    model.load_model(str(output_dir / "xgboost_borg_10.0s.json"))
    
    with open(output_dir / "scaler_10.0s.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load data
    fused_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
    df = pd.read_csv(fused_path)
    df_labeled = df.dropna(subset=["borg"]).copy()
    
    # Filter features (same as training)
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            return True
        return False
    
    feature_cols = [col for col in df_labeled.columns if not is_metadata(col)]
    X = df_labeled[feature_cols].values
    y = df_labeled["borg"].values
    
    # Feature selection (top 100 by variance)
    feature_variance = np.var(X, axis=0)
    top_indices = np.argsort(feature_variance)[-100:][::-1]
    X_selected = X[:, top_indices]
    selected_cols = [feature_cols[i] for i in top_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'feature_cols': selected_cols,
        'df': df_labeled,
    }


def plot_predictions_detailed(data, output_dir):
    """Large detailed predictions vs actual plot with statistics"""
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Scatter plot
    scatter = ax.scatter(y_test, y_test_pred, s=100, alpha=0.6, 
                        c=np.abs(y_test - y_test_pred), cmap='RdYlGn_r',
                        edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
    
    # Regression line
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), 'b-', lw=2.5, label=f'Fit Line (slope={z[0]:.3f})', alpha=0.8)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    
    textstr = f'Test Set Metrics\n' \
              f'R² = {r2:.4f}\n' \
              f'MAE = {mae:.4f}\n' \
              f'RMSE = {rmse:.4f}\n' \
              f'n = {len(y_test)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    ax.set_title('Model Predictions vs Actual Effort Ratings\n(Color = Absolute Error)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_predictions_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_predictions_detailed.png")


def plot_residuals_analysis(data, output_dir):
    """Comprehensive residuals analysis"""
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_test_pred, residuals, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax.axhline(y=residuals.std(), color='orange', linestyle=':', lw=2, label='±1 Std Dev')
    ax.axhline(y=-residuals.std(), color='orange', linestyle=':', lw=2)
    ax.set_xlabel('Predicted Borg', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Residuals Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plot
    ax = axes[1, 0]
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.random.normal(0, residuals.std(), len(residuals))
    theoretical_quantiles = np.sort(theoretical_quantiles)
    ax.scatter(theoretical_quantiles, sorted_residuals, s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Error magnitude by actual value
    ax = axes[1, 1]
    abs_residuals = np.abs(residuals)
    ax.scatter(y_test, abs_residuals, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.plot(np.sort(y_test), np.mean(abs_residuals) * np.ones(len(y_test)), 
            'r--', lw=2, label=f'Mean Error: {np.mean(abs_residuals):.4f}')
    ax.set_xlabel('Actual Borg', fontsize=11, fontweight='bold')
    ax.set_ylabel('|Residuals|', fontsize=11, fontweight='bold')
    ax.set_title('Absolute Error vs Actual Effort', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_residuals_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_residuals_comprehensive.png")


def plot_train_test_comparison(data, output_dir):
    """Train vs Test performance comparison"""
    y_train = data['y_train']
    y_test = data['y_test']
    y_train_pred = data['y_train_pred']
    y_test_pred = data['y_test_pred']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train
    ax = axes[0]
    ax.scatter(y_train, y_train_pred, s=60, alpha=0.5, label='Training Samples', edgecolors='black', linewidth=0.5)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5)
    ax.set_xlabel('Actual Borg', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Borg', fontsize=11, fontweight='bold')
    r2_train = r2_score(y_train, y_train_pred)
    ax.set_title(f'Training Set (n={len(y_train)})\nR² = {r2_train:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Test
    ax = axes[1]
    ax.scatter(y_test, y_test_pred, s=80, alpha=0.6, label='Test Samples', 
              c=np.abs(y_test - y_test_pred), cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5)
    ax.set_xlabel('Actual Borg', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Borg', fontsize=11, fontweight='bold')
    r2_test = r2_score(y_test, y_test_pred)
    ax.set_title(f'Test Set (n={len(y_test)})\nR² = {r2_test:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_train_vs_test.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_train_vs_test.png")


def plot_feature_importance_detailed(data, output_dir):
    """Top features importance with more details"""
    model = data['model']
    feature_cols = data['feature_cols']
    
    # Get importance
    importance = model.get_booster().get_score(importance_type='weight')
    
    # Convert to feature names
    feature_importance = {}
    for fid, score in importance.items():
        idx = int(fid.split('f')[1])
        if idx < len(feature_cols):
            feature_importance[feature_cols[idx]] = score
    
    # Sort and get top 25
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_n = 25
    top_features = sorted_features[:top_n]
    names = [f[0].replace('_', ' ').title() for f in top_features]
    scores = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score, i, f' {score:.0f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features\n(Based on XGBoost Tree Splits)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_feature_importance.png")


def plot_error_distribution(data, output_dir):
    """Beautiful error distribution visualization"""
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    errors = np.abs(y_test - y_test_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    ax = axes[0]
    ax.hist(errors, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
    # Add KDE
    from scipy import stats
    x_range = np.linspace(errors.min(), errors.max(), 100)
    kde = stats.gaussian_kde(errors)
    ax.plot(x_range, kde(x_range), 'r-', lw=2.5, label='KDE')
    ax.axvline(errors.mean(), color='green', linestyle='--', lw=2.5, label=f'Mean: {errors.mean():.4f}')
    ax.axvline(np.median(errors), color='orange', linestyle='--', lw=2.5, label=f'Median: {np.median(errors):.4f}')
    ax.set_xlabel('Absolute Error (Borg Points)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative error
    ax = axes[1]
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, lw=3, color='darkblue')
    
    # Add percentile markers
    percentiles = [50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(errors, p)
        ax.plot(val, p, 'ro', markersize=8)
        ax.text(val, p + 2, f'{p}th: {val:.3f}', fontsize=9, ha='center')
    
    ax.set_xlabel('Absolute Error (Borg Points)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentile (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_error_distribution.png")


def plot_model_metrics_summary(data, output_dir):
    """Summary metrics as beautiful visualization"""
    y_train = data['y_train']
    y_test = data['y_test']
    y_train_pred = data['y_train_pred']
    y_test_pred = data['y_test_pred']
    
    # Calculate metrics
    metrics_data = {
        'R² Score': [r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)],
        'MAE': [mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
        'RMSE': [np.sqrt(mean_squared_error(y_train, y_train_pred)), 
                 np.sqrt(mean_squared_error(y_test, y_test_pred))],
    }
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    train_vals = [metrics_data[m][0] for m in metrics_data.keys()]
    test_vals = [metrics_data[m][1] for m in metrics_data.keys()]
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score / Error', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Training vs Test', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data.keys(), fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample size annotation
    textstr = f'Training samples: {len(y_train)}\nTest samples: {len(y_test)}\nTotal: {len(y_train) + len(y_test)}'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_metrics_summary.png")


def main():
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE MODEL VISUALIZATIONS")
    print("="*80 + "\n")
    
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/xgboost_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Loading model and data...")
    data = load_model_data()
    
    print("🎨 Creating visualizations...\n")
    plot_predictions_detailed(data, output_dir)
    plot_residuals_analysis(data, output_dir)
    plot_train_test_comparison(data, output_dir)
    plot_feature_importance_detailed(data, output_dir)
    plot_error_distribution(data, output_dir)
    plot_model_metrics_summary(data, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. 01_predictions_detailed.png - Predictions vs Actual with error color mapping")
    print("  2. 02_residuals_comprehensive.png - 4-panel residual analysis")
    print("  3. 03_train_vs_test.png - Training vs Test comparison")
    print("  4. 04_feature_importance.png - Top 25 important features")
    print("  5. 05_error_distribution.png - Histogram and cumulative error")
    print("  6. 06_metrics_summary.png - Key metrics comparison")


if __name__ == "__main__":
    main()
