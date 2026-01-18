#!/usr/bin/env python3
"""
Enhanced Model Evaluation - Detailed plots like V1 version

Generates comprehensive evaluation plots:
- Predicted vs Actual with regression line
- Residual plots
- Error distribution by Borg score range
- Absolute error histogram
- Train vs Test overfitting check
- Performance summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import json


# Enhanced styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 14)
plt.rcParams['font.size'] = 10


def create_detailed_evaluation_plots(output_dir: str, fused_data_path: str):
    """Create comprehensive V1-style evaluation plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(fused_data_path)
    df_labeled = df.dropna(subset=["borg"]).copy()
    
    print(f"   Total samples: {len(df)}")
    print(f"   Labeled samples: {len(df_labeled)}")
    
    # Prepare features
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    feature_cols = [
        col for col in df_labeled.columns
        if col not in skip_cols and not col.endswith("_r")
    ]
    
    X = df_labeled[feature_cols].values
    y = df_labeled["borg"].values
    
    print(f"   Features: {len(feature_cols)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection
    k_features = 100
    feature_variance = np.var(X_train, axis=0)
    top_indices = np.argsort(feature_variance)[-k_features:]
    top_indices = np.sort(top_indices)
    
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    selected_cols = [feature_cols[i] for i in top_indices]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train model
    print("\n🔨 Training model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"   Train - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"   Test  - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    
    # Create comprehensive figure
    print("\n📊 Creating detailed evaluation plots...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Predicted vs Actual (TEST)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.scatter(y_test, y_test_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, c='steelblue')
    y_min, y_max = y_test.min(), y_test.max()
    ax1.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2.5, label='Perfect', alpha=0.8)
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_min, y_max, 100)
    ax1.plot(x_line, p(x_line), 'g-', lw=2.5, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', alpha=0.8)
    ax1.set_xlabel('Actual Borg', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Borg', fontsize=11, fontweight='bold')
    ax1.set_title(f'Predicted vs Actual (TEST)\nR²={test_r2:.4f} | Pearson r={np.corrcoef(y_test, y_test_pred)[0,1]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Residual Plot (TEST)
    ax2 = fig.add_subplot(gs[0, 2])
    residuals_test = y_test - y_test_pred
    ax2.scatter(y_test_pred, residuals_test, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, c='coral')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2.5, alpha=0.8)
    std_res = np.std(residuals_test)
    ax2.fill_between([y_test_pred.min(), y_test_pred.max()], -std_res, std_res, alpha=0.2, color='green')
    ax2.set_xlabel('Predicted Borg', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax2.set_title(f'Residual Plot (TEST)\nMean Residual: {residuals_test.mean():.4f}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute Error Distribution (TEST)
    ax3 = fig.add_subplot(gs[1, 2])
    abs_errors = np.abs(residuals_test)
    ax3.hist(abs_errors, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(test_mae, color='red', linestyle='--', lw=2.5, label=f'MAE={test_mae:.4f}')
    ax3.axvline(np.median(abs_errors), color='green', linestyle='--', lw=2.5, label=f'Median={np.median(abs_errors):.4f}')
    ax3.set_xlabel('Absolute Error (Borg Points)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Absolute Error Distribution (TEST)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. RMSE & MAE by Borg Score Range
    ax4 = fig.add_subplot(gs[1, 0:2])
    ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    range_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7']
    rmse_by_range = []
    mae_by_range = []
    
    for low, high in ranges:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            rmse_by_range.append(np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask])))
            mae_by_range.append(mean_absolute_error(y_test[mask], y_test_pred[mask]))
        else:
            rmse_by_range.append(0)
            mae_by_range.append(0)
    
    x = np.arange(len(range_labels))
    width = 0.35
    bars1 = ax4.bar(x - width/2, rmse_by_range, width, label='RMSE', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax4.bar(x + width/2, mae_by_range, width, label='MAE', color='coral', alpha=0.8, edgecolor='black')
    
    ax4.set_xlabel('Borg Score Range', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Error (Borg Points)', fontsize=11, fontweight='bold')
    ax4.set_title('RMSE & MAE by Borg Score Range', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(range_labels)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. OVERFITTING CHECK: Train vs Test
    ax5 = fig.add_subplot(gs[2, 0])
    metrics_names = ['R²', 'RMSE', 'MAE']
    train_vals = [train_r2, train_rmse, train_mae]
    test_vals = [test_r2, test_rmse, test_mae]
    
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax5.bar(x_pos - width/2, train_vals, width, label='Train', color='green', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x_pos + width/2, test_vals, width, label='Test', color='coral', alpha=0.7, edgecolor='black')
    
    ax5.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax5.set_title('OVERFITTING CHECK: Train vs Test', fontsize=12, fontweight='bold', color='darkred')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metrics_names, fontsize=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Feature Importance (Top 15)
    ax6 = fig.add_subplot(gs[2, 1:3])
    importances = model.feature_importances_
    top_n = 15
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_imp = importances[top_idx]
    top_feat = [selected_cols[i] for i in top_idx]
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax6.barh(range(top_n), top_imp, color=colors, edgecolor='black', linewidth=1)
    ax6.set_yticks(range(top_n))
    ax6.set_yticklabels(top_feat, fontsize=9)
    ax6.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax6.set_title(f'Top {top_n} Most Important Features', fontsize=12, fontweight='bold')
    ax6.invert_yaxis()
    ax6.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(bars, top_imp)):
        ax6.text(imp, i, f' {imp:.4f}', va='center', fontsize=8)
    
    # Add text box with performance summary
    textstr = f"""MODEL PERFORMANCE SUMMARY

TEST SET:
• R² Score: {test_r2:.4f}
• RMSE: {test_rmse:.4f}
• MAE: {test_mae:.4f}

TRAIN SET (Overfitting):
• R² Score: {train_r2:.4f}
• RMSE: {train_rmse:.4f}
• MAE: {train_mae:.4f}

COMPARISON TO V1:
• V1 R²: 0.9291
• V2 R²: {test_r2:.4f}
• Change: {((test_r2 - 0.9291)/0.9291)*100:+.1f}%

• V1 RMSE: 0.3611
• V2 RMSE: {test_rmse:.4f}
• Change: {((test_rmse - 0.3611)/0.3611)*100:+.1f}%
"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.suptitle('XGBoost V2 Model Evaluation (188 features: IMU+PPG_green+PPG_infra+PPG_red+EDA)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save plot
    plot_path = output_dir / "V2_detailed_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path.name}")
    plt.close()
    
    # Save model and scaler
    model_path = output_dir / "xgboost_borg_10.0s.json"
    scaler_path = output_dir / "scaler_10.0s.pkl"
    model.save_model(str(model_path))
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_data = {
        "10.0s": {
            "train": {
                "rmse": float(train_rmse),
                "mae": float(train_mae),
                "r2": float(train_r2),
                "n_samples": len(y_train),
            },
            "test": {
                "rmse": float(test_rmse),
                "mae": float(test_mae),
                "r2": float(test_r2),
                "n_samples": len(y_test),
            }
        }
    }
    
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Metrics Summary:")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.4f}")
    print(f"   Test MAE:  {test_mae:.4f}")
    print(f"\n📂 Output saved to: {output_dir}")


if __name__ == "__main__":
    output_dir = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/xgboost_models"
    fused_data = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv"
    
    create_detailed_evaluation_plots(output_dir, fused_data)
