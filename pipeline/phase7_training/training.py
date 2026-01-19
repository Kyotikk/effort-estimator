#!/usr/bin/env python3
"""
Train model FRESH and save everything needed for plotting
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def main():
    print("\n" + "="*100)
    print("TRAINING MODEL FRESH - SAVING ALL DATA FOR PLOTTING")
    print("="*100)
    
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/xgboost_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    fused_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
    df = pd.read_csv(fused_path)
    df_labeled = df.dropna(subset=["borg"]).copy()
    print(f"  ✓ Loaded {len(df_labeled)} labeled samples")
    
    # Filter metadata
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
    print(f"  ✓ {len(feature_cols)} features (metadata removed)")
    
    # Train-test split FIRST
    print("\n🔀 Train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  ✓ Train: {len(X_train)} samples")
    print(f"  ✓ Test: {len(X_test)} samples")
    
    # Feature selection on training data only
    print("\n🎯 Feature selection (top 100 by correlation)...")
    correlations = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
    correlations = np.abs(np.nan_to_num(correlations, nan=0))
    top_indices = np.argsort(correlations)[-100:][::-1]
    selected_cols = [feature_cols[i] for i in top_indices]
    
    X_train_sel = X_train[:, top_indices]
    
    eda_count_before = sum(1 for c in selected_cols if c.startswith('eda_'))
    imu_count_before = sum(1 for c in selected_cols if c.startswith('acc_'))
    ppg_count_before = sum(1 for c in selected_cols if c.startswith('ppg_'))
    print(f"  ✓ Before pruning - EDA: {eda_count_before}, IMU: {imu_count_before}, PPG: {ppg_count_before}")
    
    # Prune redundant features within each modality (threshold=0.90)
    print("\n🔪 Pruning redundant features within modalities (correlation threshold=0.90)...")
    pruned_indices = []
    pruned_cols = []
    corr_threshold = 0.90
    
    for modality_prefix in ['eda_', 'acc_', 'ppg_']:
        # Get indices for this modality in selected set
        mod_mask = [i for i, c in enumerate(selected_cols) if c.startswith(modality_prefix)]
        if len(mod_mask) <= 1:
            pruned_indices.extend([top_indices[i] for i in mod_mask])
            pruned_cols.extend([selected_cols[i] for i in mod_mask])
            continue
        
        # Compute correlation matrix for this modality
        X_mod = X_train_sel[:, mod_mask]
        corr_matrix = np.abs(np.corrcoef(X_mod.T))
        np.fill_diagonal(corr_matrix, 0.0)
        
        # Greedy pruning: remove one from each highly correlated pair
        keep = set(range(len(mod_mask)))
        while True:
            sub = corr_matrix[np.ix_(list(keep), list(keep))]
            max_val = sub.max() if sub.size > 0 else 0
            if max_val < corr_threshold:
                break
            keep_list = sorted(list(keep))
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            fi, fj = keep_list[i], keep_list[j]
            
            # Drop the one with lower correlation to Borg
            corr_fi = np.abs(np.corrcoef(X_mod[:, fi], y_train)[0, 1])
            corr_fj = np.abs(np.corrcoef(X_mod[:, fj], y_train)[0, 1])
            drop_idx = fi if corr_fi <= corr_fj else fj
            keep.discard(drop_idx)
        
        for idx in sorted(keep):
            pruned_indices.append(top_indices[mod_mask[idx]])
            pruned_cols.append(selected_cols[mod_mask[idx]])
    
    X_train_sel = X_train[:, pruned_indices]
    X_test_sel = X_test[:, pruned_indices]
    selected_cols = pruned_cols
    
    eda_count = sum(1 for c in selected_cols if c.startswith('eda_'))
    imu_count = sum(1 for c in selected_cols if c.startswith('acc_'))
    ppg_count = sum(1 for c in selected_cols if c.startswith('ppg_'))
    print(f"  ✓ After pruning - EDA: {eda_count}, IMU: {imu_count}, PPG: {ppg_count}")
    print(f"  ✓ Total features: {len(selected_cols)} (from {len(selected_cols) + sum([eda_count_before - eda_count, imu_count_before - imu_count, ppg_count_before - ppg_count])})")
    
    # Scale
    print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    print(f"  ✓ Scaled")
    
    # Train model
    print("\n🚀 Training XGBoost...")
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
    print(f"  ✓ Model trained")
    
    # Get predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    print("\n📊 METRICS:")
    print(f"\n  TRAINING SET (n={len(y_train)}):")
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"    R² = {train_r2:.4f}")
    print(f"    MAE = {train_mae:.4f}")
    print(f"    RMSE = {train_rmse:.4f}")
    
    print(f"\n  TEST SET (n={len(y_test)}):")
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"    R² = {test_r2:.4f}")
    print(f"    MAE = {test_mae:.4f}")
    print(f"    RMSE = {test_rmse:.4f}")
    
    # Save everything needed for plotting
    print("\n💾 Saving model, data, and metrics...")
    model.save_model(str(output_dir / "xgboost_borg_10.0s.json"))
    
    with open(output_dir / "scaler_10.0s.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Save train/test data
    save_data = {
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
    }
    
    with open(output_dir / "predictions_10.0s.pkl", "wb") as f:
        pickle.dump(save_data, f)
    
    # Save metrics
    metrics = {
        "10.0s": {
            "train": {
                "r2": float(train_r2),
                "mae": float(train_mae),
                "rmse": float(train_rmse),
                "n_samples": len(y_train),
            },
            "test": {
                "r2": float(test_r2),
                "mae": float(test_mae),
                "rmse": float(test_rmse),
                "n_samples": len(y_test),
            }
        }
    }
    
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Saved: xgboost_borg_10.0s.json")
    print(f"  ✓ Saved: scaler_10.0s.pkl")
    print(f"  ✓ Saved: predictions_10.0s.pkl")
    print(f"  ✓ Saved: evaluation_metrics.json")
    
    return y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols


def make_plots(y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols):
    """Make plots from actual data"""
    
    print("\n" + "="*100)
    print("CREATING PLOTS")
    print("="*100)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Plot 1: Train vs Test
    print("\n🎨 Creating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train
    ax = axes[0]
    ax.scatter(y_train, y_train_pred, s=80, alpha=0.6, color='#1f77b4', 
              edgecolors='black', linewidth=0.5)
    min_v, max_v = y_train.min(), y_train.max()
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=3, label='Perfect Prediction', zorder=5)
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    ax.set_title(f'TRAINING SET (n={len(y_train)})\nR² = {train_r2:.4f} | MAE = {train_mae:.4f}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Test
    ax = axes[1]
    errors = np.abs(y_test - y_test_pred)
    ax.scatter(y_test, y_test_pred, s=100, alpha=0.6, 
              c=errors, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    min_v, max_v = y_test.min(), y_test.max()
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=3, label='Perfect Prediction', zorder=5)
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    ax.set_title(f'TEST SET (n={len(y_test)})\nR² = {test_r2:.4f} | MAE = {test_mae:.4f}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Absolute Error', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/01_TRAIN_VS_TEST_ACTUAL.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 01_TRAIN_VS_TEST_ACTUAL.png")
    plt.close()
    
    # Plot 2: Metrics bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['R² Score', 'MAE', 'RMSE']
    train_vals = [train_r2, train_mae, np.sqrt(mean_squared_error(y_train, y_train_pred))]
    test_vals = [test_r2, test_mae, test_rmse]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Train vs Test\n(Higher R² is better, Lower MAE/RMSE is better)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/02_METRICS_ACTUAL.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 02_METRICS_ACTUAL.png")
    plt.close()
    
    print("\n" + "="*100)
    print("✅ COMPLETE!")
    print("="*100)
    print("\nGenerated plots:")
    print("  • 01_TRAIN_VS_TEST_ACTUAL.png")
    print("  • 02_METRICS_ACTUAL.png")
    print("\nAll plots show ACTUAL data from fresh training!")

    # --- Additional Plots ---
    # 3. Residuals vs. Predicted (Train/Test)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Train residuals
    axes[0].scatter(y_train_pred, y_train - y_train_pred, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_title('TRAINING SET Residuals', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    # Test residuals
    axes[1].scatter(y_test_pred, y_test - y_test_pred, alpha=0.6, color='#e74c3c', edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_title('TEST SET Residuals', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/03_RESIDUALS_VS_PREDICTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  • 03_RESIDUALS_VS_PREDICTED.png")

    # 4. Histogram/KDE of residuals (Train/Test)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.histplot(y_train - y_train_pred, bins=30, kde=True, color='#1f77b4', ax=axes[0])
    axes[0].set_title('TRAINING SET Residuals Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    sns.histplot(y_test - y_test_pred, bins=30, kde=True, color='#e74c3c', ax=axes[1])
    axes[1].set_title('TEST SET Residuals Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/04_RESIDUALS_HISTOGRAM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  • 04_RESIDUALS_HISTOGRAM.png")

    # 5. Error vs. True Value (Test set)
    fig, ax = plt.subplots(figsize=(10, 7))
    abs_errors = np.abs(y_test - y_test_pred)
    ax.scatter(y_test, abs_errors, alpha=0.7, color='#e67e22', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Error vs. True Borg Rating (Test Set)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/05_ERROR_VS_TRUE_TEST.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  • 05_ERROR_VS_TRUE_TEST.png")

    # 6. Predicted vs. True with density (Test set)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.kdeplot(x=y_test, y=y_test_pred, cmap="Blues", fill=True, thresh=0.05, levels=100, ax=ax)
    ax.scatter(y_test, y_test_pred, s=60, alpha=0.5, color='#2980b9', edgecolors='black', linewidth=0.5)
    min_v, max_v = y_test.min(), y_test.max()
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Borg Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Borg Rating', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs. Actual (Test Set) with Density', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/06_PREDICTED_VS_TRUE_DENSITY_TEST.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  • 06_PREDICTED_VS_TRUE_DENSITY_TEST.png")

    # 7. Feature Importance (TOP 30 from actual trained model)
    fig, ax = plt.subplots(figsize=(12, 11))
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': selected_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    top_n = 30
    top_features = feature_importance_df.head(top_n)
    
    colors = ['#2ecc71' if 'ppg_' in f else '#e74c3c' if 'eda_' in f else '#3498db' 
              for f in top_features['feature']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors, 
                   edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 30 Most Important Features (from Trained Model)', fontsize=13, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (feat, val) in enumerate(zip(top_features['feature'], top_features['importance'])):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='PPG'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='EDA'),
        Patch(facecolor='#3498db', edgecolor='black', label='IMU (Acc)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/pascalschlegel/effort-estimator/07_TOP_FEATURES_IMPORTANCE.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  • 07_TOP_FEATURES_IMPORTANCE.png")


if __name__ == "__main__":
    y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols = main()
    make_plots(y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae, test_rmse, model, selected_cols)
