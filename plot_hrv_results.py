"""
Visualization of HRV Recovery Model Results

Plots:
1. Feature importance
2. Train vs Test predictions
3. Residuals plot
4. Error metrics (MAE, RMSE)
5. Model comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
reduced_df = pd.read_csv("output/hrv_recovery_reduced.csv")
print(f"✓ Loaded {len(reduced_df)} samples")

# Exclude metadata columns
exclude_cols = [
    'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
    'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
    'qc_ok', 'effort', 'subject_id'
]

feature_cols = [c for c in reduced_df.columns if c not in exclude_cols]
X = reduced_df[feature_cols].values.astype(np.float64)
y = reduced_df['delta_rmssd'].values

# Handle NaN values
print(f"NaN count in X: {np.isnan(X).sum()}")
print(f"NaN count in y: {np.isnan(y).sum()}")

# Drop rows with any NaN
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[valid_mask]
y = y[valid_mask]
reduced_df = reduced_df[valid_mask].reset_index(drop=True)

print(f"✓ After removing NaN: {len(X)} samples")

# Train/test split (same as in training script)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
print(f"\nTraining models on {len(X_train)} samples...")
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet
en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)
en.fit(X_train_scaled, y_train)
y_train_pred_en = en.predict(X_train_scaled)
y_test_pred_en = en.predict(X_test_scaled)

# XGBoost
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                   random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_train_pred_xgb = xgb.predict(X_train)
y_test_pred_xgb = xgb.predict(X_test)

# Calculate metrics
def calc_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2}

metrics = []
metrics.append(calc_metrics(y_train, y_train_pred_en, "ElasticNet (Train)"))
metrics.append(calc_metrics(y_test, y_test_pred_en, "ElasticNet (Test)"))
metrics.append(calc_metrics(y_train, y_train_pred_xgb, "XGBoost (Train)"))
metrics.append(calc_metrics(y_test, y_test_pred_xgb, "XGBoost (Test)"))

metrics_df = pd.DataFrame(metrics)
print("\n" + "="*70)
print("METRICS")
print("="*70)
print(metrics_df.to_string(index=False))

# ============================================================================
# FIGURE 1: Feature Importance (XGBoost)
# ============================================================================
print("\n✓ Creating visualization 1: Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 6))

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

ax.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'plot_01_feature_importance.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_01_feature_importance.png")
plt.close()

# ============================================================================
# FIGURE 2: Train vs Test Predictions (ElasticNet)
# ============================================================================
print("✓ Creating visualization 2: Train/Test Predictions (ElasticNet)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
axes[0].scatter(y_train, y_train_pred_en, alpha=0.6, s=100, color='blue', label='Train')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Δ RMSSD', fontsize=11)
axes[0].set_ylabel('Predicted Δ RMSSD', fontsize=11)
axes[0].set_title(f'Train Set (n={len(y_train)})\nR²={r2_score(y_train, y_train_pred_en):.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

# Test
axes[1].scatter(y_test, y_test_pred_en, alpha=0.6, s=100, color='green', label='Test')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Δ RMSSD', fontsize=11)
axes[1].set_ylabel('Predicted Δ RMSSD', fontsize=11)
axes[1].set_title(f'Test Set (n={len(y_test)})\nR²={r2_score(y_test, y_test_pred_en):.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.suptitle('ElasticNet: Actual vs Predicted', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'plot_02_predictions_elasticnet.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_02_predictions_elasticnet.png")
plt.close()

# ============================================================================
# FIGURE 3: Train vs Test Predictions (XGBoost)
# ============================================================================
print("✓ Creating visualization 3: Train/Test Predictions (XGBoost)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
axes[0].scatter(y_train, y_train_pred_xgb, alpha=0.6, s=100, color='blue', label='Train')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Δ RMSSD', fontsize=11)
axes[0].set_ylabel('Predicted Δ RMSSD', fontsize=11)
axes[0].set_title(f'Train Set (n={len(y_train)})\nR²={r2_score(y_train, y_train_pred_xgb):.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

# Test
axes[1].scatter(y_test, y_test_pred_xgb, alpha=0.6, s=100, color='green', label='Test')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Δ RMSSD', fontsize=11)
axes[1].set_ylabel('Predicted Δ RMSSD', fontsize=11)
axes[1].set_title(f'Test Set (n={len(y_test)})\nR²={r2_score(y_test, y_test_pred_xgb):.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.suptitle('XGBoost: Actual vs Predicted', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'plot_03_predictions_xgboost.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_03_predictions_xgboost.png")
plt.close()

# ============================================================================
# FIGURE 4: Residuals Plot
# ============================================================================
print("✓ Creating visualization 4: Residuals...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ElasticNet Test Residuals
residuals_en = y_test - y_test_pred_en
axes[0, 0].scatter(y_test_pred_en, residuals_en, alpha=0.6, s=100, color='green')
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Δ RMSSD', fontsize=11)
axes[0, 0].set_ylabel('Residuals', fontsize=11)
axes[0, 0].set_title('ElasticNet (Test) - Residuals vs Predicted', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# XGBoost Test Residuals
residuals_xgb = y_test - y_test_pred_xgb
axes[0, 1].scatter(y_test_pred_xgb, residuals_xgb, alpha=0.6, s=100, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Δ RMSSD', fontsize=11)
axes[0, 1].set_ylabel('Residuals', fontsize=11)
axes[0, 1].set_title('XGBoost (Test) - Residuals vs Predicted', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# ElasticNet Residuals Distribution
axes[1, 0].hist(residuals_en, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('ElasticNet (Test) - Residuals Distribution', fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# XGBoost Residuals Distribution
axes[1, 1].hist(residuals_xgb, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('XGBoost (Test) - Residuals Distribution', fontsize=11, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'plot_04_residuals.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_04_residuals.png")
plt.close()

# ============================================================================
# FIGURE 5: Error Metrics Comparison
# ============================================================================
print("✓ Creating visualization 5: Error Metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['ElasticNet', 'XGBoost']
mae_train = [
    mean_absolute_error(y_train, y_train_pred_en),
    mean_absolute_error(y_train, y_train_pred_xgb)
]
mae_test = [
    mean_absolute_error(y_test, y_test_pred_en),
    mean_absolute_error(y_test, y_test_pred_xgb)
]
rmse_train = [
    np.sqrt(mean_squared_error(y_train, y_train_pred_en)),
    np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
]
rmse_test = [
    np.sqrt(mean_squared_error(y_test, y_test_pred_en)),
    np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
]
r2_train = [
    r2_score(y_train, y_train_pred_en),
    r2_score(y_train, y_train_pred_xgb)
]
r2_test = [
    r2_score(y_test, y_test_pred_en),
    r2_score(y_test, y_test_pred_xgb)
]

x = np.arange(len(models))
width = 0.35

# MAE
axes[0].bar(x - width/2, mae_train, width, label='Train', color='skyblue')
axes[0].bar(x + width/2, mae_test, width, label='Test', color='lightcoral')
axes[0].set_ylabel('MAE', fontsize=12, fontweight='bold')
axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
for i, v in enumerate(mae_train):
    axes[0].text(i - width/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
for i, v in enumerate(mae_test):
    axes[0].text(i + width/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

# RMSE
axes[1].bar(x - width/2, rmse_train, width, label='Train', color='skyblue')
axes[1].bar(x + width/2, rmse_test, width, label='Test', color='lightcoral')
axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')
for i, v in enumerate(rmse_train):
    axes[1].text(i - width/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
for i, v in enumerate(rmse_test):
    axes[1].text(i + width/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

# R²
axes[2].bar(x - width/2, r2_train, width, label='Train', color='skyblue')
axes[2].bar(x + width/2, r2_test, width, label='Test', color='lightcoral')
axes[2].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models)
axes[2].legend()
axes[2].grid(alpha=0.3, axis='y')
axes[2].axhline(y=0, color='black', linestyle='-', lw=0.5)
for i, v in enumerate(r2_train):
    axes[2].text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(r2_test):
    axes[2].text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'plot_05_error_metrics.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_05_error_metrics.png")
plt.close()

# ============================================================================
# FIGURE 6: Model Comparison Summary
# ============================================================================
print("✓ Creating visualization 6: Model Comparison Summary...")
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Metrics table
ax_table = fig.add_subplot(gs[0, :])
ax_table.axis('tight')
ax_table.axis('off')

table_data = []
for _, row in metrics_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['MAE']:.4f}",
        f"{row['RMSE']:.4f}",
        f"{row['R²']:.4f}"
    ])

table = ax_table.table(
    cellText=table_data,
    colLabels=['Model', 'MAE', 'RMSE', 'R²'],
    cellLoc='center',
    loc='center',
    colColours=['lightgray']*4,
    cellColours=[['white']*4]*len(table_data)
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax_table.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)

# Distribution of actual values
ax1 = fig.add_subplot(gs[1, 0])
ax1.hist(y_train, bins=8, alpha=0.5, label='Train', color='blue')
ax1.hist(y_test, bins=8, alpha=0.5, label='Test', color='green')
ax1.set_xlabel('Δ RMSSD', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Target Variable', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Sample distribution
ax2 = fig.add_subplot(gs[1, 1])
subjects_count = reduced_df['subject_id'].value_counts()
ax2.bar(subjects_count.index, subjects_count.values, color='steelblue', edgecolor='black')
ax2.set_ylabel('Number of Samples', fontsize=11)
ax2.set_xlabel('Subject', fontsize=11)
ax2.set_title('Sample Distribution by Subject', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# Dataset info
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')
info_text = f"""
Dataset Summary:
  • Total Samples: {len(reduced_df)}
  • Training Samples: {len(X_train)} ({len(X_train)/len(reduced_df)*100:.1f}%)
  • Test Samples: {len(X_test)} ({len(X_test)/len(reduced_df)*100:.1f}%)
  • Features Used: {len(feature_cols)}
  • Target Variable: Δ RMSSD (HRV Recovery)
  • Range: [{y.min():.4f}, {y.max():.4f}]
  
Key Findings:
  • Best Model: ElasticNet (R²_test = {r2_score(y_test, y_test_pred_en):.4f})
  • Test MAE: {mean_absolute_error(y_test, y_test_pred_en):.4f}
  • Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_en)):.4f}
  • Pearson r (Test): {np.corrcoef(y_test, y_test_pred_en)[0,1]:.4f}
"""
ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig(output_dir / 'plot_06_summary.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_06_summary.png")
plt.close()

# ============================================================================
# FIGURE 7: Distribution Comparison
# ============================================================================
print("✓ Creating visualization 7: Distribution Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ElasticNet train predictions distribution
axes[0, 0].hist(y_train_pred_en, bins=10, alpha=0.7, color='skyblue', label='Predicted', edgecolor='black')
axes[0, 0].hist(y_train, bins=10, alpha=0.7, color='coral', label='Actual', edgecolor='black')
axes[0, 0].set_xlabel('Δ RMSSD', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('ElasticNet Train - Distribution', fontsize=11, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# ElasticNet test predictions distribution
axes[0, 1].hist(y_test_pred_en, bins=8, alpha=0.7, color='skyblue', label='Predicted', edgecolor='black')
axes[0, 1].hist(y_test, bins=8, alpha=0.7, color='coral', label='Actual', edgecolor='black')
axes[0, 1].set_xlabel('Δ RMSSD', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('ElasticNet Test - Distribution', fontsize=11, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# XGBoost train predictions distribution
axes[1, 0].hist(y_train_pred_xgb, bins=10, alpha=0.7, color='lightgreen', label='Predicted', edgecolor='black')
axes[1, 0].hist(y_train, bins=10, alpha=0.7, color='coral', label='Actual', edgecolor='black')
axes[1, 0].set_xlabel('Δ RMSSD', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('XGBoost Train - Distribution', fontsize=11, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# XGBoost test predictions distribution
axes[1, 1].hist(y_test_pred_xgb, bins=8, alpha=0.7, color='lightgreen', label='Predicted', edgecolor='black')
axes[1, 1].hist(y_test, bins=8, alpha=0.7, color='coral', label='Actual', edgecolor='black')
axes[1, 1].set_xlabel('Δ RMSSD', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('XGBoost Test - Distribution', fontsize=11, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'plot_07_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: plot_07_distribution_comparison.png")
plt.close()

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS COMPLETE!")
print("="*70)
print("\nGenerated plots:")
print("  1. plot_01_feature_importance.png")
print("  2. plot_02_predictions_elasticnet.png")
print("  3. plot_03_predictions_xgboost.png")
print("  4. plot_04_residuals.png")
print("  5. plot_05_error_metrics.png")
print("  6. plot_06_summary.png")
print("  7. plot_07_distribution_comparison.png")
print(f"\nAll saved to: {output_dir}/")
print("="*70)
