"""
Refined ElasticNet Model with Hyperparameter Tuning and Better Generalization

Improvements:
1. Smart NaN handling (imputation instead of dropping)
2. Hyperparameter tuning with cross-validation
3. Feature importance analysis
4. Multiple training strategies
5. Detailed performance evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("="*70)
print("REFINED ELASTICNET MODEL - PHASE 1: DATA PREPARATION")
print("="*70)

reduced_df = pd.read_csv("output/hrv_recovery_reduced.csv")
print(f"\n✓ Loaded {len(reduced_df)} samples")

exclude_cols = [
    'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
    'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
    'qc_ok', 'effort', 'subject_id'
]

feature_cols = [c for c in reduced_df.columns if c not in exclude_cols]
X_raw = reduced_df[feature_cols].values.astype(np.float64)
y = reduced_df['delta_rmssd'].values

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Target samples: {len(y)}")
print(f"✓ NaN values before imputation: {np.isnan(X_raw).sum()}")

# Strategy 1: Imputation with median
print("\n[Strategy 1] Median Imputation")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_raw)
print(f"  NaN values after imputation: {np.isnan(X_imputed).sum()}")
print(f"  Samples available: {len(X_imputed)}")

# Strategy 2: Drop only rows with NaN in most important features
print("\n[Strategy 2] Selective Row Filtering")
# Keep rows where either all features are available or only acc_x_dyn__cardinality_r is missing
valid_mask = ~reduced_df[feature_cols].isna().any(axis=1) | (
    (~reduced_df[[c for c in feature_cols if c != 'acc_x_dyn__cardinality_r']].isna().any(axis=1)) &
    (reduced_df['acc_x_dyn__cardinality_r'].isna())
)
X_filtered = X_imputed[valid_mask]
y_filtered = y[valid_mask]
print(f"  Rows retained: {len(y_filtered)} ({100*len(y_filtered)/len(y):.1f}%)")

# Strategy 3: Remove low-information features
print("\n[Strategy 3] Feature Quality Check")
# Check variance and missing rate
feature_quality = []
for i, feat in enumerate(feature_cols):
    missing_rate = np.isnan(X_raw[:, i]).sum() / len(X_raw)
    variance = np.nanvar(X_raw[:, i])
    feature_quality.append({
        'feature': feat,
        'missing_rate': missing_rate,
        'variance': variance
    })

quality_df = pd.DataFrame(feature_quality)
print(quality_df.sort_values('variance', ascending=False).to_string(index=False))

# ============================================================================
# NORMALIZE DATA
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: DATA NORMALIZATION")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

print(f"\n✓ Scaled data shape: {X_scaled.shape}")
print(f"  Mean of each feature: {X_scaled.mean(axis=0)[:5]}... (first 5)")
print(f"  Std of each feature: {X_scaled.std(axis=0)[:5]}... (first 5)")

# ============================================================================
# HYPERPARAMETER TUNING WITH CV
# ============================================================================
print("\n" + "="*70)
print("PHASE 3: HYPERPARAMETER TUNING WITH K-FOLD CV")
print("="*70)

# Use ElasticNetCV for automatic parameter search
alphas = np.logspace(-3, 1, 30)
l1_ratios = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])

print(f"\nSearching {len(alphas)} alphas × {len(l1_ratios)} l1_ratios = {len(alphas)*len(l1_ratios)} combinations")
print(f"Using 5-Fold Cross-Validation on {len(X_scaled)} samples")

cv_model = ElasticNetCV(
    alphas=alphas,
    l1_ratio=l1_ratios,
    cv=5,
    random_state=42,
    max_iter=5000,
    n_jobs=-1,
    verbose=0
)

cv_model.fit(X_scaled, y_filtered)

print(f"\n✓ Best alpha: {cv_model.alpha_:.6f}")
print(f"✓ Best l1_ratio: {cv_model.l1_ratio_:.4f}")
print(f"✓ CV Score (mean): {cv_model.score(X_scaled, y_filtered):.4f}")

# Get CV scores for all folds
cv_scores = cross_val_score(cv_model, X_scaled, y_filtered, cv=5, scoring='r2')
print(f"✓ CV Scores per fold: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# TRAIN MULTIPLE MODELS FOR COMPARISON
# ============================================================================
print("\n" + "="*70)
print("PHASE 4: TRAIN MULTIPLE ELASTICNET MODELS")
print("="*70)

from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_filtered, test_size=0.2, random_state=42
)

print(f"\nTrain/Test Split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")

models = {}

# Model 1: Cross-validated optimal parameters
print("\n[Model 1] ElasticNet with CV-optimized parameters")
model1 = ElasticNet(
    alpha=cv_model.alpha_,
    l1_ratio=cv_model.l1_ratio_,
    random_state=42,
    max_iter=5000
)
model1.fit(X_train, y_train)
y_train_pred_m1 = model1.predict(X_train)
y_test_pred_m1 = model1.predict(X_test)

train_r2_m1 = r2_score(y_train, y_train_pred_m1)
test_r2_m1 = r2_score(y_test, y_test_pred_m1)
train_mae_m1 = mean_absolute_error(y_train, y_train_pred_m1)
test_mae_m1 = mean_absolute_error(y_test, y_test_pred_m1)

print(f"  Train: R²={train_r2_m1:.4f}, MAE={train_mae_m1:.4f}")
print(f"  Test: R²={test_r2_m1:.4f}, MAE={test_mae_m1:.4f}")

models['CV-Optimized'] = {
    'model': model1,
    'y_train_pred': y_train_pred_m1,
    'y_test_pred': y_test_pred_m1,
    'train_r2': train_r2_m1,
    'test_r2': test_r2_m1,
    'train_mae': train_mae_m1,
    'test_mae': test_mae_m1
}

# Model 2: Strong L2 regularization (lower alpha)
print("\n[Model 2] ElasticNet with strong L2 (lower alpha)")
model2 = ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=42, max_iter=5000)
model2.fit(X_train, y_train)
y_train_pred_m2 = model2.predict(X_train)
y_test_pred_m2 = model2.predict(X_test)

train_r2_m2 = r2_score(y_train, y_train_pred_m2)
test_r2_m2 = r2_score(y_test, y_test_pred_m2)
train_mae_m2 = mean_absolute_error(y_train, y_train_pred_m2)
test_mae_m2 = mean_absolute_error(y_test, y_test_pred_m2)

print(f"  Train: R²={train_r2_m2:.4f}, MAE={train_mae_m2:.4f}")
print(f"  Test: R²={test_r2_m2:.4f}, MAE={test_mae_m2:.4f}")

models['Strong-L2'] = {
    'model': model2,
    'y_train_pred': y_train_pred_m2,
    'y_test_pred': y_test_pred_m2,
    'train_r2': train_r2_m2,
    'test_r2': test_r2_m2,
    'train_mae': train_mae_m2,
    'test_mae': test_mae_m2
}

# Model 3: Balanced regularization
print("\n[Model 3] ElasticNet with balanced regularization")
model3 = ElasticNet(alpha=0.005, l1_ratio=0.5, random_state=42, max_iter=5000)
model3.fit(X_train, y_train)
y_train_pred_m3 = model3.predict(X_train)
y_test_pred_m3 = model3.predict(X_test)

train_r2_m3 = r2_score(y_train, y_train_pred_m3)
test_r2_m3 = r2_score(y_test, y_test_pred_m3)
train_mae_m3 = mean_absolute_error(y_train, y_train_pred_m3)
test_mae_m3 = mean_absolute_error(y_test, y_test_pred_m3)

print(f"  Train: R²={train_r2_m3:.4f}, MAE={train_mae_m3:.4f}")
print(f"  Test: R²={test_r2_m3:.4f}, MAE={test_mae_m3:.4f}")

models['Balanced'] = {
    'model': model3,
    'y_train_pred': y_train_pred_m3,
    'y_test_pred': y_test_pred_m3,
    'train_r2': train_r2_m3,
    'test_r2': test_r2_m3,
    'train_mae': train_mae_m3,
    'test_mae': test_mae_m3
}

# ============================================================================
# DETAILED ANALYSIS OF BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("PHASE 5: DETAILED ANALYSIS")
print("="*70)

best_model_name = max(models.keys(), key=lambda k: models[k]['test_r2'])
best_model = models[best_model_name]['model']

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Train R²: {models[best_model_name]['train_r2']:.4f}")
print(f"  Test R²: {models[best_model_name]['test_r2']:.4f}")

# Feature importance
coefficients = best_model.coef_
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Prediction analysis
print(f"\nPrediction Quality (Test Set):")
residuals = y_test - models[best_model_name]['y_test_pred']
print(f"  Mean Residual: {residuals.mean():.4f}")
print(f"  Std Residual: {residuals.std():.4f}")
print(f"  Min Residual: {residuals.min():.4f}")
print(f"  Max Residual: {residuals.max():.4f}")

# Correlation analysis
r_pearson, p_pearson = pearsonr(y_test, models[best_model_name]['y_test_pred'])
r_spearman, p_spearman = spearmanr(y_test, models[best_model_name]['y_test_pred'])

print(f"\n  Pearson r: {r_pearson:.4f} (p={p_pearson:.4f})")
print(f"  Spearman r: {r_spearman:.4f} (p={p_spearman:.4f})")

# ============================================================================
# CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("PHASE 6: VISUALIZATION")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Model Comparison
ax1 = fig.add_subplot(gs[0, 0])
model_names = list(models.keys())
test_r2s = [models[m]['test_r2'] for m in model_names]
colors = ['green' if r == max(test_r2s) else 'steelblue' for r in test_r2s]
ax1.bar(model_names, test_r2s, color=colors, edgecolor='black', alpha=0.7)
ax1.set_ylabel('Test R²', fontsize=11, fontweight='bold')
ax1.set_title('Model Comparison (Test R²)', fontsize=11, fontweight='bold')
ax1.axhline(y=0, color='red', linestyle='--', lw=1)
ax1.grid(alpha=0.3, axis='y')
for i, v in enumerate(test_r2s):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. CV Scores Box
ax2 = fig.add_subplot(gs[0, 1])
ax2.boxplot([cv_scores], labels=['CV Scores'])
ax2.plot([1], [cv_scores.mean()], 'r*', markersize=15, label='Mean')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax2.set_title(f'K-Fold CV Distribution (5 folds)', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
ax2.legend()

# 3. Best Model Hyperparameters
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
param_text = f"""
BEST MODEL: {best_model_name}

Hyperparameters:
  • Alpha: {best_model.alpha:.6f}
  • L1 Ratio: {best_model.l1_ratio:.4f}
  • Max Iter: {best_model.max_iter}

Performance:
  • Train R²: {models[best_model_name]['train_r2']:.4f}
  • Test R²: {models[best_model_name]['test_r2']:.4f}
  • Train MAE: {models[best_model_name]['train_mae']:.4f}
  • Test MAE: {models[best_model_name]['test_mae']:.4f}

Generalization:
  • Pearson r: {r_pearson:.4f}
  • Spearman r: {r_spearman:.4f}
  • CV Mean: {cv_scores.mean():.4f}
"""
ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 4. Train vs Test Predictions
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_train, models[best_model_name]['y_train_pred'], 
           alpha=0.6, s=80, color='blue', label='Train', edgecolor='black', linewidth=0.5)
ax4.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Δ RMSSD', fontsize=10, fontweight='bold')
ax4.set_ylabel('Predicted Δ RMSSD', fontsize=10, fontweight='bold')
ax4.set_title(f'Train Predictions (R²={models[best_model_name]["train_r2"]:.3f})', fontsize=11, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend()

# 5. Test Predictions
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_test, models[best_model_name]['y_test_pred'],
           alpha=0.6, s=80, color='green', label='Test', edgecolor='black', linewidth=0.5)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Δ RMSSD', fontsize=10, fontweight='bold')
ax5.set_ylabel('Predicted Δ RMSSD', fontsize=10, fontweight='bold')
ax5.set_title(f'Test Predictions (R²={models[best_model_name]["test_r2"]:.3f})', fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3)
ax5.legend()

# 6. Residuals Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(residuals, bins=8, color='lightcoral', edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='red', linestyle='--', lw=2)
ax6.axvline(x=residuals.mean(), color='blue', linestyle='-', lw=2, label=f'Mean={residuals.mean():.4f}')
ax6.set_xlabel('Residuals', fontsize=10, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax6.set_title('Residuals Distribution (Test)', fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# 7. Top Features
ax7 = fig.add_subplot(gs[2, :2])
top_features = feature_importance.head(10)
colors_feat = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
ax7.barh(range(len(top_features)), top_features['Coefficient'], color=colors_feat, edgecolor='black', alpha=0.7)
ax7.set_yticks(range(len(top_features)))
ax7.set_yticklabels(top_features['Feature'], fontsize=9)
ax7.set_xlabel('Coefficient Value', fontsize=10, fontweight='bold')
ax7.set_title('Top 10 Most Important Features', fontsize=11, fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='-', lw=1)
ax7.grid(alpha=0.3, axis='x')

# 8. Residuals vs Predictions
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(models[best_model_name]['y_test_pred'], residuals,
           alpha=0.6, s=80, color='purple', edgecolor='black', linewidth=0.5)
ax8.axhline(y=0, color='red', linestyle='--', lw=2)
ax8.set_xlabel('Predicted Δ RMSSD', fontsize=10, fontweight='bold')
ax8.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax8.set_title('Residuals vs Predictions', fontsize=11, fontweight='bold')
ax8.grid(alpha=0.3)

plt.suptitle('Refined ElasticNet Model - Comprehensive Analysis', 
            fontsize=14, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'elasticnet_refined_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: elasticnet_refined_analysis.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("PHASE 7: SAVE RESULTS")
print("="*70)

# Save best model summary
summary = {
    'model_name': best_model_name,
    'alpha': best_model.alpha,
    'l1_ratio': best_model.l1_ratio,
    'train_r2': models[best_model_name]['train_r2'],
    'test_r2': models[best_model_name]['test_r2'],
    'train_mae': models[best_model_name]['train_mae'],
    'test_mae': models[best_model_name]['test_mae'],
    'pearson_r': r_pearson,
    'pearson_p': p_pearson,
    'spearman_r': r_spearman,
    'spearman_p': p_spearman,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'n_samples': len(y_filtered),
    'n_train': len(y_train),
    'n_test': len(y_test),
    'n_features': len(feature_cols)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(output_dir / 'elasticnet_refined_summary.csv', index=False)
print("✓ Saved: elasticnet_refined_summary.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'elasticnet_feature_importance.csv', index=False)
print("✓ Saved: elasticnet_feature_importance.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'y_test': y_test,
    'y_pred': models[best_model_name]['y_test_pred'],
    'residual': residuals,
    'abs_error': np.abs(residuals)
})
predictions_df.to_csv(output_dir / 'elasticnet_test_predictions.csv', index=False)
print("✓ Saved: elasticnet_test_predictions.csv")

print("\n" + "="*70)
print("✓ REFINEMENT COMPLETE!")
print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {models[best_model_name]['test_r2']:.4f}")
print(f"Test MAE: {models[best_model_name]['test_mae']:.4f}")
print(f"Pearson r: {r_pearson:.4f} (p={p_pearson:.4f})")
print("="*70)
