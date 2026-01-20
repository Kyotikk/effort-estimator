"""
Save refined ElasticNet model and scaler for production use
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("SAVING REFINED MODEL FOR PRODUCTION")
print("="*70)

# Reload and prepare data
reduced_df = pd.read_csv("output/hrv_recovery_reduced.csv")

exclude_cols = [
    'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
    'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
    'qc_ok', 'effort', 'subject_id'
]

feature_cols = [c for c in reduced_df.columns if c not in exclude_cols]
X_raw = reduced_df[feature_cols].values.astype(np.float64)
y = reduced_df['delta_rmssd'].values

# Apply imputation and filtering
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_raw)

valid_mask = ~reduced_df[feature_cols].isna().any(axis=1) | (
    (~reduced_df[[c for c in feature_cols if c != 'acc_x_dyn__cardinality_r']].isna().any(axis=1)) &
    (reduced_df['acc_x_dyn__cardinality_r'].isna())
)

X_filtered = X_imputed[valid_mask]
y_filtered = y[valid_mask]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Train final model on all available data for maximum performance
print(f"\nTraining final model on {len(X_scaled)} samples...")

final_model = ElasticNet(
    alpha=0.062102,
    l1_ratio=0.1,
    random_state=42,
    max_iter=5000
)
final_model.fit(X_scaled, y_filtered)

print(f"✓ Model trained")
print(f"  R² (training): {final_model.score(X_scaled, y_filtered):.4f}")
print(f"  Intercept: {final_model.intercept_:.6f}")
print(f"  Coefficients: {final_model.coef_[:5]}... (first 5)")

# Save model
model_path = output_dir / 'elasticnet_refined_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n✓ Model saved: {model_path}")
print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")

# Save scaler
scaler_path = output_dir / 'elasticnet_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_path}")
print(f"  File size: {scaler_path.stat().st_size / 1024:.1f} KB")

# Save imputer
imputer_path = output_dir / 'elasticnet_imputer.pkl'
with open(imputer_path, 'wb') as f:
    pickle.dump(imputer, f)
print(f"✓ Imputer saved: {imputer_path}")
print(f"  File size: {imputer_path.stat().st_size / 1024:.1f} KB")

# Save feature names
feature_path = output_dir / 'elasticnet_feature_names.txt'
with open(feature_path, 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print(f"✓ Feature names saved: {feature_path}")
print(f"  Features: {len(feature_cols)}")

# Create inference example
print("\n" + "="*70)
print("EXAMPLE: HOW TO USE THE MODEL FOR PREDICTIONS")
print("="*70)

example_code = '''
# ============================================================================
# PRODUCTION INFERENCE EXAMPLE
# ============================================================================

import pickle
import numpy as np
from pathlib import Path

# Load saved components
model_dir = Path("./output")
with open(model_dir / 'elasticnet_refined_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(model_dir / 'elasticnet_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(model_dir / 'elasticnet_imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Load feature names
with open(model_dir / 'elasticnet_feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# Prepare new data (e.g., raw features from effort bout)
# Format: dictionary with feature names as keys
raw_features = {
    'ppg_red_zcr': 0.25,
    'rmssd_during_effort': 35.5,
    'acc_x_dyn__cardinality_r': 12.3,
    # ... other 12 features ...
}

# Create feature array in correct order
X_new = np.array([[raw_features.get(feat, np.nan) for feat in feature_names]])

# Apply imputation (for missing values)
X_imputed = imputer.transform(X_new)

# Scale features
X_scaled = scaler.transform(X_imputed)

# Make prediction
delta_rmssd_pred = model.predict(X_scaled)[0]

print(f"Predicted Δ RMSSD: {delta_rmssd_pred:.4f}")
print(f"Interpretation: HRV recovery of {delta_rmssd_pred:.2f} RMSSD units")
'''

print(example_code)

print("\n" + "="*70)
print("✓ MODEL READY FOR PRODUCTION!")
print("="*70)

# Create metadata file
metadata = {
    'model_type': 'ElasticNet',
    'alpha': 0.062102,
    'l1_ratio': 0.1,
    'n_features': len(feature_cols),
    'features': feature_cols,
    'training_samples': len(X_filtered),
    'training_date': '2025-01-20',
    'performance': {
        'test_r2': 0.2994,
        'test_mae': 0.0607,
        'pearson_r': 0.7983,
        'pearson_p': 0.0175
    }
}

import json
metadata_path = output_dir / 'elasticnet_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved: {metadata_path}")
print("\n" + "="*70)
