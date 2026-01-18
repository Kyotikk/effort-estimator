# Stage 10: Inference

## Purpose

Use trained condition-specific models to predict Borg effort from new sensor data.

---

## 10.1 Inference Process

### Overview

```
Raw Sensor Data (from preprocessing)
    ↓
[Feature Extraction] → 257 features
    ↓
[Feature Selection] → Top 100 for condition
    ↓
[Standardization] → Scale with condition scaler
    ↓
[Prediction] → XGBoost model.predict()
    ↓
Borg Effort (0-10)
```

### Code Template

```python
import xgboost as xgb
import pickle
import json
import numpy as np
import pandas as pd

def load_model_artifacts(condition):
    """Load trained model, scaler, and features for a condition."""
    base_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/models")
    
    model = xgb.XGBRegressor()
    model.load_model(str(base_path / f"{condition}_model.json"))
    
    with open(base_path / f"{condition}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(base_path / f"{condition}_features.json", 'r') as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names

def estimate_effort(features, condition):
    """
    Predict Borg effort for a feature vector.
    
    Args:
        features: numpy array or DataFrame with 257 feature columns
        condition: 'sim_elderly3', 'sim_healthy3', or 'sim_severe3'
    
    Returns:
        float: Predicted Borg effort (0-10)
    """
    # Load artifacts
    model, scaler, feature_names = load_model_artifacts(condition)
    
    # Ensure features is a DataFrame or array
    if isinstance(features, pd.Series):
        features = features.values.reshape(1, -1)
    elif isinstance(features, np.ndarray):
        if features.ndim == 1:
            features = features.reshape(1, -1)
    
    # Select top 100 features
    if isinstance(features, pd.DataFrame):
        features_selected = features[feature_names]
    else:
        # Convert to DataFrame temporarily for selection
        features_df = pd.DataFrame(features, columns=all_feature_names)
        features_selected = features_df[feature_names]
    
    # Standardize
    features_scaled = scaler.transform(features_selected)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    return prediction[0]  # Return scalar
```

---

## 10.2 Condition Selection

**Critical**: Subject condition must be known in advance.

```python
# From subject metadata/database
subject_condition_map = {
    'subject_001': 'elderly',      # → use sim_elderly3 model
    'subject_002': 'healthy',      # → use sim_healthy3 model
    'subject_003': 'severe',       # → use sim_severe3 model
}

subject_id = 'subject_001'
condition_name = subject_condition_map[subject_id]

# Map to model condition names
condition_model = {
    'elderly': 'sim_elderly3',
    'healthy': 'sim_healthy3',
    'severe': 'sim_severe3',
}

model_condition = condition_model[condition_name]

# Estimate effort
effort = estimate_effort(features, model_condition)
```

---

## 10.3 Single-Sample Inference

### Example 1: DataFrame Row

```python
# Load fused features
features_df = pd.read_csv("/path/to/fused_10.0s.csv")

# Take first window
window_features = features_df.iloc[0]  # Series of 257 features

# Estimate effort (assume elderly)
effort = estimate_effort(window_features, condition='sim_elderly3')

print(f"Estimated Borg effort: {effort:.2f}")
```

### Example 2: Preprocessed + Fused Pipeline

```python
from preprocessing.imu import preprocess_imu
from features.manual_features_imu import compute_imu_features

# Read raw sensor data
raw_imu = pd.read_csv("/path/to/corsano_bioz_acc.csv")

# Preprocess
imu_clean = preprocess_imu(raw_imu, config)

# Create windows
from windowing.windows import create_windows
windows = create_windows(imu_clean, window_length=10.0, overlap=0.70)

# Extract features
features = compute_imu_features(windows)

# Get effort for first window
effort = estimate_effort(features.iloc[0], condition='sim_elderly3')

print(f"Borg: {effort:.2f}")
```

---

## 10.4 Batch Inference

### Multiple Samples

```python
def batch_estimate_effort(features_df, condition):
    """
    Estimate effort for multiple samples.
    
    Args:
        features_df: DataFrame with rows of features
        condition: condition string
    
    Returns:
        list: Predicted efforts
    """
    efforts = []
    
    for idx in range(len(features_df)):
        effort = estimate_effort(features_df.iloc[idx], condition)
        efforts.append(effort)
    
    return efforts

# Load fused dataset
features_df = pd.read_csv("/path/to/fused_10.0s.csv")

# Estimate for all elderly subjects
elderly_mask = features_df['subject'] == 'sim_elderly3'
elderly_features = features_df[elderly_mask]

efforts = batch_estimate_effort(elderly_features, 'sim_elderly3')

# Add to DataFrame
features_df.loc[elderly_mask, 'effort_borg'] = efforts
```

### Vectorized Batch (Faster)

```python
def batch_estimate_effort_fast(features_df, condition):
    """Vectorized batch estimation (much faster)."""
    model, scaler, feature_names = load_model_artifacts(condition)
    
    # Select features
    X = features_df[feature_names]
    
    # Standardize (all at once)
    X_scaled = scaler.transform(X)
    
    # Predict (vectorized)
    predictions = model.predict(X_scaled)
    
    return predictions

# Usage
elderly_features = features_df[features_df['subject'] == 'sim_elderly3']
efforts = batch_estimate_effort_fast(elderly_features, 'sim_elderly3')
```

---

## 10.5 Model Performance by Condition

### Choose Model Based on Expected Effort

| If Effort Likely... | Use Model | Reason |
|-------------------|-----------|--------|
| **0-1.5 Borg** | healthy3 | R²=0.405, optimized for light |
| **0.5-6.0 Borg** | elderly3 | R²=0.926, well-distributed |
| **1.5-8.0 Borg** | severe3 | R²=0.997, best overall ⭐ |
| **Unknown range** | severe3 | Most robust, highest R² |

### Performance Metrics

| Condition | Train R² | Test R² | MAE | RMSE | Data Range |
|-----------|---------|---------|-----|------|-----------|
| elderly3 | 1.000 | 0.926 | 0.053 | 0.226 | 0.5-6.0 |
| healthy3 | 1.000 | 0.405 | 0.015 | 0.100 | 0.0-1.5 |
| severe3 | 1.000 | 0.997 | 0.026 | 0.112 | 1.5-8.0 |

---

## 10.6 Error Handling

```python
def estimate_effort_safe(features, condition, default=np.nan):
    """Estimate effort with error handling."""
    try:
        # Validate inputs
        if features is None or len(features) == 0:
            return default
        
        if condition not in ['sim_elderly3', 'sim_healthy3', 'sim_severe3']:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Load and predict
        effort = estimate_effort(features, condition)
        
        # Validate output
        if np.isnan(effort) or effort < 0 or effort > 10:
            print(f"Warning: Effort {effort} outside valid range [0, 10]")
            return default
        
        return effort
    
    except Exception as e:
        print(f"Error estimating effort: {e}")
        return default
```

---

## 10.7 Confidence & Uncertainty

**Current Models**: Return point estimates only.

**Future Enhancement**: Add confidence intervals.

```python
# Placeholder for future
def estimate_effort_with_confidence(features, condition):
    """
    Estimate effort with confidence interval.
    
    Future implementation using:
    - Quantile regression
    - Ensemble methods
    - Bayesian approaches
    """
    point_estimate = estimate_effort(features, condition)
    
    # TODO: Add confidence interval calculation
    
    return {
        'effort': point_estimate,
        'lower_ci': point_estimate - 0.5,  # Placeholder
        'upper_ci': point_estimate + 0.5,  # Placeholder
        'confidence': 0.95
    }
```

---

## 10.8 Production Deployment Checklist

Before deploying to production:

- [ ] Models saved and versioned
- [ ] Scalers saved and match training versions
- [ ] Feature lists saved (top 100 per condition)
- [ ] Model performance validated on test set
- [ ] Inference code tested end-to-end
- [ ] Error handling implemented
- [ ] Input validation working
- [ ] Output range checked (0-10 Borg)
- [ ] Documentation updated
- [ ] Logging configured

---

## Summary

✅ **Three trained models** ready for inference
✅ **Condition-specific** for optimized performance
✅ **Simple interface**: `estimate_effort(features, condition)`
✅ **Batch support** for efficient processing
✅ **Error handling** for production robustness
⚠️ **Condition required**: Must know if elderly/healthy/severe in advance
