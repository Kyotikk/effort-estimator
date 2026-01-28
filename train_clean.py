#!/usr/bin/env python3
"""Train XGBoost with ALL features including HRV."""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
df_labeled = df.dropna(subset=['borg'])
print(f'Labeled samples: {len(df_labeled)}')

# Feature columns (ONLY actual features, not metadata)
meta = ['t_center', 'borg', 'subject', 'activity', 'modality']
feature_cols = [c for c in df_labeled.columns if c not in meta and df_labeled[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Keep columns with <80% NaN (instead of dropping any with NaN)
X = df_labeled[feature_cols]
nan_pct = X.isna().mean()
good_cols = nan_pct[nan_pct < 0.8].index.tolist()
X = X[good_cols]

# Impute remaining NaN with median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

y = df_labeled['borg']
groups = df_labeled['subject']

print(f'Features: {len(X_imputed.columns)}')

# HRV check
hrv = [c for c in X_imputed.columns if 'rmssd' in c or 'hr_mean' in c or 'mean_ibi' in c or 'sdnn' in c]
print(f'HRV features in X: {hrv}')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'\nTest R²: {r2_score(y_test, y_pred):.3f}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')
print(f'Test MAE: {mean_absolute_error(y_test, y_pred):.3f}')

# Feature importance - top 20
importance = pd.DataFrame({
    'feature': X_imputed.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(20)
print(f'\nTop 20 features:')
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Check if any HRV in top features
hrv_in_top = [f for f in importance['feature'].values if any(h in f for h in ['rmssd', 'hr_mean', 'mean_ibi', 'sdnn'])]
print(f'\nHRV in top 20: {hrv_in_top}')

# Leave-one-subject-out CV
print('\n' + '='*60)
print('LEAVE-ONE-SUBJECT-OUT CV')
print('='*60)
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
r2_scores = []
for train_idx, test_idx in logo.split(X_imputed, y, groups):
    X_tr, X_te = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    subj = groups.iloc[test_idx].unique()[0]
    
    m = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    m.fit(X_tr, y_tr)
    y_p = m.predict(X_te)
    r2 = r2_score(y_te, y_p)
    r2_scores.append(r2)
    print(f'  {subj}: R² = {r2:.3f}')

print(f'\nLOSO Mean R²: {np.mean(r2_scores):.3f} (±{np.std(r2_scores):.3f})')
