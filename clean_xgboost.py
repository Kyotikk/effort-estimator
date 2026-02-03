#!/usr/bin/env python3
"""
CLEAN XGBoost Pipeline - Use features that correlate with BORG (not time)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr

print("="*70)
print("CLEAN XGBOOST PIPELINE")
print("="*70)

# 1. LOAD DATA
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')
df = df.dropna(subset=['borg']).reset_index(drop=True)
y = df['borg'].values
print(f"\nData: {len(df)} windows, Borg range [{y.min():.1f}, {y.max():.1f}]")

# 2. GET ALL FEATURES
meta_cols = ['t_center', 'borg', 'activity', 'activity_id', 'subject_id', 'valid', 'n_samples', 'win_sec', 'modality']
X = df[[c for c in df.columns if c not in meta_cols and not c.startswith('Unnamed')]].copy()
X = X.loc[:, X.nunique() > 1]
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
print(f"Total features: {X.shape[1]}")

# 3. SELECT FEATURES BY CORRELATION WITH BORG (not MI!)
print("\n" + "-"*70)
print("FEATURE SELECTION BY CORRELATION WITH BORG")
print("-"*70)

correlations = []
for col in X.columns:
    r, p = pearsonr(X[col], y)
    correlations.append({'feature': col, 'r': r, 'abs_r': abs(r), 'p': p})

corr_df = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)

print("\nTop 15 features by |correlation| with Borg:")
for i, row in corr_df.head(15).iterrows():
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
    print(f"  r={row['r']:+.3f}{sig}  {row['feature']}")

# Use top 15 features by correlation
top_features = corr_df.head(15)['feature'].tolist()
print(f"\nSelected features: {top_features}")

# 4. GROUPKFOLD CV
print("\n" + "-"*70)
print("GROUPKFOLD CROSS-VALIDATION (by activity)")
print("-"*70)

# Activity boundaries from Borg changes
activity_ids = np.cumsum(np.diff(y, prepend=y[0]) != 0)
n_activities = len(np.unique(activity_ids))
print(f"Activities: {n_activities}")

X_sel = X[top_features]
gkf = GroupKFold(n_splits=5)
predictions = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sel, y, activity_ids)):
    X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train_sc, y_train)
    predictions[test_idx] = model.predict(X_test_sc)
    
    fold_r, _ = pearsonr(y_test, predictions[test_idx])
    print(f"  Fold {fold+1}: r={fold_r:.3f}")

# 5. RESULTS
print("\n" + "-"*70)
print("RESULTS")
print("-"*70)

overall_r, p = pearsonr(y, predictions)
rmse = np.sqrt(np.mean((y - predictions)**2))
mae = np.mean(np.abs(y - predictions))

print(f"""
  XGBoost (15 features selected by correlation):
    Pearson r:  {overall_r:.3f}{'***' if p < 0.001 else ''}
    RMSE:       {rmse:.2f} Borg points
    MAE:        {mae:.2f} Borg points

  Literature formula (HR_delta × √duration):
    Pearson r:  0.843***
    
  Gap: Literature formula still beats ML by {0.843 - overall_r:.2f}
""")

# 6. FEATURE IMPORTANCE
print("-"*70)
print("FEATURE IMPORTANCE (from final model)")
print("-"*70)

# Train on all data for importance
scaler = StandardScaler()
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
model.fit(scaler.fit_transform(X_sel), y)

importance = pd.DataFrame({
    'feature': top_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    print(f"  {row['importance']:.3f}  {row['feature']}")
