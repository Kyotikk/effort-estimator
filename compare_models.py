#!/usr/bin/env python3
"""Quick model comparison"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

DATA_PATH = '/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv'
FEAT_PATH = '/Users/pascalschlegel/data/interim/elderly_combined/qc_5.0s/features_selected_pruned.csv'

df = pd.read_csv(DATA_PATH).dropna(subset=['borg'])
feat_cols = pd.read_csv(FEAT_PATH, header=None)[0].tolist()
feat_cols = [c for c in feat_cols if c in df.columns]

subjects = df['subject'].unique()

print('='*60)
print('COMPARING MODELS WITH SAME LOSO')
print('='*60)

for model_name in ['Ridge', 'XGBoost']:
    all_preds = []
    all_true = []
    
    for test_subj in subjects:
        train_df = df[df['subject'] != test_subj]
        test_df = df[df['subject'] == test_subj]
        
        X_train = train_df[feat_cols].values
        y_train = train_df['borg'].values
        X_test = test_df[feat_cols].values
        y_test = test_df['borg'].values
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        if model_name == 'Ridge':
            model = Ridge(alpha=1.0)
        else:
            model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
        
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
    
    r, _ = pearsonr(all_true, all_preds)
    mae = np.mean(np.abs(np.array(all_true) - np.array(all_preds)))
    within1 = np.mean(np.abs(np.array(all_true) - np.array(all_preds)) <= 1) * 100
    print(f'{model_name:10s}: r={r:.3f}, MAE={mae:.2f}, ±1 Borg={within1:.1f}%')

print('='*60)
print('\nThe original pipeline used XGBoost -> r=0.58')
print('My plot script used Ridge -> r=0.24')
print('THATS WHY THE NUMBERS CHANGED - MY MISTAKE!')
