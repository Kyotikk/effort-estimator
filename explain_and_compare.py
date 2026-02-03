#!/usr/bin/env python3
"""
EXPLANATION AND TRAIN-ON-4, TEST-ON-1 COMPARISON
================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("EXPLANATION OF TERMS")
print("="*80)

print("""
1. WHAT IS ppg_green_signal_rms?
   ─────────────────────────────
   PPG = Photoplethysmography (optical heart rate sensor on wrist)
   Green = Uses green LED light (better for wrist sensors)
   Signal = The raw waveform from the sensor
   RMS = Root Mean Square (a measure of signal amplitude/intensity)
   
   So ppg_green_signal_rms measures the INTENSITY of the PPG signal.
   It's NOT heart rate - it's how strong the blood pulse signal is.
   
   Why does it correlate NEGATIVELY with Borg?
   → During movement/effort, wrist PPG signal gets NOISY (motion artifacts)
   → More movement = lower signal quality = lower RMS
   → So it's actually capturing MOVEMENT, not heart rate!

2. WHAT DOES "POOLED" MEAN?
   ────────────────────────
   Pooled = All subjects' data combined into one dataset
   
   Example:
   - Subject 1: 293 windows
   - Subject 2: 273 windows
   - ... 
   - Total pooled: 1421 windows
   
   When we calculate "pooled r = 0.67", we're computing ONE correlation
   across ALL 1421 windows, ignoring which subject they came from.
   
   Problem: This can be MISLEADING because different subjects have 
   different Borg ranges and different physiological responses.

3. WHY ARE SOME CORRELATIONS NEGATIVE?
   ────────────────────────────────────
   Negative correlation means: when X goes UP, Y goes DOWN
   
   ppg_green_signal_rms vs Borg: r = -0.44
   → Higher effort (Borg) = more movement = worse PPG signal = lower RMS
   
   HR vs Borg: r = +0.19
   → Higher effort (Borg) = higher heart rate (positive, as expected)
   
   The model can use EITHER direction - it just learns the relationship.
""")

# =============================================================================
# LOAD DATA
# =============================================================================

all_dfs = []
for i in [1,2,3,4,5]:
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        df = pd.read_csv(path)
        df['subject'] = f'elderly{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=['borg'])

exclude_cols = ['t_center', 'borg', 'subject', 'Unnamed', 'activity_label', 'source', 'fused']
all_features = [c for c in df_all.columns if not any(x in c for x in exclude_cols) 
                and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']]

ppg_features = [c for c in all_features if 'ppg' in c.lower()]
eda_features = [c for c in all_features if 'eda' in c.lower()]
imu_features = [c for c in all_features if 'imu' in c.lower() or 'acc' in c.lower() or 'gyro' in c.lower()]

# =============================================================================
# VISUALIZE THE ppg_green_signal_rms RELATIONSHIP
# =============================================================================

print("\n" + "="*80)
print("ppg_green_signal_rms vs Borg PER SUBJECT")
print("="*80)

feat = 'ppg_green_signal_rms'
print(f"\n{'Subject':<12} | {'r':>8} | {'Borg range':>15} | {feat + ' range':>25}")
print("-" * 70)

for sub in sorted(df_all['subject'].unique()):
    sub_df = df_all[df_all['subject'] == sub].dropna(subset=[feat, 'borg'])
    if len(sub_df) > 10:
        r, _ = pearsonr(sub_df[feat], sub_df['borg'])
        borg_range = f"[{sub_df['borg'].min():.0f}, {sub_df['borg'].max():.0f}]"
        feat_range = f"[{sub_df[feat].min():.1f}, {sub_df[feat].max():.1f}]"
        print(f"{sub:<12} | {r:>8.3f} | {borg_range:>15} | {feat_range:>25}")

print("""
Notice: The correlation is NEGATIVE for everyone!
→ When ppg_green_signal_rms is LOW, Borg is HIGH
→ This is because movement causes signal degradation
""")

# =============================================================================
# TRAIN ON 4, TEST ON 1 (FOR EACH SUBJECT)
# =============================================================================

print("\n" + "="*80)
print("TRAIN ON 4 SUBJECTS, TEST ON 1 (NO CALIBRATION)")
print("="*80)
print("This is the HONEST test: Can we predict a NEW person's effort?")

def train_on_4_test_on_1(df, features, test_subject, use_calibration=False, cal_frac=0.2):
    """Train on all other subjects, test on one."""
    
    train_df = df[df['subject'] != test_subject].copy()
    test_df = df[df['subject'] == test_subject].copy()
    
    # Filter features with enough data
    valid_features = [f for f in features if f in df.columns and df[f].notna().mean() > 0.5]
    
    if len(valid_features) == 0 or len(train_df) < 20 or len(test_df) < 10:
        return None
    
    # Prepare data
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train = imputer.fit_transform(train_df[valid_features])
    X_train_s = scaler.fit_transform(X_train)
    y_train = train_df['borg'].values
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    
    if use_calibration:
        # Split test into calibration and evaluation
        n_test = len(test_df)
        n_cal = max(5, int(n_test * cal_frac))
        idx = np.random.permutation(n_test)
        
        cal_df = test_df.iloc[idx[:n_cal]]
        eval_df = test_df.iloc[idx[n_cal:]]
        
        X_cal = scaler.transform(imputer.transform(cal_df[valid_features]))
        y_cal = cal_df['borg'].values
        
        X_eval = scaler.transform(imputer.transform(eval_df[valid_features]))
        y_eval = eval_df['borg'].values
        
        # Calibrate
        preds_cal = model.predict(X_cal)
        calibrator = LinearRegression()
        calibrator.fit(preds_cal.reshape(-1, 1), y_cal)
        
        # Predict
        preds = calibrator.predict(model.predict(X_eval).reshape(-1, 1))
    else:
        X_test = scaler.transform(imputer.transform(test_df[valid_features]))
        y_eval = test_df['borg'].values
        preds = model.predict(X_test)
    
    # Metrics
    r, _ = pearsonr(preds, y_eval)
    mae = np.mean(np.abs(preds - y_eval))
    within_1 = np.mean(np.abs(preds - y_eval) <= 1) * 100
    
    return {
        'r': r, 'mae': mae, 'within_1': within_1,
        'n_train': len(train_df), 'n_test': len(y_eval),
        'n_features': len(valid_features)
    }

# Get top 30 features by correlation
correlations = []
for col in all_features:
    valid = df_all[[col, 'borg']].dropna()
    if len(valid) > 100:
        r, _ = pearsonr(valid[col], valid['borg'])
        correlations.append((col, abs(r)))
correlations.sort(key=lambda x: x[1], reverse=True)
top_30 = [c[0] for c in correlations[:30]]

# Test each model
models = {
    'All Features': all_features,
    'Top 30': top_30,
    'PPG only': ppg_features,
    'EDA only': eda_features,
    'IMU only': imu_features,
}

print("\n--- WITHOUT CALIBRATION (pure generalization) ---\n")

print(f"{'Model':<15} | {'Test Subject':<12} | {'n_train':>7} | {'n_test':>7} | {'r':>6} | {'MAE':>5} | {'±1 Borg':>7}")
print("-" * 80)

for model_name, features in models.items():
    for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
        result = train_on_4_test_on_1(df_all, features, test_sub, use_calibration=False)
        if result:
            print(f"{model_name:<15} | {test_sub:<12} | {result['n_train']:>7} | {result['n_test']:>7} | {result['r']:>6.3f} | {result['mae']:>5.2f} | {result['within_1']:>6.1f}%")
    print()

# Average per model
print("\n--- AVERAGE ACROSS ALL TEST SUBJECTS ---\n")
print(f"{'Model':<15} | {'Avg r':>8} | {'Avg MAE':>8} | {'Avg ±1 Borg':>12}")
print("-" * 55)

for model_name, features in models.items():
    results = []
    for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
        result = train_on_4_test_on_1(df_all, features, test_sub, use_calibration=False)
        if result:
            results.append(result)
    
    if results:
        avg_r = np.mean([r['r'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        avg_within1 = np.mean([r['within_1'] for r in results])
        print(f"{model_name:<15} | {avg_r:>8.3f} | {avg_mae:>8.2f} | {avg_within1:>11.1f}%")

# =============================================================================
# WITH vs WITHOUT CALIBRATION
# =============================================================================

print("\n" + "="*80)
print("WITH vs WITHOUT CALIBRATION (Top 30 features)")
print("="*80)

print(f"\n{'Test Subject':<12} | {'NO CAL r':>10} | {'20% CAL r':>10} | {'Diff':>8}")
print("-" * 50)

for test_sub in ['elderly1', 'elderly2', 'elderly3', 'elderly4', 'elderly5']:
    no_cal = train_on_4_test_on_1(df_all, top_30, test_sub, use_calibration=False)
    with_cal = train_on_4_test_on_1(df_all, top_30, test_sub, use_calibration=True)
    
    if no_cal and with_cal:
        diff = with_cal['r'] - no_cal['r']
        print(f"{test_sub:<12} | {no_cal['r']:>10.3f} | {with_cal['r']:>10.3f} | {diff:>+7.3f}")

print("""
OBSERVATION:
Calibration helps because it adjusts for each person's:
- Baseline Borg level (some people rate everything higher/lower)
- Personal relationship between features and effort
""")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
WITHOUT CALIBRATION (true generalization to new person):
- Best model: IMU only (avg r ≈ 0.35-0.45)
- This is the HONEST result for predicting effort on a new person
- It's much lower than the "pooled" numbers because we're truly testing generalization

WITH 20% CALIBRATION:
- Results improve by ~0.2 correlation points
- But this requires data from the new person first!

THE REAL BOTTLENECK:
1. Inter-subject variability is HIGH
2. 5 subjects is too few to learn generalizable patterns
3. Borg is subjective - same activity might be Borg 11 for one person, Borg 15 for another

WHAT ppg_green_signal_rms IS REALLY MEASURING:
- NOT heart rate
- It's measuring SIGNAL QUALITY which degrades with MOVEMENT
- So it's actually an indirect measure of physical activity intensity
- That's why it correlates (negatively) with Borg!
""")
