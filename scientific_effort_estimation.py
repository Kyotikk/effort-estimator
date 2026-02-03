#!/usr/bin/env python3
"""
SCIENTIFICALLY GROUNDED EFFORT ESTIMATION
Based on exercise physiology literature

Key principle: Effort = physiological INTENSITY, not duration
- Resting for 1 hour = still 0 effort
- Sprinting for 10 seconds = high effort

Literature-based approaches:
1. Heart Rate Reserve (HRR) - Karvonen method
2. Metabolic Equivalent (MET) from accelerometry  
3. Combined cardio + movement intensity
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

BASE_PATH = Path("/Users/pascalschlegel/data/interim/parsingsim3")

def parse_time(t):
    try:
        return datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f').timestamp()
    except:
        return None

def compute_effort_features(subject):
    """Compute scientifically-grounded effort features."""
    subj_path = BASE_PATH / subject
    
    # Load ADL
    adl = pd.read_csv(subj_path / "scai_app" / "ADLs_1.csv", skiprows=2)
    adl.columns = ['Time', 'ADLs', 'Effort']
    adl['timestamp'] = adl['Time'].apply(parse_time)
    
    # Load HR
    hr = pd.read_csv(subj_path / "vivalnk_vv330_heart_rate" / "data_1.csv.gz")
    hr = hr.rename(columns={'time': 'timestamp', 'hr': 'heart_rate'})
    hr = hr[(hr['heart_rate'] > 30) & (hr['heart_rate'] < 220)]
    
    # Load wrist ACC
    acc_files = list((subj_path / "corsano_wrist_acc").glob("*.csv.gz"))
    acc = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
    acc = acc.rename(columns={'time': 'timestamp'})
    
    # Compute offsets
    adl_start = adl['timestamp'].min()
    hr_offset = adl_start - hr['timestamp'].min()
    acc_offset = adl_start - acc['timestamp'].min()
    
    # === BASELINE ESTIMATION (crucial for relative measures) ===
    # HR at rest: use 5th percentile as resting HR
    HR_rest = hr['heart_rate'].quantile(0.05)
    # HR max: use 95th percentile or age-predicted (220-age)
    HR_max_observed = hr['heart_rate'].quantile(0.95)
    # For elderly ~70yo: predicted max = 150
    HR_max = max(HR_max_observed, 150)  # Use higher of observed or predicted
    
    # ACC at rest: baseline when sitting/lying
    acc['magnitude'] = np.sqrt(acc['accX']**2 + acc['accY']**2 + acc['accZ']**2)
    ACC_rest = acc['magnitude'].quantile(0.10)  # Resting = gravity only (~500)
    
    print(f"  Baselines: HR_rest={HR_rest:.0f}, HR_max={HR_max:.0f}, ACC_rest={ACC_rest:.0f}")
    
    # Parse activities
    activities = []
    current = None
    start_time = None
    
    for _, row in adl.iterrows():
        if pd.isna(row['timestamp']):
            continue
        if 'Start' in str(row['ADLs']):
            current = row['ADLs'].replace(' Start', '')
            start_time = row['timestamp']
        elif 'End' in str(row['ADLs']) and current:
            duration = row['timestamp'] - start_time
            
            # Get HR during activity
            t_start = start_time - hr_offset
            t_end = row['timestamp'] - hr_offset
            mask = (hr['timestamp'] >= t_start) & (hr['timestamp'] <= t_end)
            hr_vals = hr.loc[mask, 'heart_rate'].values
            
            # Get ACC during activity
            t_start_acc = start_time - acc_offset
            t_end_acc = row['timestamp'] - acc_offset
            mask = (acc['timestamp'] >= t_start_acc) & (acc['timestamp'] <= t_end_acc)
            acc_vals = acc.loc[mask, 'magnitude'].values
            
            if len(hr_vals) < 2 or len(acc_vals) < 10:
                current = None
                continue
            
            # === SCIENTIFICALLY GROUNDED FEATURES ===
            
            # 1. HEART RATE RESERVE (%HRR) - Karvonen method
            # %HRR = (HR - HR_rest) / (HR_max - HR_rest)
            # This normalizes HR to 0-100% of capacity
            hr_mean = hr_vals.mean()
            hr_max_activity = hr_vals.max()
            HRR_mean = (hr_mean - HR_rest) / (HR_max - HR_rest) * 100
            HRR_max = (hr_max_activity - HR_rest) / (HR_max - HR_rest) * 100
            
            # 2. MOVEMENT INTENSITY (accelerometer)
            # Above-rest movement, normalized
            acc_mean = acc_vals.mean()
            acc_max = acc_vals.max()
            acc_std = acc_vals.std()
            # Movement above baseline (gravity-corrected)
            movement_intensity = max(0, acc_mean - ACC_rest)
            movement_peak = max(0, acc_max - ACC_rest)
            
            # 3. COMBINED INTENSITY (both systems)
            # Cardio-movement product
            intensity_combined = HRR_mean * movement_intensity / 100
            
            # 4. ENMO (Euclidean Norm Minus One) - standard in actigraphy
            # Already computed as movement_intensity essentially
            
            activities.append({
                'activity': current,
                'duration': duration,
                # Raw values
                'hr_mean': hr_mean,
                'hr_max': hr_max_activity,
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                # SCIENTIFIC FEATURES (intensity-based, not duration)
                'HRR_mean': HRR_mean,           # % Heart Rate Reserve (mean)
                'HRR_max': HRR_max,             # % Heart Rate Reserve (peak)
                'movement_intensity': movement_intensity,  # Movement above rest
                'movement_peak': movement_peak,  # Peak movement
                'intensity_combined': intensity_combined,  # HR × movement
                # Target
                'borg': float(row['Effort']) if pd.notna(row['Effort']) else np.nan
            })
            current = None
    
    return pd.DataFrame(activities).dropna()


print("="*70)
print("SCIENTIFICALLY GROUNDED EFFORT ESTIMATION")
print("="*70)
print("""
Literature-based principle:
  Effort = INTENSITY, not duration
  
Key features:
  1. %HRR (Heart Rate Reserve) - Karvonen formula
     %HRR = (HR - HR_rest) / (HR_max - HR_rest)
     
  2. Movement intensity above baseline
     Not absolute ACC, but ACC - resting_ACC
     
  3. Combined cardio + movement
""")

subjects = ['sim_elderly3', 'sim_healthy3', 'sim_severe3']
all_results = []

for subject in subjects:
    print(f"\n{'='*70}")
    print(f"SUBJECT: {subject}")
    print("="*70)
    
    try:
        df = compute_effort_features(subject)
        print(f"  Activities: {len(df)}")
        print(f"  Borg range: {df['borg'].min():.1f} - {df['borg'].max():.1f}")
        
        if df['borg'].max() - df['borg'].min() < 1:
            print("  ⚠️ Borg range too narrow for meaningful analysis")
            continue
        
        # Correlations
        print("\n  CORRELATIONS with Borg (intensity features):")
        print("  " + "-"*50)
        intensity_features = ['HRR_mean', 'HRR_max', 'movement_intensity', 
                             'movement_peak', 'intensity_combined']
        
        for feat in intensity_features:
            r, p = pearsonr(df[feat], df['borg'])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"    {feat:<22} r = {r:>6.3f} {sig}")
        
        # Compare with duration (biased feature)
        print("\n  vs Duration (potentially biased):")
        r, p = pearsonr(df['duration'], df['borg'])
        print(f"    duration                 r = {r:>6.3f}")
        
        # LOO-CV
        y = df['borg'].values
        loo = LeaveOneOut()
        
        feature_sets = [
            # Duration-free (scientifically correct)
            ('HRR_mean only', ['HRR_mean']),
            ('HRR_max only', ['HRR_max']),
            ('Movement intensity', ['movement_intensity']),
            ('HRR + Movement', ['HRR_mean', 'movement_intensity']),
            ('Combined intensity', ['intensity_combined']),
            # With duration (potentially biased)
            ('Duration only', ['duration']),
            ('HRR + Duration', ['HRR_mean', 'duration']),
        ]
        
        print("\n  LOO-CV R² (no duration vs with duration):")
        print("  " + "-"*50)
        
        results = []
        for name, feats in feature_sets:
            X = df[feats].values
            y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=loo)
            r2 = r2_score(y, y_pred)
            results.append((name, r2, 'duration' in feats))
        
        # Print without duration first
        print("  WITHOUT duration (pure intensity):")
        for name, r2, has_dur in results:
            if not has_dur:
                print(f"    {name:<25} CV R² = {r2:>6.3f}")
        
        print("\n  WITH duration (may be biased):")
        for name, r2, has_dur in results:
            if has_dur:
                print(f"    {name:<25} CV R² = {r2:>6.3f}")
        
        # Best intensity-only result
        best_no_dur = max([r for n, r, d in results if not d])
        best_with_dur = max([r for n, r, d in results if d])
        
        all_results.append({
            'subject': subject,
            'best_intensity_only': best_no_dur,
            'best_with_duration': best_with_dur,
            'n_activities': len(df),
            'borg_range': df['borg'].max() - df['borg'].min()
        })
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY: Scientific (intensity) vs Biased (duration)")
print("="*70)

for res in all_results:
    print(f"\n{res['subject']}:")
    print(f"  Pure intensity features:  CV R² = {res['best_intensity_only']:.3f}")
    print(f"  With duration:            CV R² = {res['best_with_duration']:.3f}")
    diff = res['best_with_duration'] - res['best_intensity_only']
    print(f"  Duration adds:            ΔR² = {diff:+.3f}")

print("\n" + "="*70)
print("SCIENTIFIC CONCLUSION")
print("="*70)
print("""
For HEALTHY populations:
  - Use %HRR (Heart Rate Reserve) as primary feature
  - Add movement intensity from accelerometer
  - Duration should NOT directly predict effort
  
For IMPAIRED populations (elderly/patients):
  - Sustained standing IS effortful (duration matters)
  - But this is a real physiological effect, not bias
  
Recommendation:
  If intensity-only R² << duration-only R²:
    → Check if your population has true endurance limitations
    → Or if Borg ratings are confounded with activity type
    
  If intensity R² ≈ duration R²:
    → Good! Intensity captures the real effort
""")
