#!/usr/bin/env python3
"""
Analyze how HRV predicts Borg - what activities, what patterns?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load the data
    df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')
    
    print("=" * 70)
    print("HOW IS HRV (mean_ibi) PREDICTING BORG?")
    print("=" * 70)
    
    # Check for activity columns
    activity_cols = [c for c in df.columns if 'adl' in c.lower() or 'activity' in c.lower() or 'task' in c.lower()]
    print(f'\nActivity columns in data: {activity_cols}')
    
    # Look at HRV features
    hrv_cols = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_sdnn', 'ppg_green_hr_mean', 'ppg_green_n_peaks']
    hrv_available = [c for c in hrv_cols if c in df.columns]
    print(f'HRV features available: {hrv_available}')
    
    # Check correlation with borg for each subject
    print('\n' + '='*70)
    print('HRV vs BORG CORRELATION BY SUBJECT')
    print('='*70)
    
    for sub in sorted(df['subject_id'].unique()):
        sub_df = df[df['subject_id'] == sub].dropna(subset=['borg'])
        print(f'\n{sub} (n={len(sub_df)}):')
        print(f'  Borg range: {sub_df["borg"].min():.1f} - {sub_df["borg"].max():.1f}')
        
        for col in hrv_available:
            if col in sub_df.columns:
                valid = sub_df[[col, 'borg']].dropna()
                if len(valid) > 10:
                    corr = valid[col].corr(valid['borg'])
                    print(f'  {col}: corr={corr:.3f}')
    
    # Load the original ADL files to understand activities
    print('\n' + '='*70)
    print('ACTIVITIES IN THE DATASET')
    print('='*70)
    
    for sub in ['sim_elderly3', 'sim_healthy3', 'sim_severe3']:
        adl_paths = list(Path(f'/Users/pascalschlegel/data/interim/parsingsim3/{sub}').rglob('*ADL*'))
        if adl_paths:
            adl_df = pd.read_csv(adl_paths[0])
            print(f'\n{sub} ADL file columns: {list(adl_df.columns)}')
            if 'ADL_ID' in adl_df.columns:
                print(f'  Activities: {adl_df["ADL_ID"].unique()[:10]}')
            if 'borg' in adl_df.columns or 'borg_rating' in adl_df.columns:
                borg_col = 'borg' if 'borg' in adl_df.columns else 'borg_rating'
                for adl_id in adl_df['ADL_ID'].unique()[:5]:
                    adl_subset = adl_df[adl_df['ADL_ID'] == adl_id]
                    borg_vals = adl_subset[borg_col].dropna()
                    if len(borg_vals) > 0:
                        print(f'    {adl_id}: Borg {borg_vals.mean():.1f}')
    
    # Analyze the physiological mechanism
    print('\n' + '='*70)
    print('WHY HRV (mean_ibi) PREDICTS EFFORT')
    print('='*70)
    
    print("""
    PHYSIOLOGICAL EXPLANATION:
    ─────────────────────────────────────────────────────────────────────
    
    mean_ibi = Mean Inter-Beat Interval (milliseconds between heartbeats)
    
    Higher effort → Higher heart rate → SHORTER IBI (lower mean_ibi)
    Lower effort  → Lower heart rate  → LONGER IBI (higher mean_ibi)
    
    So we expect: NEGATIVE correlation between mean_ibi and Borg
    (as effort increases, IBI decreases)
    """)
    
    # Verify the direction
    print('\nVERIFYING CORRELATION DIRECTION:')
    for sub in sorted(df['subject_id'].unique()):
        sub_df = df[df['subject_id'] == sub].dropna(subset=['borg'])
        if 'ppg_green_mean_ibi' in sub_df.columns:
            valid = sub_df[['ppg_green_mean_ibi', 'borg']].dropna()
            if len(valid) > 10:
                corr = valid['ppg_green_mean_ibi'].corr(valid['borg'])
                direction = "↓ IBI as effort ↑" if corr < 0 else "↑ IBI as effort ↑ (unexpected!)"
                print(f'  {sub}: corr={corr:.3f} → {direction}')
    
    # Create scatter plot
    print('\n' + '='*70)
    print('CREATING VISUALIZATION')
    print('='*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, sub in enumerate(sorted(df['subject_id'].unique())):
        ax = axes[i]
        sub_df = df[df['subject_id'] == sub].dropna(subset=['borg'])
        
        if 'ppg_green_mean_ibi' in sub_df.columns:
            valid = sub_df[['ppg_green_mean_ibi', 'borg']].dropna()
            
            ax.scatter(valid['ppg_green_mean_ibi'], valid['borg'], alpha=0.5, s=20)
            
            # Add trend line
            if len(valid) > 10:
                z = np.polyfit(valid['ppg_green_mean_ibi'], valid['borg'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid['ppg_green_mean_ibi'].min(), valid['ppg_green_mean_ibi'].max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (r={valid["ppg_green_mean_ibi"].corr(valid["borg"]):.2f})')
            
            ax.set_xlabel('Mean IBI (ms)', fontsize=11)
            ax.set_ylabel('Borg CR10', fontsize=11)
            ax.set_title(f'{sub}\n(n={len(valid)})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_top_features/hrv_vs_borg.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {out_path}')
    
    # Show what the model is learning
    print('\n' + '='*70)
    print('WHAT THE MODEL IS LEARNING')
    print('='*70)
    
    print("""
    The model learns from ALL labeled windows across the session:
    ─────────────────────────────────────────────────────────────────────
    
    1. DATA STRUCTURE:
       - Each 10-second window has features (HRV, EDA, PPG, IMU)
       - Windows are labeled with Borg rating during activities
       - Model sees ~400 labeled windows per subject
    
    2. WHAT HRV CAPTURES:
       - Heart rate changes with physical effort
       - mean_ibi (inter-beat interval) directly reflects heart rate:
         * Walking/exercise → faster HR → shorter IBI
         * Resting/light activity → slower HR → longer IBI
       
    3. WHY IT WORKS WELL FOR ELDERLY:
       - Elderly subject has widest Borg range (0.5-6.0)
       - Clear physiological response: effort → HR increase → IBI decrease
       - HRV is a validated biomarker for cardiovascular load
    
    4. XGBOOST LEARNING:
       - Learns decision rules like:
         "IF mean_ibi < 600ms AND eda_range > 5 THEN borg > 4"
       - Combines multiple features for robust prediction
    """)


if __name__ == "__main__":
    main()
