#!/usr/bin/env python3
"""
HRV Features Analysis - Single Patient (sim_elderly3) at Bout Level
Matching the literature-backed HR features plot style
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
})

OUT = "/Users/pascalschlegel/effort-estimator/thesis_plots_final"

# Color scheme (matching teal theme)
C_PRIMARY = '#3d7a8c'
C_PRIMARY_LIGHT = '#5a9aad'
C_PRIMARY_DARK = '#2c5a68'
C_ACCENT = '#c0392b'  # For scatter points

def main():
    # Load window-level data for elderly3
    df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_features_5.0s.csv')
    
    print(f"Total windows for sim_elderly3: {len(df)}")
    
    # We need borg labels - check if they exist or load from ADL
    if 'borg' not in df.columns:
        # Load aligned file that has borg
        aligned_path = '/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/elderly_sim_elderly3/fused_aligned_5.0s.csv'
        try:
            df = pd.read_csv(aligned_path)
            print(f"Loaded aligned file with {len(df)} rows")
        except:
            print("Could not find aligned file with borg labels")
            return
    
    print(f"Borg column exists: {'borg' in df.columns}")
    
    # HRV features we want to analyze
    hrv_features = {
        'ppg_green_mean_ibi': ('Mean IBI (ms)', C_PRIMARY),
        'ppg_green_rmssd': ('RMSSD (ms)', C_PRIMARY_LIGHT),
        'ppg_green_sdnn': ('SDNN (ms)', C_PRIMARY_DARK),
        'ppg_green_hr_mean': ('Mean HR (bpm)', C_ACCENT),
        'ppg_green_pnn50': ('pNN50 (%)', '#8e44ad'),
    }
    
    # Create bout IDs based on consecutive same borg values
    df['bout_id'] = (df['borg'].diff() != 0).cumsum()
    
    # Aggregate to bout level
    agg_dict = {'borg': 'mean'}
    for col in hrv_features.keys():
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    bout_df = df.groupby('bout_id').agg(agg_dict).reset_index()
    bout_df = bout_df.dropna(subset=['borg'])
    print(f"Bouts with valid data: {len(bout_df)}")
    
    # ==========================================================================
    # CREATE PLOT - Similar to literature-backed features plot
    # ==========================================================================
    
    fig = plt.figure(figsize=(14, 10))
    
    fig.suptitle('HRV Features Analysis - Single Patient (sim_elderly3)\nBout-Level Aggregation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Gather valid features
    valid_features = []
    for col, (name, color) in hrv_features.items():
        if col in bout_df.columns and bout_df[col].notna().sum() > 3:
            valid_features.append((col, name, color))
    
    print(f"Valid features: {[f[1] for f in valid_features]}")
    
    # --- Top row: Feature correlation bar chart ---
    ax_bars = fig.add_subplot(2, 3, 1)
    
    correlations = []
    names = []
    colors = []
    
    for col, name, color in valid_features:
        valid = bout_df[[col, 'borg']].dropna()
        if len(valid) > 3:
            r, p = stats.pearsonr(valid[col], valid['borg'])
            correlations.append(r)  # Keep sign for HR (negative expected)
            names.append(name)
            colors.append(color)
    
    if correlations:
        y_pos = np.arange(len(names))
        # Color bars by sign
        bar_colors = [c if r > 0 else '#95a5a6' for r, c in zip(correlations, colors)]
        ax_bars.barh(y_pos, correlations, color=bar_colors, alpha=0.8, edgecolor='white')
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(names)
        ax_bars.set_xlabel('Correlation (r) with Borg CR10')
        ax_bars.set_title('HRV Features', fontweight='bold')
        ax_bars.set_xlim(-1, 1)
        ax_bars.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax_bars.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax_bars.axvline(-0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add values
        for i, r in enumerate(correlations):
            x_pos = r + 0.05 if r > 0 else r - 0.15
            ax_bars.text(x_pos, i, f'{r:.2f}', va='center', fontsize=9)
    
    # --- Scatter plots for each feature ---
    plot_positions = [2, 3, 4, 5, 6]
    for idx, (col, name, color) in enumerate(valid_features[:5]):
        ax = fig.add_subplot(2, 3, plot_positions[idx])
        
        valid = bout_df[[col, 'borg']].dropna()
        if len(valid) > 3:
            x = valid[col].values
            y = valid['borg'].values
            
            # Scatter
            ax.scatter(x, y, c=color, s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel(name)
            ax.set_ylabel('Borg CR10')
            ax.set_title(f'r = {r_value:.3f}', fontweight='bold')
        
        if idx >= 4:
            break
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/45_hrv_features_single_patient.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUT}/45_hrv_features_single_patient.png")
    plt.close()

if __name__ == '__main__':
    main()
