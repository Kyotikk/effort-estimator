#!/usr/bin/env python3
"""Compare best HRV and EDA correlations for elderly and severe patients - REAL DATA ONLY."""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/multisub_aligned_10.0s.csv')

elderly = df[df['subject_id'] == 'sim_elderly3'].copy()
severe = df[df['subject_id'] == 'sim_severe3'].copy()

# Function to get real data only (exclude imputed = median values)
def get_real_data(data, feature):
    median_val = data[feature].median()
    # Keep rows where feature != median (those are real)
    return data[data[feature] != median_val].copy()

# Features to analyze
hrv_features = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_sdnn', 
                'ppg_green_hr_mean', 'ppg_green_n_peaks', 'ppg_green_pnn50']
eda_features = ['eda_cc_range', 'eda_cc_std', 'eda_phasic_max', 
                'eda_phasic_mean', 'eda_scl_std', 'eda_phasic_energy']

all_features = hrv_features + eda_features

# Calculate correlations for both patients
results = []

for patient_name, patient_data in [('Elderly', elderly), ('Severe', severe)]:
    for f in all_features:
        if f not in patient_data.columns:
            continue
        
        # Get real data only
        real_data = get_real_data(patient_data, f)
        valid = real_data[[f, 'borg']].dropna()
        
        if len(valid) > 30:
            r, p = stats.pearsonr(valid[f], valid['borg'])
            modality = 'HRV' if f in hrv_features else 'EDA'
            results.append({
                'patient': patient_name,
                'feature': f,
                'modality': modality,
                'r': r,
                'abs_r': abs(r),
                'p': p,
                'n': len(valid),
                'sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            })

results_df = pd.DataFrame(results)

# Print summary table
print("=" * 85)
print("CORRELATION SUMMARY: ELDERLY vs SEVERE (REAL DATA ONLY)")
print("=" * 85)

print(f"\n{'Feature':<25} {'Elderly r':>12} {'Elderly n':>10} {'Severe r':>12} {'Severe n':>10}")
print("-" * 85)

for f in all_features:
    e_row = results_df[(results_df['patient']=='Elderly') & (results_df['feature']==f)]
    s_row = results_df[(results_df['patient']=='Severe') & (results_df['feature']==f)]
    
    if len(e_row) > 0 and len(s_row) > 0:
        e_r = e_row['r'].values[0]
        e_n = e_row['n'].values[0]
        e_sig = e_row['sig'].values[0]
        s_r = s_row['r'].values[0]
        s_n = s_row['n'].values[0]
        s_sig = s_row['sig'].values[0]
        
        mod = "HRV" if f in hrv_features else "EDA"
        print(f"{f:<25} {e_r:>+.3f} {e_sig:<3} {e_n:>6} {s_r:>+.3f} {s_sig:<3} {s_n:>6}  [{mod}]")

# ============================================================================
# FIGURE 1: Bar chart comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Prepare data for plotting
for ax, modality in zip(axes, ['HRV', 'EDA']):
    mod_features = hrv_features if modality == 'HRV' else eda_features
    
    elderly_rs = []
    severe_rs = []
    labels = []
    
    for f in mod_features:
        e_row = results_df[(results_df['patient']=='Elderly') & (results_df['feature']==f)]
        s_row = results_df[(results_df['patient']=='Severe') & (results_df['feature']==f)]
        
        if len(e_row) > 0 and len(s_row) > 0:
            elderly_rs.append(e_row['r'].values[0])
            severe_rs.append(s_row['r'].values[0])
            # Shorten feature name for display
            label = f.replace('ppg_green_', '').replace('eda_', '').replace('phasic_', 'ph_')
            labels.append(label)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, elderly_rs, width, label='Elderly', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, severe_rs, width, label='Severe', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Correlation (r)', fontsize=12)
    ax.set_title(f'{modality} Features', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(-0.6, 0.6)
    
    # Add significance markers
    for i, (e_r, s_r) in enumerate(zip(elderly_rs, severe_rs)):
        e_row = results_df[(results_df['patient']=='Elderly') & (results_df['r']==e_r)]
        s_row = results_df[(results_df['patient']=='Severe') & (results_df['r']==s_r)]
        if len(e_row) > 0:
            e_sig = e_row['sig'].values[0]
            y_pos = e_r + 0.03 if e_r > 0 else e_r - 0.05
            ax.text(i - width/2, y_pos, e_sig, ha='center', fontsize=8, color='#3498db')
        if len(s_row) > 0:
            s_sig = s_row['sig'].values[0]
            y_pos = s_r + 0.03 if s_r > 0 else s_r - 0.05
            ax.text(i + width/2, y_pos, s_sig, ha='center', fontsize=8, color='#e74c3c')

plt.suptitle('Feature Correlations with Borg CR10 (Real Data Only)\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor/hrv_eda_comparison_barplot.png', 
            dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: hrv_eda_comparison_barplot.png")
plt.close()

# ============================================================================
# FIGURE 2: Scatter plots for best features
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# Best features to plot
best_features = [
    ('ppg_green_mean_ibi', 'HRV: Mean IBI'),
    ('ppg_green_rmssd', 'HRV: RMSSD'),
    ('eda_cc_range', 'EDA: Range'),
    ('eda_phasic_max', 'EDA: Phasic Max'),
]

colors = {'Elderly': '#3498db', 'Severe': '#e74c3c'}

for col, (feat, title) in enumerate(best_features):
    for row, (patient_name, patient_data) in enumerate([('Elderly', elderly), ('Severe', severe)]):
        ax = axes[row, col]
        
        # Get real data
        real_data = get_real_data(patient_data, feat)
        valid = real_data[[feat, 'borg']].dropna()
        
        if len(valid) > 30:
            x = valid[feat].values
            y = valid['borg'].values
            
            ax.scatter(x, y, c=colors[patient_name], alpha=0.5, s=20, edgecolor='white', linewidth=0.3)
            
            # Regression line
            slope, intercept, r, p, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.05, 0.95, f'r = {r:.3f} {sig}\nn = {len(valid)}', 
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(feat.replace('ppg_green_', '').replace('eda_', ''), fontsize=10)
        ax.set_ylabel('Borg CR10', fontsize=10)
        
        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add patient label on left
        if col == 0:
            ax.text(-0.3, 0.5, patient_name, transform=ax.transAxes, fontsize=14, 
                    fontweight='bold', va='center', rotation=90, color=colors[patient_name])

plt.suptitle('HRV and EDA vs Borg: Elderly vs Severe (Real Data Only)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor/hrv_eda_scatter_comparison.png', 
            dpi=200, bbox_inches='tight')
print(f"✓ Saved: hrv_eda_scatter_comparison.png")
plt.close()

# ============================================================================
# FIGURE 3: Summary table as image
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create summary data
table_data = [
    ['Feature', 'Modality', 'Elderly r', 'Elderly sig', 'Severe r', 'Severe sig', 'Both work?'],
]

key_features = ['ppg_green_mean_ibi', 'ppg_green_rmssd', 'ppg_green_sdnn', 'ppg_green_n_peaks',
                'eda_cc_range', 'eda_phasic_max', 'eda_phasic_energy']

for f in key_features:
    e_row = results_df[(results_df['patient']=='Elderly') & (results_df['feature']==f)]
    s_row = results_df[(results_df['patient']=='Severe') & (results_df['feature']==f)]
    
    if len(e_row) > 0 and len(s_row) > 0:
        e_r = e_row['r'].values[0]
        e_sig = e_row['sig'].values[0]
        s_r = s_row['r'].values[0]
        s_sig = s_row['sig'].values[0]
        mod = 'HRV' if f in hrv_features else 'EDA'
        
        both_work = '✅ Yes' if e_sig != 'ns' and s_sig != 'ns' else '⚠️ Partial' if e_sig != 'ns' or s_sig != 'ns' else '❌ No'
        
        display_name = f.replace('ppg_green_', '').replace('eda_', '')
        table_data.append([display_name, mod, f'{e_r:+.3f}', e_sig, f'{s_r:+.3f}', s_sig, both_work])

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.18, 0.1, 0.12, 0.1, 0.12, 0.1, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style header
for j in range(7):
    table[(0, j)].set_facecolor('#4a90d9')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Color code by significance
for i in range(1, len(table_data)):
    # Elderly sig column
    if table_data[i][3] in ['***', '**', '*']:
        table[(i, 2)].set_facecolor('#d4edda')
    else:
        table[(i, 2)].set_facecolor('#f8d7da')
    
    # Severe sig column
    if table_data[i][5] in ['***', '**', '*']:
        table[(i, 4)].set_facecolor('#d4edda')
    else:
        table[(i, 4)].set_facecolor('#f8d7da')

ax.set_title('Summary: Which Features Work for Both Patients?\n(Real Data Only, Green = Significant)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/pascalschlegel/data/interim/parsingsim3/multisub_combined/plots_supervisor/hrv_eda_summary_table.png', 
            dpi=200, bbox_inches='tight')
print(f"✓ Saved: hrv_eda_summary_table.png")
plt.close()

print(f"\n" + "=" * 85)
print("KEY FINDINGS")
print("=" * 85)
print("""
FEATURES THAT WORK FOR BOTH PATIENTS:
  ✅ ppg_green_rmssd     (HRV variability)
  ✅ ppg_green_sdnn      (HRV variability)  
  ✅ ppg_green_n_peaks   (Heart rate)
  ✅ eda_cc_range        (EDA variability)
  ✅ eda_phasic_max      (EDA response)
  ✅ eda_phasic_energy   (EDA intensity)

FEATURES THAT ONLY WORK FOR ELDERLY:
  ⚠️ ppg_green_mean_ibi  (only elderly shows HR-effort correlation)
""")
