import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
DF = pd.read_csv('rmssd_per_bout.csv')
DF = DF[(DF['n_beats'] >= 10) & (DF['duration'] >= 60)].copy()


# Only include therapy effort activities
effort_activities = [
    'Level Walking', 'Lower/Raise Pants', 'Sit to Stand', 'Stand to Sit', 'Lying to Sit',
    'Transfer to Bed', 'Transfer from Bed', 'Transfer to Toilet', 'Turn Over (right)',
    'Turn Over (left)', 'Sit to lying', 'Stand', 'Indoor Activity'
]
DF = DF[DF['activity'].isin(effort_activities)].copy()

# Normalize ln(RMSSD) within-patient
DF['ln_rmssd_z'] = DF.groupby('patient')['ln_rmssd'].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0)

# Assign health status
DF['status'] = DF['patient'].apply(lambda p: 'Healthy' if 'healthy' in p else ('Elderly' if 'elderly' in p else ('Severe' if 'severe' in p else 'Unknown')))

# Assign intensity (therapy effort focused)
high = ['Level Walking', 'Lower/Raise Pants', 'Sit to Stand', 'Stand to Sit', 'Lying to Sit', 'Transfer to Bed', 'Transfer from Bed', 'Transfer to Toilet']
moderate = ['Turn Over (right)', 'Turn Over (left)', 'Sit to lying', 'Indoor Activity']
low = ['Stand']
def categorize(a):
    if a in high:
        return 'High (dynamic)'
    elif a in moderate:
        return 'Moderate (postural)'
    elif a in low:
        return 'Low (static)'
    else:
        return 'Other'
DF['intensity'] = DF['activity'].apply(categorize)

# Boxplot: z-lnRMSSD by intensity, colored by status
plt.figure(figsize=(10,6))
sns.boxplot(x='intensity', y='ln_rmssd_z', hue='status', data=DF, showfliers=False)
sns.stripplot(x='intensity', y='ln_rmssd_z', hue='patient', data=DF, dodge=True, marker='o', alpha=0.5, linewidth=0.5, edgecolor='gray', legend=False)
plt.title('z-scored ln(RMSSD) by Activity Intensity and Health Status')
plt.ylabel('z-scored ln(RMSSD)')
plt.xlabel('Activity Intensity')
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('rmssd_z_by_intensity_status.png', dpi=150)
plt.close()

# Per-patient summary table
summary = DF.groupby(['patient','status']).agg({
    'ln_rmssd_z':['mean','std','count'],
    'intensity':lambda x: ', '.join(sorted(x.unique())),
    'activity':lambda x: ', '.join(sorted(x.unique()))
}).reset_index()
summary.columns = ['patient','status','z_lnrmssd_mean','z_lnrmssd_std','n_bouts','intensities','activities']
summary.to_csv('rmssd_per_patient_summary.csv', index=False)

print('Visualization saved as rmssd_z_by_intensity_status.png')
print('Per-patient summary saved as rmssd_per_patient_summary.csv')
print('\nPer-patient summary:')
print(summary.to_string(index=False))
