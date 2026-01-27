import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
DF = pd.read_csv('rmssd_per_bout.csv')
DF = DF[(DF['n_beats'] >= 10) & (DF['duration'] >= 60)].copy()

# Only keep sim_healthy3, sim_elderly3, sim_severe3
DF = DF[DF['patient'].isin(['sim_healthy3', 'sim_elderly3', 'sim_severe3'])].copy()

# Normalize ln(RMSSD) within-patient
DF['ln_rmssd_z'] = DF.groupby('patient')['ln_rmssd'].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0)

# Assign health status
DF['status'] = DF['patient'].apply(lambda p: 'Healthy' if 'healthy' in p else ('Elderly' if 'elderly' in p else ('Severe' if 'severe' in p else 'Unknown')))

# Plot: z-lnRMSSD by activity, colored by status
plt.figure(figsize=(12,6))
sns.boxplot(x='activity', y='ln_rmssd_z', hue='status', data=DF, showfliers=False)
sns.stripplot(x='activity', y='ln_rmssd_z', hue='patient', data=DF, dodge=True, marker='o', alpha=0.6, linewidth=0.5, edgecolor='gray', legend=False)
plt.title('z-scored ln(RMSSD) by Activity Bout for parsingsim3 (all conditions)')
plt.ylabel('z-scored ln(RMSSD)')
plt.xlabel('Activity Bout')
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('rmssd_z_by_activity_parsingsim3.png', dpi=150)
plt.close()

# Per-patient/activity summary
summary = DF.groupby(['patient','status','activity']).agg({
    'ln_rmssd_z':['mean','std','count'],
    'mean_hr':'mean',
    'duration':'mean'
}).reset_index()
summary.columns = ['patient','status','activity','z_lnrmssd_mean','z_lnrmssd_std','n_bouts','mean_hr','duration']
summary.to_csv('rmssd_parsingsim3_per_activity_summary.csv', index=False)

print('Visualization saved as rmssd_z_by_activity_parsingsim3.png')
print('Per-patient/activity summary saved as rmssd_parsingsim3_per_activity_summary.csv')
print('\nPer-patient/activity summary:')
print(summary.to_string(index=False))
