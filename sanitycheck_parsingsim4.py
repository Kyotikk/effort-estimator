import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
DF = pd.read_csv('rmssd_per_bout.csv')
DF = DF[(DF['n_beats'] >= 10) & (DF['duration'] >= 60)].copy()

# Pick 3 key activities for sanity check
key_activities = ['Stand', 'Level Walking', 'Turn Over (left)']
DF = DF[DF['activity'].isin(key_activities)]

# Only keep sim_healthy4, sim_elderly4, sim_severe4
DF = DF[DF['patient'].isin(['sim_healthy4', 'sim_elderly4', 'sim_severe4'])].copy()

# Plot raw ln(RMSSD) per bout, colored by patient, with bout duration as marker size
plt.figure(figsize=(10,6))
for patient, color in zip(['sim_healthy4', 'sim_elderly4', 'sim_severe4'], ['orange', 'blue', 'green']):
    sub = DF[DF['patient'] == patient]
    plt.scatter(sub['activity'], sub['ln_rmssd'], s=40 + 200 * (sub['duration'] - DF['duration'].min()) / (DF['duration'].max() - DF['duration'].min()),
                color=color, alpha=0.7, label=patient)
plt.title('Sanity Check: ln(RMSSD) by Activity (parsingsim4)\nMarker size = bout duration (s)')
plt.ylabel('ln(RMSSD)')
plt.xlabel('Activity')
plt.legend(title='Patient')
plt.tight_layout()
plt.savefig('sanitycheck_lnrmssd_by_activity_parsingsim4.png', dpi=150)
plt.close()

# Print summary table
summary = DF.groupby(['patient','activity']).agg({
    'ln_rmssd':['mean','std','count'],
    'mean_hr':'mean',
    'duration':'mean'
}).reset_index()
summary.columns = ['patient','activity','lnrmssd_mean','lnrmssd_std','n_bouts','mean_hr','duration']
summary.to_csv('sanitycheck_parsingsim4_summary.csv', index=False)

print('Sanity check plot saved as sanitycheck_lnrmssd_by_activity_parsingsim4.png')
print('Summary saved as sanitycheck_parsingsim4_summary.csv')
print('\nSummary:')
print(summary.to_string(index=False))
