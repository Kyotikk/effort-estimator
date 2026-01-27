import pandas as pd
import matplotlib.pyplot as plt

# Parameters: patient and activity to audit

# Parameters: patient and activity to audit
PATIENT = 'sim_healthy4'
ACTIVITY = 'Level Walking'
START = 1764899361.086
END = 1764899427.273

# Load RR intervals for this patient
rr_path = f"/Users/pascalschlegel/data/interim/parsingsim4/{PATIENT}/effort_estimation_output/parsingsim4_{PATIENT}/rr/rr_preprocessed.csv"
df_rr = pd.read_csv(rr_path)

# Load ADL annotations
adl_path = f"/Users/pascalschlegel/data/interim/parsingsim4/{PATIENT}/scai_app/ADLs_1.csv.gz"
df_adl = pd.read_csv(adl_path)

# Restrict to bout window
rr_bout = df_rr[(df_rr['time'] >= START) & (df_rr['time'] <= END)]
adl_bout = df_adl[(df_adl['time'] >= START) & (df_adl['time'] <= END)]

# Plot RR intervals and activity label
plt.figure(figsize=(12,5))
plt.plot(rr_bout['time'], rr_bout['rr'], label='RR interval (ms)')
for _, row in adl_bout.iterrows():
    plt.axvspan(row['time'], row['time']+row['duration'], color='orange', alpha=0.2)
plt.title(f'Raw RR intervals and activity\n{PATIENT}, {ACTIVITY}, bout {START:.0f}-{END:.0f}')
plt.xlabel('Time (s)')
plt.ylabel('RR interval (ms)')
plt.legend()
plt.tight_layout()
plt.savefig(f'raw_rr_and_activity_{PATIENT}_{ACTIVITY.replace(" ", "_")}.png', dpi=150)
plt.close()

print(f'Raw RR and activity plot saved as raw_rr_and_activity_{PATIENT}_{ACTIVITY.replace(" ", "_")}.png')
