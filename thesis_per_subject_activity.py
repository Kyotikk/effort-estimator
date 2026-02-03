#!/usr/bin/env python3
"""
Per-subject actual vs predicted Borg with activity labels.
Using LOSO prediction - train on other subjects, predict on held-out subject.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('output/tli_all_subjects.csv')
df = df[df['borg'].notna()].copy()

# Features available: imu_load, rms_acc_mag, rms_jerk, hr_mean, hr_max, hr_delta, hr_load
# Using IMU features only for this viz
feature_cols = ['rms_acc_mag', 'imu_load', 'rms_jerk']

subjects = sorted(df['subject'].unique())
print(f"Subjects: {subjects}")
print(f"Total samples: {len(df)}")

# LOSO predictions
all_preds = []
for test_subj in subjects:
    train = df[df['subject'] != test_subj]
    test = df[df['subject'] == test_subj]
    
    X_train = train[feature_cols].values
    y_train = train['borg'].values
    X_test = test[feature_cols].values
    y_test = test['borg'].values
    
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    for i, (idx, row) in enumerate(test.iterrows()):
        all_preds.append({
            'subject': test_subj,
            'activity': row['activity'],
            'borg_actual': row['borg'],
            'borg_pred': preds[i],
            'imu_load': row['imu_load'],
            'rms_acc_mag': row['rms_acc_mag'],
            't_start': row['t_start'],
            't_end': row['t_end']
        })

results = pd.DataFrame(all_preds)

# Overall LOSO correlation
r_overall, _ = pearsonr(results['borg_actual'], results['borg_pred'])
print(f"\nOverall LOSO r = {r_overall:.3f}")

# Per-subject correlation
print("\nPer-subject r:")
for subj in subjects:
    sub = results[results['subject'] == subj]
    if len(sub) > 2:
        r, _ = pearsonr(sub['borg_actual'], sub['borg_pred'])
        print(f"  {subj}: r = {r:.3f}")

# Create figure - one subplot per subject
fig, axes = plt.subplots(len(subjects), 1, figsize=(16, 4*len(subjects)))

# Colors for dynamic/static activities
def get_activity_color(activity):
    static = ['Stand', 'Resting', 'Stand to Sit', 'Sit to Stand', 'Using Phone', 'Eating/Drinking']
    if any(s in activity for s in static):
        return '#E8F5E9', '#4CAF50'  # green for static
    else:
        return '#E3F2FD', '#2196F3'  # blue for dynamic

for ax_idx, subj in enumerate(subjects):
    ax = axes[ax_idx]
    sub = results[results['subject'] == subj].copy()
    sub = sub.sort_values('t_start').reset_index(drop=True)
    
    n = len(sub)
    x = np.arange(n)
    
    # Plot activity backgrounds
    for i, (_, row) in enumerate(sub.iterrows()):
        bg_color, edge_color = get_activity_color(row['activity'])
        ax.axvspan(i-0.4, i+0.4, color=bg_color, alpha=0.5, zorder=0)
    
    # Plot actual and predicted
    ax.plot(x, sub['borg_actual'], 'ko-', linewidth=2, markersize=8, label='Actual Borg', zorder=3)
    ax.plot(x, sub['borg_pred'], 's-', color='#2196F3', linewidth=1.5, markersize=6, label='Predicted', zorder=2)
    
    # Fill between
    ax.fill_between(x, sub['borg_actual'], sub['borg_pred'], alpha=0.2, color='#2196F3')
    
    # Calculate r for this subject
    if len(sub) > 2:
        r, _ = pearsonr(sub['borg_actual'], sub['borg_pred'])
        r_text = f"r = {r:.2f}"
    else:
        r_text = "n/a"
    
    # Activity labels on x-axis (shortened)
    activity_labels = []
    for act in sub['activity']:
        # Shorten long names
        short = act.replace('Transfer ', 'T.').replace('Lower/Raise ', 'L/R ')
        short = short.replace('Indoor Activity', 'Indoor').replace('Eating/Drinking', 'Eat/Drink')
        short = short.replace('Style Beard/Hair', 'Style').replace('Put Toothpaste', 'Toothpaste')
        short = short.replace('(Un)button Shirt', 'Button').replace('Arm in Sleeves', 'Sleeves')
        short = short.replace('Fold Clothes', 'Fold').replace('Put on/off Clothes', 'Clothes')
        short = short.replace('Self Propulsion', 'Propulsion').replace('Level Walking', 'Walk')
        short = short.replace('Transfer from Toilet', 'T.fr.Toilet').replace('Transfer to Toilet', 'T.to.Toilet')
        short = short.replace('Transfer to Bed', 'T.to.Bed').replace('Transfer from Bed', 'T.fr.Bed')
        activity_labels.append(short[:12])
    
    ax.set_xticks(x)
    ax.set_xticklabels(activity_labels, rotation=45, ha='right', fontsize=7)
    
    ax.set_ylabel('Borg CR10 (0-10)', fontsize=10)
    ax.set_ylim(-0.5, 10)
    ax.set_xlim(-0.5, n-0.5)
    ax.set_title(f'{subj.replace("sim_", "").title()} ({n} activities) — {r_text}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight standing activities
    for i, (_, row) in enumerate(sub.iterrows()):
        if row['activity'] == 'Stand':
            ax.annotate('STAND', (i, row['borg_actual']+0.5), ha='center', fontsize=8, 
                       color='red', fontweight='bold')

plt.suptitle(f'LOSO IMU Model: Actual vs Predicted Borg per Subject\n(Overall r = {r_overall:.2f})', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('data/feature_extraction/analysis/63_per_subject_activity_loso.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Saved: 63_per_subject_activity_loso.png")
plt.show()
