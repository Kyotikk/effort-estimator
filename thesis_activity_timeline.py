#!/usr/bin/env python3
"""
Visualize Actual vs Predicted Borg with Activity Labels
Shows where each activity starts/ends along the timeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load the TLI data with activities
df = pd.read_csv('output/tli_all_subjects.csv')

# Pick one subject for visualization
subject = 'sim_elderly3'
sub = df[df['subject'] == subject].sort_values('t_start').reset_index(drop=True)

# Normalize time to start from 0 (in minutes)
t_min = sub['t_start'].min()
sub['t_start_min'] = (sub['t_start'] - t_min) / 60
sub['t_end_min'] = (sub['t_end'] - t_min) / 60
sub['t_center_min'] = (sub['t_start_min'] + sub['t_end_min']) / 2

# Features for prediction (what's available)
feature_cols = ['rms_acc_mag', 'rms_jerk', 'hr_mean', 'hr_delta', 'duration_s']
available_features = [c for c in feature_cols if c in sub.columns and sub[c].notna().any()]

print(f"Using features: {available_features}")
print(f"Activities: {len(sub)}")

# Simple prediction using available features (cross-validation style)
X = sub[available_features].fillna(0).values
y = sub['borg'].values

# Leave-one-out prediction for this subject
predictions = []
for i in range(len(sub)):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X[i:i+1]
    
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    predictions.append(pred)

sub['predicted'] = predictions

# Calculate correlation
r, p = pearsonr(sub['borg'], sub['predicted'])
print(f"Correlation: r = {r:.2f}")

# ============================================
# PLOT
# ============================================
fig, ax = plt.subplots(figsize=(16, 6))

# Color map for activities
activities = sub['activity'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(activities)))
activity_colors = {act: colors[i] for i, act in enumerate(activities)}

# Plot activity regions as background
for i, row in sub.iterrows():
    ax.axvspan(row['t_start_min'], row['t_end_min'], 
               alpha=0.2, color=activity_colors[row['activity']])

# Plot actual Borg (step function for activities)
for i, row in sub.iterrows():
    ax.hlines(row['borg'], row['t_start_min'], row['t_end_min'], 
              colors='black', linewidth=2.5, label='Actual Borg' if i == 0 else '')

# Plot predicted Borg
ax.plot(sub['t_center_min'], sub['predicted'], 'o-', color='#3182bd', 
        linewidth=1.5, markersize=6, alpha=0.8, label='Predicted Borg')

# Fill between actual and predicted
for i, row in sub.iterrows():
    ax.fill_between([row['t_start_min'], row['t_end_min']], 
                    [row['borg'], row['borg']], 
                    [row['predicted'], row['predicted']], 
                    alpha=0.15, color='#3182bd')

# Add activity labels at top (rotated)
y_label = 6.8
for i, row in sub.iterrows():
    # Shorten some activity names
    act_short = row['activity'].replace('Transfer ', 'Xfer ').replace(' (right)', ' R').replace(' (left)', ' L')
    if len(act_short) > 12:
        act_short = act_short[:11] + '.'
    
    t_mid = (row['t_start_min'] + row['t_end_min']) / 2
    
    # Only label if activity is wide enough
    if (row['t_end_min'] - row['t_start_min']) > 0.3:
        ax.text(t_mid, y_label, act_short, rotation=45, ha='left', va='bottom',
                fontsize=7, color='#333333')
        ax.axvline(row['t_start_min'], color='gray', linestyle=':', alpha=0.4, linewidth=0.5)

# Formatting
ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Borg CR10 (0-10)', fontsize=12)
ax.set_title(f'Actual vs Predicted Borg with Activity Labels — {subject}\n(r = {r:.2f})', fontsize=14)
ax.set_ylim(-0.5, 8)
ax.set_xlim(sub['t_start_min'].min() - 0.5, sub['t_end_min'].max() + 0.5)

# Legend
ax.legend(loc='upper left', fontsize=10)

# Grid
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('data/feature_extraction/analysis/63_activity_timeline.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 63_activity_timeline.png")
plt.show()
