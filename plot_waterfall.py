#!/usr/bin/env python3
"""Generate waterfall plot of feature importance."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

OUTPUT_DIR = Path('/Users/pascalschlegel/data/interim/elderly_combined')
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Load feature importance
importance_df = pd.read_csv(OUTPUT_DIR / 'xgboost_results_5.0s' / 'feature_importance.csv')
top_20 = importance_df.nlargest(20, 'importance').sort_values('importance', ascending=True)

# Create waterfall-style horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 10))

# Color by modality
colors = []
for f in top_20['feature']:
    if 'ppg' in f.lower():
        colors.append('#e74c3c')
    elif 'eda' in f.lower():
        colors.append('#3498db')
    elif 'acc' in f.lower():
        colors.append('#2ecc71')
    else:
        colors.append('#95a5a6')

bars = ax.barh(range(len(top_20)), top_20['importance'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_20['importance'])):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'], fontsize=10)
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('XGBoost Feature Importance Waterfall (Top 20)', fontsize=14, fontweight='bold')

# Add cumulative line on secondary axis
cumsum = top_20['importance'].cumsum().values
ax2 = ax.twiny()
ax2.plot(cumsum, range(len(top_20)), 'ko-', markersize=5, linewidth=2, alpha=0.7)
ax2.set_xlabel('Cumulative Importance', fontsize=11, color='black')
ax2.set_xlim(0, cumsum[-1] * 1.1)

# Legend
legend_elements = [
    Patch(facecolor='#e74c3c', label='PPG'),
    Patch(facecolor='#3498db', label='EDA'),
    Patch(facecolor='#2ecc71', label='IMU'),
    plt.Line2D([0], [0], color='black', marker='o', label='Cumulative')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'feature_importance_waterfall.png', dpi=150, bbox_inches='tight')
print(f'Saved: {PLOTS_DIR}/feature_importance_waterfall.png')

# Stats
total = importance_df['importance'].sum()
print(f'Top 20 features account for {top_20["importance"].sum()/total*100:.1f}% of total importance')
