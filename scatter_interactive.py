import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
from pathlib import Path

# Load the actual data - same as ml_results_expert.py
print("Loading data...")
dfs = []
for i in range(1, 6):
    path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
    if path.exists():
        tmp = pd.read_csv(path)
        tmp['subject'] = f'P{i}'
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['borg'])

imu_cols = [c for c in df.columns if 'acc_' in c and '_r' not in c]
print(f"Data: {len(df)} windows, {df['subject'].nunique()} subjects, {len(imu_cols)} IMU features")

# Run LOSO
from sklearn.ensemble import RandomForestRegressor
subjects = sorted(df['subject'].unique())
results = []

for test_subj in subjects:
    train = df[df['subject'] != test_subj]
    test = df[df['subject'] == test_subj].copy()
    
    X_train = train[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train['borg'].values
    X_test = test[imu_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test['borg'].values
    
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    for i in range(len(test)):
        results.append({
            'subject': test_subj,
            'y_true': y_test[i],
            'y_pred': y_pred[i]
        })

results = pd.DataFrame(results)

# Create interactive scatter
fig = go.Figure()

colors = {'P1': '#e74c3c', 'P2': '#3498db', 'P3': '#2ecc71', 'P4': '#9b59b6', 'P5': '#f39c12'}

print("Per-subject statistics:")
print("-" * 50)

for subj in sorted(results['subject'].unique()):
    mask = results['subject'] == subj
    x_data = results.loc[mask, 'y_true'].values
    y_data = results.loc[mask, 'y_pred'].values
    r, _ = pearsonr(x_data, y_data)
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        name=f'{subj} (r={r:.2f})',
        marker=dict(color=colors[subj], size=8, opacity=0.6),
        hovertemplate=f'{subj}<br>Actual: %{{x:.1f}}<br>Pred: %{{y:.1f}}<extra></extra>'
    ))
    
    # Trend line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_line = np.array([x_data.min(), x_data.max()])
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode='lines',
        name=f'{subj} trend (slope={z[0]:.2f})',
        line=dict(color=colors[subj], width=3),
        showlegend=True
    ))
    
    print(f"{subj}: r={r:.2f}, slope={z[0]:.2f}, intercept={z[1]:.2f}")

# Diagonal
fig.add_trace(go.Scatter(x=[0,7], y=[0,7], mode='lines', name='Perfect (slope=1)',
                         line=dict(color='black', dash='dash', width=2)))

fig.update_layout(
    title='Interactive: Predicted vs Actual Borg (Hover for details)',
    xaxis_title='Actual Borg CR10',
    yaxis_title='Predicted Borg CR10',
    xaxis=dict(range=[-0.5, 7]),
    yaxis=dict(range=[-0.5, 7]),
    width=800, height=700
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.write_html('/Users/pascalschlegel/effort-estimator/thesis_plots_final/scatter_interactive.html')
print("\nSaved interactive plot!")
