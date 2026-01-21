"""
Create synthetic test data to validate HRV recovery pipeline modules.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path('./test_data_hrv')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# SYNTHETIC PPG DATA
# ============================================================================
# Simulate 10 minutes of PPG at 32 Hz with realistic heart rate
fs = 32.0
duration_sec = 600
n_samples = int(fs * duration_sec)
t = np.arange(n_samples) / fs

# HR varies: baseline ~70 bpm, spike to 120 bpm around 300s
hr_baseline = 70
hr_peak = 120
spike_start = 300
spike_end = 360

# Create HR vector
hr = np.ones_like(t) * hr_baseline
spike_mask = (t >= spike_start) & (t < spike_end)
hr[spike_mask] = np.linspace(hr_baseline, hr_peak, spike_mask.sum())
hr[(t >= spike_end)] = np.linspace(hr_peak, hr_baseline, (t >= spike_end).sum())

# Convert HR to IBI (beat intervals)
ibi_mean = 60.0 / hr  # seconds
beat_times = [0]
while beat_times[-1] < duration_sec:
    next_ibi = np.random.normal(ibi_mean[int(beat_times[-1] * fs)], 0.02)
    next_beat = beat_times[-1] + next_ibi
    if next_beat < duration_sec:
        beat_times.append(next_beat)

# Create synthetic PPG signal: sum of sinusoids at beat times
ppg_signal = np.zeros(n_samples)
for beat_time in beat_times:
    beat_idx = int(beat_time * fs)
    # Gaussian pulse centered at beat time
    if beat_idx < n_samples:
        pulse = np.exp(-((np.arange(n_samples) - beat_idx) ** 2) / (0.5 * fs) ** 2)
        ppg_signal += 1000 * pulse

# Add noise
ppg_signal += np.random.normal(0, 50, n_samples)

ppg_df = pd.DataFrame({
    'time': t,
    'value': ppg_signal
})
ppg_path = output_dir / 'ppg_synthetic.csv'
ppg_df.to_csv(ppg_path, index=False)
print(f"✓ Created synthetic PPG: {ppg_path} ({len(ppg_df)} samples)")

# ============================================================================
# SYNTHETIC ADL DATA
# ============================================================================
adl_intervals = [
    {'t_start': 60.0, 't_end': 120.0, 'task_name': 'walk'},
    {'t_start': 150.0, 't_end': 200.0, 'task_name': 'climb'},
    {'t_start': 300.0, 't_end': 360.0, 'task_name': 'transfer'},  # High HR
    {'t_start': 420.0, 't_end': 480.0, 'task_name': 'walk'},
]

adl_df = pd.DataFrame(adl_intervals)
adl_path = output_dir / 'adl_synthetic.csv'
adl_df.to_csv(adl_path, index=False)
print(f"✓ Created synthetic ADL: {adl_path} ({len(adl_df)} intervals)")

# ============================================================================
# SYNTHETIC IMU FEATURES (per 10s windows)
# ============================================================================
window_len = 10.0
n_windows = duration_sec // window_len

imu_features = []
for i in range(n_windows):
    t_center = (i + 0.5) * window_len
    
    # Intensity peaks during effort
    intensity = 10.0  # baseline
    for adl in adl_intervals:
        if adl['t_start'] <= t_center <= adl['t_end']:
            intensity = 50.0  # active
            break
    
    intensity += np.random.normal(0, 5)
    
    imu_features.append({
        't_start': i * window_len,
        't_center': t_center,
        't_end': (i + 1) * window_len,
        'acc_mag_rms': intensity,
        'acc_mag_mean': intensity * 0.8,
        'acc_mag_std': intensity * 0.2,
    })

imu_df = pd.DataFrame(imu_features)
imu_path = output_dir / 'imu_synthetic.csv'
imu_df.to_csv(imu_path, index=False)
print(f"✓ Created synthetic IMU features: {imu_path} ({len(imu_df)} windows)")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n✓ Synthetic test data created in {output_dir}")
print(f"  - ppg_synthetic.csv: {len(ppg_df)} samples @ {fs} Hz")
print(f"  - adl_synthetic.csv: {len(adl_df)} effort intervals")
print(f"  - imu_synthetic.csv: {len(imu_df)} feature windows")
print(f"\nUpdate config_hrv_test.yaml and run:")
print(f"  .venv/bin/python run_hrv_pipeline.py config/config_hrv_test.yaml")
