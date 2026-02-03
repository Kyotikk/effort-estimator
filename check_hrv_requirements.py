#!/usr/bin/env python3
"""Check HRV requirements and beat counts."""

# RMSSD needs at least 2 consecutive RR intervals (3 heartbeats)
# SDNN needs variance (at least 2 values, so 3 heartbeats minimum)
# pNN50 needs enough beats to have meaningful percentage

print('=== Beats per window at different heart rates ===\n')

print('5s window:')
for hr in [40, 50, 60, 70, 80, 100, 120]:
    beats = hr/60 * 5
    print(f'  HR={hr} BPM -> {beats:.1f} beats')

print('\n10s window:')
for hr in [40, 50, 60, 80]:
    beats = hr/60 * 10
    print(f'  HR={hr} BPM -> {beats:.1f} beats')

print('\n30s window:')
for hr in [40, 50, 60, 80]:
    beats = hr/60 * 30
    print(f'  HR={hr} BPM -> {beats:.1f} beats')

print('\n=== HRV Metric Requirements ===')
print('RMSSD: minimum 2 RR intervals (3 beats), recommended 10+ beats')
print('SDNN: minimum 2 RR intervals (3 beats), recommended 30+ beats for stability')
print('pNN50: needs enough beats for percentage to be meaningful (10+ recommended)')
print('\nTask Force (1996) recommends: minimum 5 minutes for frequency domain HRV')
print('Ultra-short HRV (<1 min) is controversial but used in practice')
