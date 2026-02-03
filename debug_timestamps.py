#!/usr/bin/env python3
import pandas as pd
from datetime import datetime

# Check fused timestamps
fused = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_features_5.0s.csv')
print(f'Fused t_center range: {fused.t_center.min():.0f} - {fused.t_center.max():.0f}')
print(f'  As datetime: {datetime.fromtimestamp(fused.t_center.min())} - {datetime.fromtimestamp(fused.t_center.max())}')

# Check ADL timestamps
adl_raw = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv', skiprows=2)
adl_raw.columns = ['Time', 'ADLs', 'Effort']

def parse_time(t):
    try:
        dt = datetime.strptime(t, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None

adl_raw['t_center'] = adl_raw['Time'].apply(parse_time)
adl = adl_raw[adl_raw['Effort'].notna()].dropna(subset=['t_center'])
print(f'ADL t_center range: {adl.t_center.min():.0f} - {adl.t_center.max():.0f}')
print(f'  As datetime: {datetime.fromtimestamp(adl.t_center.min())} - {datetime.fromtimestamp(adl.t_center.max())}')
