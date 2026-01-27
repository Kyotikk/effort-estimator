#!/usr/bin/env python3
import pandas as pd
import datetime

eda = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_emography/2025-12-04.csv.gz')
print('EDA samples:', len(eda))
print('EDA time range:', eda['time'].min(), '-', eda['time'].max())
print('EDA start:', datetime.datetime.fromtimestamp(eda['time'].min()))
print('EDA end:', datetime.datetime.fromtimestamp(eda['time'].max()))
print()
print('Duration (sec):', eda['time'].max() - eda['time'].min())
print('Sample interval:', (eda['time'].max() - eda['time'].min()) / len(eda), 'sec')

# Check ADL shifted time
adl_offset = -8.3 * 3600
adl_start = 1764866782.089 + adl_offset
adl_end = 1764868756.646 + adl_offset
print()
print('ADL shifted start:', datetime.datetime.fromtimestamp(adl_start))
print('ADL shifted end:', datetime.datetime.fromtimestamp(adl_end))

# Check overlap
print()
print('Does EDA overlap with ADL?', eda['time'].min() <= adl_end and eda['time'].max() >= adl_start)
