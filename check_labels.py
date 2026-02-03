#!/usr/bin/env python3
"""Check label column and raw data for activity labels"""
import pandas as pd
from pathlib import Path

# Check label column in aligned file
df = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv')
print('='*70)
print('LABEL COLUMN IN ALIGNED FILE')
print('='*70)
print(f'Unique labels: {df["label"].nunique()}')
print(f'Label values: {sorted(df["label"].dropna().unique())}')
print()
print('Label distribution:')
print(df['label'].value_counts())

# Check raw ADL file
print('\n' + '='*70)
print('RAW ADL FILES')
print('='*70)

raw_path = Path('/Users/pascalschlegel/data/raw/parsingsim1/sim_elderly1_formatted')
if raw_path.exists():
    print(f'\nContents of {raw_path}:')
    for f in sorted(raw_path.iterdir()):
        print(f'  {f.name}')
    
    # Check ADL file
    adl_file = raw_path / 'adl.csv'
    if adl_file.exists():
        print(f'\nADL FILE CONTENTS:')
        adl = pd.read_csv(adl_file)
        print(f'Shape: {adl.shape}')
        print(f'Columns: {adl.columns.tolist()}')
        print(adl.head(20))

# Check raw PPG
print('\n' + '='*70)
print('RAW PPG SIGNAL')
print('='*70)

ppg_preprocessed = Path('/Users/pascalschlegel/data/interim/parsingsim1/sim_elderly1/effort_estimation_output/elderly_sim_elderly1/ppg_green/ppg_green_preprocessed.csv')
if ppg_preprocessed.exists():
    ppg = pd.read_csv(ppg_preprocessed)
    print(f'PPG preprocessed shape: {ppg.shape}')
    print(f'Columns: {ppg.columns.tolist()}')
    print(f'Sample rate: {1/(ppg["t_sec"].diff().median()):.1f} Hz')
    print(ppg.head())
