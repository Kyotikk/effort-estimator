#!/usr/bin/env python3
"""
Check raw data for:
1. Actual activity labels
2. Raw PPG signal for better HR extraction
"""
import pandas as pd
from pathlib import Path
import numpy as np

print("="*70)
print("1. CHECKING FOR ACTIVITY LABELS")
print("="*70)

# Check the aligned file columns
aligned = pd.read_csv('/Users/pascalschlegel/data/interim/elderly_combined/elderly_aligned_5.0s.csv', nrows=100)
print("\nAll columns in aligned file:")
print(aligned.columns.tolist()[:30])

act_cols = [c for c in aligned.columns if any(x in c.lower() for x in ['act', 'task', 'label', 'bout', 'proto'])]
print(f"\nActivity-related columns: {act_cols}")

# Check raw data folders
print("\n" + "="*70)
print("2. RAW DATA STRUCTURE")
print("="*70)

raw_base = Path('/Users/pascalschlegel/data/raw')
if raw_base.exists():
    print(f"\nRaw data folder contents:")
    for item in sorted(raw_base.iterdir())[:10]:
        print(f"  {item.name}")
        if item.is_dir():
            for sub in sorted(item.iterdir())[:5]:
                print(f"    {sub.name}")

# Look for elderly data
elderly_paths = list(Path('/Users/pascalschlegel/data').rglob('sim_elderly*'))
print(f"\nElderly data paths found: {len(elderly_paths)}")
for p in sorted(set([str(p.parent) for p in elderly_paths]))[:5]:
    print(f"  {p}")

# Check one raw file for PPG
print("\n" + "="*70)
print("3. RAW PPG DATA CHECK")
print("="*70)

ppg_files = list(Path('/Users/pascalschlegel/data').rglob('*ppg*.csv'))
print(f"\nPPG files found: {len(ppg_files)}")
for f in ppg_files[:5]:
    print(f"  {f}")
    try:
        df = pd.read_csv(f, nrows=10)
        print(f"    Columns: {df.columns.tolist()[:5]}")
        print(f"    Shape: {df.shape}")
    except Exception as e:
        print(f"    Error: {e}")

# Check preprocessing folder
print("\n" + "="*70)
print("4. PREPROCESSED DATA CHECK")
print("="*70)

preproc_paths = list(Path('/Users/pascalschlegel/data/interim').rglob('*elderly*'))
print(f"\nPreprocessed elderly files: {len(preproc_paths)}")
for p in sorted(preproc_paths)[:10]:
    print(f"  {p}")

# Check for ADL/activity files
print("\n" + "="*70)
print("5. ADL/ACTIVITY LABEL FILES")
print("="*70)

adl_files = list(Path('/Users/pascalschlegel/data').rglob('*adl*'))
adl_files += list(Path('/Users/pascalschlegel/data').rglob('*borg*'))
adl_files += list(Path('/Users/pascalschlegel/data').rglob('*effort*'))
print(f"\nADL/Borg files found: {len(adl_files)}")
for f in sorted(set(adl_files))[:15]:
    print(f"  {f}")
    if f.suffix == '.csv':
        try:
            df = pd.read_csv(f, nrows=5)
            print(f"    Columns: {df.columns.tolist()[:8]}")
        except:
            pass

# Check one ADL file in detail
print("\n" + "="*70)
print("6. DETAILED ADL FILE CHECK")
print("="*70)

adl_csvs = [f for f in adl_files if f.suffix == '.csv' and 'elderly' in str(f).lower()]
if adl_csvs:
    f = adl_csvs[0]
    print(f"\nChecking: {f}")
    df = pd.read_csv(f)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
