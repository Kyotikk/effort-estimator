#!/usr/bin/env python3
"""
Example script demonstrating feature extraction from sim_elderly3 sensors.

This script loads example sensor data and extracts features to verify the pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.ppg_features import PPGFeatureExtractor, load_corsano_ppg
from features.imu_features import IMUFeatureExtractor, load_corsano_acc
from features.eda_features import EDAFeatureExtractor, load_corsano_eda


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_ppg_features():
    """
    Example: Extract PPG features (heart rate level only).
    """
    print("\n" + "="*60)
    print("PPG FEATURE EXTRACTION EXAMPLE")
    print("="*60)
    
    ppg_file = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_ppg2_green_6/2025-12-04.csv.gz"
    
    print(f"\nLoading PPG from: {ppg_file}")
    
    try:
        df = load_corsano_ppg(ppg_file)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Initialize extractor
        extractor = PPGFeatureExtractor(sampling_rate=64.0)
        
        # Take first 5 minutes of data as example
        sample_size = min(len(df), 64 * 60 * 5)  # 5 minutes
        df_sample = df.iloc[:sample_size]
        
        print(f"\nExtracting features from {len(df_sample)} samples ({len(df_sample)/64/60:.1f} minutes)...")
        
        # Determine value column (adapt based on actual format)
        value_col = 'value' if 'value' in df.columns else df.columns[1]
        
        features = extractor.extract_features_from_dataframe(df_sample, value_col=value_col)
        
        print("\nExtracted PPG features (NO HRV):")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}" if not np.isnan(value) else f"  {key}: NaN")
        
        print("\n✓ PPG feature extraction successful!")
        return features
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {ppg_file}")
        print("Please verify the data path or skip this example.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_imu_features():
    """
    Example: Extract IMU features (movement/acceleration).
    """
    print("\n" + "="*60)
    print("IMU FEATURE EXTRACTION EXAMPLE")
    print("="*60)
    
    acc_file = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_wrist_acc/2025-12-04.csv.gz"
    
    print(f"\nLoading accelerometer from: {acc_file}")
    
    try:
        df = load_corsano_acc(acc_file)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Initialize extractor
        extractor = IMUFeatureExtractor(sampling_rate=50.0)
        
        # Take first 5 minutes as example
        sample_size = min(len(df), 50 * 60 * 5)
        df_sample = df.iloc[:sample_size]
        
        print(f"\nExtracting features from {len(df_sample)} samples ({len(df_sample)/50/60:.1f} minutes)...")
        
        # Determine accelerometer columns (adapt based on actual format)
        acc_cols = ('x', 'y', 'z')
        if 'x' not in df.columns:
            # Try alternative column names
            possible_names = [
                ('acc_x', 'acc_y', 'acc_z'),
                ('accel_x', 'accel_y', 'accel_z'),
                ('accX', 'accY', 'accZ'),  # Corsano format
                (df.columns[1], df.columns[2], df.columns[3])
            ]
            for names in possible_names:
                if all(col in df.columns for col in names):
                    acc_cols = names
                    break
        
        # Convert to numeric (handle string columns)
        for col in acc_cols:
            if col in df_sample.columns:
                df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
        
        features = extractor.extract_features_from_dataframe(df_sample, acc_cols=acc_cols)
        
        print("\nExtracted IMU features:")
        for key, value in features.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: NaN")
        
        print("\n✓ IMU feature extraction successful!")
        return features
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {acc_file}")
        print("Please verify the data path or skip this example.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_eda_features():
    """
    Example: Extract EDA features (sympathetic arousal).
    """
    print("\n" + "="*60)
    print("EDA FEATURE EXTRACTION EXAMPLE")
    print("="*60)
    
    eda_file = "/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/corsano_bioz_emography/2025-12-04.csv.gz"
    
    print(f"\nLoading EDA from: {eda_file}")
    
    try:
        df = load_corsano_eda(eda_file)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Initialize extractor (EDA typically low sampling rate)
        extractor = EDAFeatureExtractor(sampling_rate=4.0)
        
        # Take first 5 minutes as example
        sample_size = min(len(df), 4 * 60 * 5)
        df_sample = df.iloc[:sample_size]
        
        print(f"\nExtracting features from {len(df_sample)} samples ({len(df_sample)/4/60:.1f} minutes)...")
        
        # Determine value column - check what's available for EDA
        # The bioz_emography file has stress_skin which might be EDA-related
        value_col = None
        for col in ['value', 'stress_skin', 'cz', 'eda', 'gsr', 'skin_conductance']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None:
            print(f"WARNING: Could not find EDA column. Available: {df.columns.tolist()}")
            print("Skipping EDA extraction - please specify correct column name.")
            return None
        
        print(f"Using column '{value_col}' for EDA data")
        
        # Convert to numeric
        df_sample[value_col] = pd.to_numeric(df_sample[value_col], errors='coerce')
        df_sample = df_sample.dropna(subset=[value_col])
        
        if len(df_sample) < 10:
            print(f"WARNING: Only {len(df_sample)} valid samples after cleaning. Need more data.")
            return None
        
        features = extractor.extract_features_from_dataframe(df_sample, value_col=value_col)
        
        print("\nExtracted EDA features:")
        for key, value in features.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: NaN")
        
        print("\n✓ EDA feature extraction successful!")
        return features
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {eda_file}")
        print("Please verify the data path or skip this example.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all feature extraction examples."""
    setup_logging()
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("Testing on sim_elderly3 data")
    print("="*60)
    
    print("\nNOTE: This script tests feature extraction modules.")
    print("If data files are not found, column names will need adjustment.")
    print("The extractors are ready to use once data paths are confirmed.")
    
    # Run examples
    ppg_feats = example_ppg_features()
    imu_feats = example_imu_features()
    eda_feats = example_eda_features()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum([
        ppg_feats is not None,
        imu_feats is not None,
        eda_feats is not None
    ])
    
    print(f"\nSuccessfully extracted features from {success_count}/3 modalities")
    
    if success_count == 3:
        print("\n✓ All feature extractors working correctly!")
        print("\nCRITICAL REMINDER:")
        print("  - NO HRV features in inputs (RMSSD, SDNN, etc.)")
        print("  - PPG features are HR level only")
        print("  - ECG/RMSSD used ONLY for effort labels")
    else:
        print("\nSome extractors encountered errors.")
        print("This is expected if data files don't exist yet.")
        print("Please verify data paths and column names in the actual CSV files.")


if __name__ == '__main__':
    main()
