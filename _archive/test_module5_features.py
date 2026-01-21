"""
Test Module 5: Aggregate features per effort bout and build model table
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module5_features import aggregate_windowed_features, extract_hr_from_ibi

if __name__ == "__main__":
    # Load fused features from existing pipeline (10s windows)
    features_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_features_10.0s.csv")
    features_df = pd.read_csv(features_path)
    print(f"✓ Loaded windowed features: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"  Time range: {features_df['t_center'].min():.0f} - {features_df['t_center'].max():.0f}")
    
    # Apply timezone correction (+8 hours to match ADL data)
    TIMEZONE_OFFSET_SEC = 8 * 3600
    print(f"\n⚠ Applying timezone correction: +{TIMEZONE_OFFSET_SEC/3600:.0f} hours")
    features_df['t_start'] += TIMEZONE_OFFSET_SEC
    features_df['t_center'] += TIMEZONE_OFFSET_SEC
    features_df['t_end'] += TIMEZONE_OFFSET_SEC
    print(f"  Corrected time range: {features_df['t_center'].min():.0f} - {features_df['t_center'].max():.0f}")
    
    # Load effort bouts
    bouts_path = Path("./output/test_bouts_module3.csv")
    bouts_df = pd.read_csv(bouts_path)
    if 'bout_id' not in bouts_df.columns:
        bouts_df['bout_id'] = range(len(bouts_df))
    print(f"\n✓ Loaded effort bouts: {len(bouts_df)} rows")
    print(f"  Bout time range: {bouts_df['t_start'].min():.0f} - {bouts_df['t_end'].max():.0f}")
    
    # Load IBI data
    ibi_path = Path("./output/test_ibi_elderly3.csv")
    ibi_df = pd.read_csv(ibi_path)
    ibi_df['t'] += TIMEZONE_OFFSET_SEC  # Apply timezone correction
    print(f"\n✓ Loaded IBI data: {len(ibi_df)} IBIs")
    
    print("\n" + "="*70)
    print("Aggregating features per bout...")
    print("="*70)
    
    # Aggregate windowed features per bout (mean aggregation)
    bout_features = aggregate_windowed_features(
        features_df,
        bouts_df,
        time_col="t_center",
        method="mean",
        prefix=""
    )
    
    print(f"\n✓ Aggregated features: {len(bout_features)} bouts, {len(bout_features.columns)-1} features")
    
    # Add HR features from IBI
    print("\nAdding HR features from IBI data...")
    hr_features_list = []
    for _, bout in bouts_df.iterrows():
        hr_feats = extract_hr_from_ibi(ibi_df, bout['t_start'], bout['t_end'])
        hr_feats['bout_id'] = bout['bout_id']
        hr_features_list.append(hr_feats)
    
    hr_features_df = pd.DataFrame(hr_features_list)
    print(f"✓ HR features: {len(hr_features_df)} bouts, {len(hr_features_df.columns)-1} HR features")
    
    # Merge with bout features
    bout_features = bout_features.merge(hr_features_df, on='bout_id', how='left')
    
    # Load HRV recovery labels
    labels_path = Path("./output/test_labels_elderly3.csv")
    labels_df = pd.read_csv(labels_path)
    print(f"\n✓ Loaded HRV recovery labels: {len(labels_df)} bouts")
    
    # Build final model table
    print("\n" + "="*70)
    print("Building model table...")
    print("="*70)
    
    model_table = bouts_df[['bout_id', 't_start', 't_end', 'duration_sec', 'task_name']].merge(
        bout_features,
        on='bout_id',
        how='left'
    ).merge(
        labels_df[['bout_id', 'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'qc_ok']],
        on='bout_id',
        how='left'
    )
    
    # Add effort ratings if available
    if 'effort' in bouts_df.columns:
        model_table = model_table.merge(
            bouts_df[['bout_id', 'effort']],
            on='bout_id',
            how='left'
        )
    
    print(f"\n✓ Model table: {len(model_table)} rows, {len(model_table.columns)} columns")
    
    # Filter to valid labels only
    valid_table = model_table[model_table['qc_ok'] == True].copy()
    print(f"  Valid labels: {len(valid_table)} rows ({len(valid_table)/len(model_table)*100:.1f}%)")
    
    # Check feature completeness
    feature_cols = [c for c in valid_table.columns if c not in [
        'bout_id', 't_start', 't_end', 'duration_sec', 'task_name', 
        'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'qc_ok', 'effort'
    ]]
    
    print(f"\n  Feature columns: {len(feature_cols)}")
    print(f"  Features with data:")
    non_null_counts = valid_table[feature_cols].notna().sum()
    features_with_data = (non_null_counts > 0).sum()
    print(f"    {features_with_data}/{len(feature_cols)} features have non-null values")
    
    # Show sample
    print(f"\nFirst 5 rows (selected columns):")
    display_cols = ['bout_id', 'task_name', 'duration_sec', 'hr_mean', 'rmssd_during_effort', 
                    'delta_rmssd', 'effort', 'qc_ok']
    display_cols = [c for c in display_cols if c in valid_table.columns]
    print(valid_table[display_cols].head().to_string(index=False))
    
    # Save outputs
    out_dir = Path("./output")
    model_table.to_csv(out_dir / "test_model_table_all.csv", index=False)
    valid_table.to_csv(out_dir / "test_model_table_valid.csv", index=False)
    print(f"\n✓ Saved model tables:")
    print(f"  All bouts: {out_dir / 'test_model_table_all.csv'}")
    print(f"  Valid labels only: {out_dir / 'test_model_table_valid.csv'}")
    
    # Summary statistics
    if len(valid_table) > 0 and 'delta_rmssd' in valid_table.columns:
        print(f"\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        print(f"\nHRV Recovery (delta_rmssd):")
        print(f"  Mean: {valid_table['delta_rmssd'].mean():.4f}")
        print(f"  Std: {valid_table['delta_rmssd'].std():.4f}")
        print(f"  Range: [{valid_table['delta_rmssd'].min():.4f}, {valid_table['delta_rmssd'].max():.4f}]")
        
        if 'hr_mean' in valid_table.columns:
            print(f"\nHeart Rate during effort:")
            hr_valid = valid_table['hr_mean'].dropna()
            if len(hr_valid) > 0:
                print(f"  Mean: {hr_valid.mean():.1f} bpm")
                print(f"  Range: [{hr_valid.min():.1f}, {hr_valid.max():.1f}] bpm")
        
        if 'effort' in valid_table.columns:
            print(f"\nEffort ratings:")
            print(f"  Range: [{valid_table['effort'].min():.1f}, {valid_table['effort'].max():.1f}]")
            
            # Correlation with delta_rmssd
            valid_with_effort = valid_table[valid_table['effort'].notna()]
            if len(valid_with_effort) > 1:
                corr = valid_with_effort['delta_rmssd'].corr(valid_with_effort['effort'])
                print(f"  Correlation (delta_rmssd vs effort): {corr:.3f}")
