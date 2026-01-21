"""
Test Module 4: Compute HRV recovery labels from RMSSD windows and effort bouts
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module4_labels import compute_bout_labels

if __name__ == "__main__":
    # Load RMSSD windows (Module 2 output)
    rmssd_path = Path("./output/test_rmssd_elderly3.csv")
    rmssd_df = pd.read_csv(rmssd_path)
    print(f"✓ Loaded RMSSD windows: {len(rmssd_df)} rows")
    print(f"  Columns: {list(rmssd_df.columns)}")
    print(f"  Time range: {rmssd_df['t_center'].min():.0f} - {rmssd_df['t_center'].max():.0f}")
    
    # Apply timezone correction: PPG recorded in Japan time, ADL in Zurich time
    # Offset = 8 hours (UTC+9 vs UTC+1)
    TIMEZONE_OFFSET_SEC = 8 * 3600
    print(f"\n⚠ Applying timezone correction: +{TIMEZONE_OFFSET_SEC/3600:.0f} hours")
    rmssd_df['t_start'] += TIMEZONE_OFFSET_SEC
    rmssd_df['t_center'] += TIMEZONE_OFFSET_SEC
    rmssd_df['t_end'] += TIMEZONE_OFFSET_SEC
    print(f"  Corrected time range: {rmssd_df['t_center'].min():.0f} - {rmssd_df['t_center'].max():.0f}")
    
    # Load effort bouts (Module 3 output)
    bouts_path = Path("./output/test_bouts_module3.csv")
    bouts_df = pd.read_csv(bouts_path)
    print(f"\n✓ Loaded effort bouts: {len(bouts_df)} rows")
    
    # Add bout_id if not present
    if 'bout_id' not in bouts_df.columns:
        bouts_df['bout_id'] = range(len(bouts_df))
    
    print(f"  Bout duration range: {bouts_df['duration_sec'].min():.1f} - {bouts_df['duration_sec'].max():.1f} sec")
    print(f"  Bout time range: {bouts_df['t_start'].min():.0f} - {bouts_df['t_end'].max():.0f}")
    
    # Compute HRV recovery labels using delta method
    print("\n" + "="*70)
    print("Computing HRV recovery labels (delta method)...")
    print("="*70)
    
    labels_df = compute_bout_labels(
        rmssd_df,
        bouts_df,
        label_method="delta",
        recovery_end_window_sec=30.0,  # Average RMSSD in last 30s of effort
        recovery_start_sec=10.0,        # Start recovery 10s after effort ends
        recovery_end_sec=70.0,          # End recovery 70s after effort ends (60s window)
        min_recovery_windows=2,         # Require at least 2 RMSSD windows in recovery
    )
    
    print(f"\n✓ Computed labels for {len(labels_df)} bouts")
    print(f"  QC passed: {labels_df['qc_ok'].sum()} / {len(labels_df)} ({100*labels_df['qc_ok'].sum()/len(labels_df):.1f}%)")
    
    # Show QC failures
    qc_fail = labels_df[~labels_df['qc_ok']]
    if len(qc_fail) > 0:
        print(f"\n  QC failures by reason:")
        print(qc_fail['note'].value_counts())
    
    # Show valid labels
    valid_labels = labels_df[labels_df['qc_ok']]
    if len(valid_labels) > 0:
        print(f"\n✓ Valid labels statistics:")
        print(f"  Delta RMSSD range: {valid_labels['delta_rmssd'].min():.4f} - {valid_labels['delta_rmssd'].max():.4f}")
        print(f"  Mean delta RMSSD: {valid_labels['delta_rmssd'].mean():.4f}")
        print(f"  Std delta RMSSD: {valid_labels['delta_rmssd'].std():.4f}")
        
        print(f"\n  RMSSD at effort end: {valid_labels['rmssd_end'].mean():.4f} ± {valid_labels['rmssd_end'].std():.4f}")
        print(f"  RMSSD during recovery: {valid_labels['rmssd_recovery'].mean():.4f} ± {valid_labels['rmssd_recovery'].std():.4f}")
        
        print(f"\nFirst 10 bouts with valid labels:")
        display_cols = ['bout_id', 'task_name', 'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'qc_ok']
        print(valid_labels[display_cols].head(10).to_string(index=False))
        
        # Correlation with effort rating
        if 'effort' in bouts_df.columns:
            # Merge effort ratings
            labels_with_effort = labels_df.merge(
                bouts_df[['bout_id', 'effort']], 
                on='bout_id', 
                how='left'
            )
            
            valid_with_effort = labels_with_effort[
                labels_with_effort['qc_ok'] & labels_with_effort['effort'].notna()
            ]
            
            if len(valid_with_effort) > 1:
                corr = valid_with_effort['delta_rmssd'].corr(valid_with_effort['effort'])
                print(f"\n  Correlation (delta_rmssd vs effort): {corr:.3f}")
    
    # Save output
    out_path = Path("./output/test_labels_elderly3.csv")
    labels_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    
    # Also test slope method
    print("\n" + "="*70)
    print("Testing slope method...")
    print("="*70)
    
    labels_slope_df = compute_bout_labels(
        rmssd_df,
        bouts_df,
        label_method="slope",
        recovery_start_sec=10.0,
        recovery_end_sec=70.0,
        min_recovery_windows=3,  # Need at least 3 points for slope
    )
    
    valid_slope = labels_slope_df[labels_slope_df['qc_ok']]
    if len(valid_slope) > 0:
        print(f"\n✓ Slope method: {len(valid_slope)} valid labels")
        print(f"  Recovery slope range: {valid_slope['recovery_slope'].min():.6f} - {valid_slope['recovery_slope'].max():.6f}")
        print(f"  Mean recovery slope: {valid_slope['recovery_slope'].mean():.6f}")
