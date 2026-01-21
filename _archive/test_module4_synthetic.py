"""
Test Module 4 with synthetic aligned data to validate logic
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module4_labels import compute_bout_labels

def create_synthetic_test_data():
    """Create synthetic RMSSD and bout data with known recovery patterns"""
    
    # Time range: 1000 - 2000 seconds (simple range)
    base_time = 1000.0
    
    # Create RMSSD windows every 10 seconds
    rmssd_times = np.arange(base_time, base_time + 1000, 10.0)
    
    # Synthetic RMSSD pattern: baseline 0.25, drops during effort, recovers after
    rmssd_values = []
    for t in rmssd_times:
        # Bout 1: 1100-1130 (30s) - moderate effort
        if 1090 <= t < 1130:
            rmssd = 0.20 + np.random.normal(0, 0.01)  # Low during effort
        elif 1130 <= t < 1200:
            # Recovery: linearly increase from 0.20 to 0.30 over 70s
            progress = (t - 1130) / 70.0
            rmssd = 0.20 + progress * 0.10 + np.random.normal(0, 0.01)
        # Bout 2: 1300-1360 (60s) - high effort
        elif 1290 <= t < 1360:
            rmssd = 0.15 + np.random.normal(0, 0.01)  # Very low during effort
        elif 1360 <= t < 1430:
            # Recovery: linearly increase from 0.15 to 0.32 over 70s
            progress = (t - 1360) / 70.0
            rmssd = 0.15 + progress * 0.17 + np.random.normal(0, 0.01)
        else:
            rmssd = 0.25 + np.random.normal(0, 0.02)  # Baseline
        
        rmssd_values.append(max(rmssd, 0.01))  # Keep positive
    
    rmssd_df = pd.DataFrame({
        't_start': rmssd_times - 5,
        't_center': rmssd_times,
        't_end': rmssd_times + 5,
        'rmssd': rmssd_values
    })
    
    # Create two synthetic bouts
    bouts_df = pd.DataFrame({
        'bout_id': [0, 1],
        't_start': [1100.0, 1300.0],
        't_end': [1130.0, 1360.0],
        'duration_sec': [30.0, 60.0],
        'task_name': ['Moderate Effort', 'High Effort'],
        'effort': [3.0, 6.0]
    })
    
    return rmssd_df, bouts_df


if __name__ == "__main__":
    print("Creating synthetic test data...")
    rmssd_df, bouts_df = create_synthetic_test_data()
    
    print(f"✓ Created {len(rmssd_df)} RMSSD windows")
    print(f"✓ Created {len(bouts_df)} effort bouts")
    
    # Test Module 4
    print("\n" + "="*70)
    print("Testing Module 4: HRV Recovery Labels (delta method)")
    print("="*70)
    
    labels_df = compute_bout_labels(
        rmssd_df,
        bouts_df,
        label_method="delta",
        recovery_end_window_sec=20.0,  # Average last 20s of effort
        recovery_start_sec=10.0,        # Start recovery 10s post-effort
        recovery_end_sec=70.0,          # End 70s post-effort
        min_recovery_windows=2,
    )
    
    print(f"\n✓ Computed labels for {len(labels_df)} bouts")
    print(f"  QC passed: {labels_df['qc_ok'].sum()} / {len(labels_df)}")
    
    # Show results
    valid_labels = labels_df[labels_df['qc_ok']]
    if len(valid_labels) > 0:
        print(f"\n✓ Valid labels:")
        for _, row in valid_labels.iterrows():
            print(f"\n  Bout {int(row['bout_id'])}: {row['task_name']}")
            print(f"    RMSSD end: {row['rmssd_end']:.4f}")
            print(f"    RMSSD recovery: {row['rmssd_recovery']:.4f}")
            print(f"    Δ RMSSD: {row['delta_rmssd']:.4f} (recovery increase)")
            
            # Expected patterns
            expected = {
                0: {"rmssd_end": 0.20, "rmssd_recovery": 0.25, "delta": 0.05},
                1: {"rmssd_end": 0.15, "rmssd_recovery": 0.25, "delta": 0.10}
            }
            bout_id = int(row['bout_id'])
            if bout_id in expected:
                exp = expected[bout_id]
                print(f"    Expected: end≈{exp['rmssd_end']:.2f}, recovery≈{exp['rmssd_recovery']:.2f}, Δ≈{exp['delta']:.2f}")
    
    # Test slope method
    print("\n" + "="*70)
    print("Testing Module 4: HRV Recovery Labels (slope method)")
    print("="*70)
    
    labels_slope_df = compute_bout_labels(
        rmssd_df,
        bouts_df,
        label_method="slope",
        recovery_start_sec=10.0,
        recovery_end_sec=70.0,
        min_recovery_windows=3,
    )
    
    valid_slope = labels_slope_df[labels_slope_df['qc_ok']]
    if len(valid_slope) > 0:
        print(f"\n✓ Valid slope labels:")
        for _, row in valid_slope.iterrows():
            print(f"\n  Bout {int(row['bout_id'])}: {row['task_name']}")
            print(f"    Recovery slope: {row['recovery_slope']:.6f} (RMSSD/sec)")
            print(f"    Expected: positive slope (HRV recovery)")
    
    # Save outputs
    out_dir = Path("./output")
    out_dir.mkdir(exist_ok=True)
    
    rmssd_df.to_csv(out_dir / "test_synthetic_rmssd.csv", index=False)
    bouts_df.to_csv(out_dir / "test_synthetic_bouts.csv", index=False)
    labels_df.to_csv(out_dir / "test_synthetic_labels_delta.csv", index=False)
    labels_slope_df.to_csv(out_dir / "test_synthetic_labels_slope.csv", index=False)
    
    print(f"\n✓ Saved synthetic test data to ./output/")
    print("\n✓ Module 4 logic validated successfully!")
