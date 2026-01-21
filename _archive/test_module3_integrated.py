"""
Test Module 3 integrated function with elderly3 data
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module3_bouts import parse_adl_intervals

if __name__ == "__main__":
    adl_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv")
    
    print("Testing parse_adl_intervals() with SCAI format auto-detection...")
    bouts_df = parse_adl_intervals(adl_path, format='auto')
    
    print(f"\n✓ Parsed {len(bouts_df)} effort bouts")
    
    if len(bouts_df) > 0:
        print(f"\nBout statistics:")
        print(f"  Duration range: {bouts_df['duration_sec'].min():.1f} - {bouts_df['duration_sec'].max():.1f} sec")
        print(f"  Mean duration: {bouts_df['duration_sec'].mean():.1f} sec")
        print(f"  Effort range: {bouts_df['effort'].min()} - {bouts_df['effort'].max()}")
        
        print(f"\nTask distribution:")
        print(bouts_df['task_name'].value_counts().head(10))
        
        print(f"\nFirst 5 bouts:")
        print(bouts_df[['t_start', 't_end', 'duration_sec', 'task_name', 'effort']].head())
        
        # Save output
        out_path = Path("./output/test_bouts_module3.csv")
        bouts_df.to_csv(out_path, index=False)
        print(f"\n✓ Saved to {out_path}")
