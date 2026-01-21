"""
Test Module 3: Parse ADL intervals from elderly3 data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def parse_scai_adl_format(adl_path: Path):
    """
    Parse SCAI ADL format: Time, ADLs, Effort
    - "Activity Start" and "Activity End" pairs
    - Effort rating on "End" row
    
    Convert to: t_start (unix), t_end (unix), task_name, effort
    """
    # Skip first 2 rows (User ID and Start of Recording)
    df = pd.read_csv(adl_path, skiprows=2)
    
    print(f"✓ Loaded ADL file: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    # Parse timestamp format: "04-12-2025-17-46-22-089" (DD-MM-YYYY-HH-MM-SS-mmm)
    def parse_timestamp(ts_str):
        if pd.isna(ts_str):
            return None
        try:
            # Format: DD-MM-YYYY-HH-MM-SS-mmm
            dt = datetime.strptime(ts_str, "%d-%m-%Y-%H-%M-%S-%f")
            return dt.timestamp()
        except Exception as e:
            print(f"Failed to parse: {ts_str} ({e})")
            return None
    
    df['unix_time'] = df['Time'].apply(parse_timestamp)
    
    # Pair Start/End rows
    bouts = []
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        adl = row['ADLs']
        
        if isinstance(adl, str) and 'Start' in adl:
            # Look for matching End
            task_base = adl.replace(' Start', '')
            
            # Find next row
            next_row = df.iloc[i + 1]
            next_adl = next_row['ADLs']
            
            if isinstance(next_adl, str) and next_adl == f"{task_base} End":
                t_start = row['unix_time']
                t_end = next_row['unix_time']
                effort = next_row['Effort']
                
                if t_start and t_end:
                    bouts.append({
                        't_start': t_start,
                        't_end': t_end,
                        'duration_sec': t_end - t_start,
                        'task_name': task_base,
                        'effort': effort if pd.notna(effort) else None
                    })
                
                i += 2  # Skip both Start and End
                continue
        
        i += 1
    
    bouts_df = pd.DataFrame(bouts)
    
    print(f"\n✓ Parsed {len(bouts_df)} effort bouts")
    if len(bouts_df) > 0:
        print(f"  Duration range: {bouts_df['duration_sec'].min():.1f} - {bouts_df['duration_sec'].max():.1f} sec")
        print(f"  Mean duration: {bouts_df['duration_sec'].mean():.1f} sec")
        print(f"  Effort range: {bouts_df['effort'].min()} - {bouts_df['effort'].max()}")
        print(f"\nFirst 5 bouts:")
        print(bouts_df.head())
    
    return bouts_df


if __name__ == "__main__":
    adl_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv")
    
    bouts_df = parse_scai_adl_format(adl_path)
    
    # Save output
    out_path = Path("./output/test_bouts_elderly3.csv")
    out_path.parent.mkdir(exist_ok=True)
    bouts_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
