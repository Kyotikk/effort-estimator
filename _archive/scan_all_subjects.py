"""
Scan all subjects in parsingsim3/4/5 to find which have overlapping PPG and ADL data.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path("/Users/pascalschlegel/data/interim")

def check_time_overlap(ppg_path, adl_path, timezone_offset=28800):
    """Check if PPG and ADL files overlap in time"""
    try:
        # Read PPG
        compression = 'gzip' if str(ppg_path).endswith('.gz') else None
        ppg_df = pd.read_csv(ppg_path, compression=compression, nrows=10)
        ppg_start = ppg_df['time'].min() + timezone_offset
        
        ppg_df_end = pd.read_csv(ppg_path, compression=compression)
        ppg_end = ppg_df_end['time'].max() + timezone_offset
        
        # Read ADL
        compression = 'gzip' if str(adl_path).endswith('.gz') else None
        adl_df = pd.read_csv(adl_path, compression=compression)
        
        if 'time' in adl_df.columns:
            adl_start = adl_df['time'].min()
            adl_end = adl_df['time'].max()
        else:
            # Skip header rows for SCAI format
            adl_df = pd.read_csv(adl_path, compression=compression, skiprows=2)
            if 'Time' in adl_df.columns:
                # Parse SCAI timestamps
                return None  # Will check differently
            return None
        
        # Check overlap
        overlap = not (ppg_end < adl_start or ppg_start > adl_end)
        
        return {
            'ppg_start': ppg_start,
            'ppg_end': ppg_end,
            'ppg_duration_min': (ppg_end - ppg_start) / 60,
            'adl_start': adl_start,
            'adl_end': adl_end,
            'overlap': overlap,
            'gap_hours': (adl_start - ppg_end) / 3600 if not overlap else 0
        }
    except Exception as e:
        return {'error': str(e)}


def scan_all_subjects():
    """Scan all subjects in parsingsim3/4/5"""
    
    results = []
    
    for parsingsim in ['parsingsim3', 'parsingsim4', 'parsingsim5']:
        parsingsim_dir = DATA_ROOT / parsingsim
        
        if not parsingsim_dir.exists():
            continue
        
        for subject_dir in sorted(parsingsim_dir.glob('sim_*')):
            subject_name = subject_dir.name
            
            print(f"\n{'='*70}")
            print(f"{parsingsim}/{subject_name}")
            print(f"{'='*70}")
            
            # Find PPG files
            ppg_dir = subject_dir / "corsano_wrist_ppg2_green_6"
            ppg_files = []
            if ppg_dir.exists():
                ppg_files = list(ppg_dir.glob("2025-12-*.csv")) + list(ppg_dir.glob("2025-12-*.csv.gz"))
            
            # Find ADL files
            adl_dir = subject_dir / "scai_app"
            adl_files = []
            if adl_dir.exists():
                adl_files = list(adl_dir.glob("ADLs*.csv")) + list(adl_dir.glob("ADLs*.csv.gz"))
            
            print(f"PPG files: {len(ppg_files)}")
            for f in ppg_files:
                print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
            
            print(f"ADL files: {len(adl_files)}")
            for f in adl_files:
                print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
            
            # Check combinations
            for ppg_file in ppg_files:
                for adl_file in adl_files:
                    print(f"\nChecking: {ppg_file.name} + {adl_file.name}")
                    
                    overlap_info = check_time_overlap(ppg_file, adl_file)
                    
                    if overlap_info is None:
                        print("  → Cannot determine (special format)")
                        continue
                    
                    if 'error' in overlap_info:
                        print(f"  → Error: {overlap_info['error']}")
                        continue
                    
                    ppg_start_dt = datetime.fromtimestamp(overlap_info['ppg_start'])
                    ppg_end_dt = datetime.fromtimestamp(overlap_info['ppg_end'])
                    adl_start_dt = datetime.fromtimestamp(overlap_info['adl_start'])
                    adl_end_dt = datetime.fromtimestamp(overlap_info['adl_end'])
                    
                    print(f"  PPG: {ppg_start_dt} → {ppg_end_dt} ({overlap_info['ppg_duration_min']:.1f} min)")
                    print(f"  ADL: {adl_start_dt} → {adl_end_dt}")
                    
                    if overlap_info['overlap']:
                        print(f"  ✓ OVERLAP - Can process!")
                        results.append({
                            'project': parsingsim,
                            'subject': subject_name,
                            'ppg_file': ppg_file,
                            'adl_file': adl_file,
                            'status': 'overlap'
                        })
                    else:
                        print(f"  ✗ No overlap (gap: {overlap_info['gap_hours']:.1f} hours)")
                        results.append({
                            'project': parsingsim,
                            'subject': subject_name,
                            'ppg_file': ppg_file,
                            'adl_file': adl_file,
                            'status': 'no_overlap',
                            'gap_hours': overlap_info['gap_hours']
                        })
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("SCANNING ALL SUBJECTS FOR OVERLAPPING PPG + ADL DATA")
    print("="*70)
    
    results = scan_all_subjects()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    working = [r for r in results if r['status'] == 'overlap']
    
    print(f"\n✓ Found {len(working)} working combinations:")
    for r in working:
        print(f"  • {r['project']}/{r['subject']}")
        print(f"    PPG: {r['ppg_file'].name}")
        print(f"    ADL: {r['adl_file'].name}")
    
    print(f"\n✗ {len(results) - len(working)} combinations don't overlap")
    
    print("\n" + "="*70)
