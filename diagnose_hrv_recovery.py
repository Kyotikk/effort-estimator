"""
DIAGNOSTIC: Verify HRV Recovery Label Extraction

Shows exactly:
1. Which subject is being processed
2. Each ADL activity timing
3. RMSSD DURING effort (baseline)
4. RMSSD AFTER effort (recovery period)
5. Delta RMSSD calculation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module1_ibi import extract_ibi_timeseries
from hrv_recovery_pipeline.module2_rmssd import compute_rmssd_windows
from hrv_recovery_pipeline.module3_bouts import parse_adl_intervals

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("/Users/pascalschlegel/data/interim")
TIMEZONE_OFFSET_SEC = 8 * 3600
FS_PPG = 128

# Subjects with overlapping PPG and ADL data
SUBJECTS = [
    ("parsingsim3", "sim_elderly3", "2025-12-04.csv", "ADLs_1.csv"),  # Uncompressed
    ("parsingsim3", "sim_healthy3", "2025-12-04.csv.gz", "ADLs_1.csv.gz"),  # Compressed
    ("parsingsim4", "sim_elderly4", "2025-12-05.csv.gz", "ADLs_1.csv.gz"),  # Compressed
    ("parsingsim5", "sim_severe5", "2025-12-05.csv.gz", "ADLs_1.csv.gz"),  # Compressed
]


def diagnose_subject(project, subject_name, ppg_file, adl_file):
    """Diagnose one subject's HRV recovery extraction"""
    
    logger.info("\n" + "="*80)
    logger.info(f"SUBJECT: {project}/{subject_name}")
    logger.info("="*80)
    
    subject_dir = DATA_ROOT / project / subject_name
    # Use specified files
    ppg_path = subject_dir / "corsano_wrist_ppg2_green_6" / ppg_file
    adl_path = subject_dir / "scai_app" / adl_file
    
    if not ppg_path.exists():
        logger.error(f"✗ PPG not found: {ppg_path}")
        return
    if not adl_path.exists():
        logger.error(f"✗ ADL not found: {adl_path}")
        return
    
    # Step 1: Extract IBIs
    logger.info("\nStep 1: Extracting IBIs from PPG...")
    # Check if compressed or not
    compression = 'gzip' if str(ppg_path).endswith('.gz') else None
    ppg_df = pd.read_csv(ppg_path, compression=compression)
    
    ibi_df = extract_ibi_timeseries(
        ppg_df,
        value_col='value',
        time_col='time',
        fs=FS_PPG,
        distance_ms=300
    )
    
    # Apply timezone correction
    # PPG timestamps are in Japan/Beijing time (UTC+8), ADL is in Zurich time (UTC+1)
    # Need to add 8 hours to PPG to align with ADL
    logger.info(f"  IBI time range (BEFORE correction): {ibi_df['t'].min():.1f} → {ibi_df['t'].max():.1f}")
    ibi_df['t'] += TIMEZONE_OFFSET_SEC
    logger.info(f"  IBI time range (AFTER +8h): {ibi_df['t'].min():.1f} → {ibi_df['t'].max():.1f}")
    
    logger.info(f"  ✓ Extracted {len(ibi_df)} IBIs")
    logger.info(f"  Time range: {ibi_df['t'].min():.1f} → {ibi_df['t'].max():.1f}")
    logger.info(f"  Duration: {(ibi_df['t'].max() - ibi_df['t'].min())/60:.1f} minutes")
    
    # Step 2: Compute RMSSD windows
    logger.info("\nStep 2: Computing RMSSD windows (60s window, 10s step)...")
    rmssd_df = compute_rmssd_windows(
        ibi_df,
        window_len_sec=60.0,
        step_sec=10.0,
        min_beats=10
    )
    
    # DON'T apply timezone correction - already applied to IBI timestamps
    # rmssd_df['t_start'] += TIMEZONE_OFFSET_SEC
    # rmssd_df['t_center'] += TIMEZONE_OFFSET_SEC
    # rmssd_df['t_end'] += TIMEZONE_OFFSET_SEC
    
    valid_rmssd = rmssd_df[rmssd_df['rmssd'].notna()]
    logger.info(f"  ✓ Computed {len(rmssd_df)} windows ({len(valid_rmssd)} valid)")
    logger.info(f"  Time range: {valid_rmssd['t_center'].min():.1f} → {valid_rmssd['t_center'].max():.1f}")
    
    # Step 3: Parse ADL bouts
    logger.info("\nStep 3: Parsing ADL activities...")
    bouts_df = parse_adl_intervals(adl_path, format='auto')
    
    if bouts_df.empty:
        logger.error("  ✗ No ADL activities found")
        return
    
    logger.info(f"  ✓ Parsed {len(bouts_df)} ADL activities")
    
    # Step 4: Show alignment for each activity
    logger.info("\n" + "="*80)
    logger.info("ALIGNMENT ANALYSIS FOR EACH ADL ACTIVITY")
    logger.info("="*80)
    
    for idx, bout in bouts_df.iterrows():
        logger.info(f"\n{'-'*80}")
        logger.info(f"Activity #{idx+1}: {bout['task_name']}")
        logger.info(f"{'-'*80}")
        logger.info(f"  Duration: {bout['duration_sec']:.1f}s")
        logger.info(f"  Effort period: {bout['t_start']:.1f} → {bout['t_end']:.1f}")
        
        # DURING EFFORT: Find RMSSD at end of effort
        effort_rmssd_windows = rmssd_df[
            (rmssd_df['t_center'] >= bout['t_start']) & 
            (rmssd_df['t_center'] <= bout['t_end'])
        ]
        
        if len(effort_rmssd_windows) > 0:
            effort_rmssd = effort_rmssd_windows['rmssd'].median()
            logger.info(f"  RMSSD during effort: {effort_rmssd:.1f}ms (from {len(effort_rmssd_windows)} windows)")
        else:
            logger.info(f"  RMSSD during effort: NO DATA")
        
        # Get RMSSD at end of effort (baseline for recovery)
        end_window = 30.0  # 30s window at end of effort
        end_rmssd_windows = rmssd_df[
            (rmssd_df['t_center'] >= bout['t_end'] - end_window) & 
            (rmssd_df['t_center'] <= bout['t_end'])
        ]
        
        if len(end_rmssd_windows) > 0:
            rmssd_end = end_rmssd_windows['rmssd'].median()
            logger.info(f"  RMSSD at end (baseline): {rmssd_end:.1f}ms (from {len(end_rmssd_windows)} windows)")
        else:
            rmssd_end = None
            logger.info(f"  RMSSD at end (baseline): NO DATA ⚠")
        
        # AFTER EFFORT: Recovery period (10s to 70s after bout ends)
        recovery_start = bout['t_end'] + 10.0
        recovery_end = bout['t_end'] + 70.0
        
        logger.info(f"\n  Recovery period: {recovery_start:.1f} → {recovery_end:.1f}")
        logger.info(f"    (starts 10s after activity, ends 70s after)")
        
        recovery_rmssd_windows = rmssd_df[
            (rmssd_df['t_center'] >= recovery_start) & 
            (rmssd_df['t_center'] <= recovery_end) &
            (rmssd_df['rmssd'].notna())
        ]
        
        if len(recovery_rmssd_windows) >= 2:
            rmssd_recovery = recovery_rmssd_windows['rmssd'].median()
            logger.info(f"  RMSSD during recovery: {rmssd_recovery:.1f}ms (from {len(recovery_rmssd_windows)} windows)")
            
            if rmssd_end is not None:
                delta_rmssd = rmssd_recovery - rmssd_end
                pct_change = (delta_rmssd / rmssd_end) * 100
                logger.info(f"\n  DELTA RMSSD: {delta_rmssd:+.1f}ms ({pct_change:+.1f}%)")
                
                if delta_rmssd > 0:
                    logger.info(f"  → HRV RECOVERED (increased)")
                else:
                    logger.info(f"  → HRV DECLINED (decreased)")
                
                logger.info(f"  QC: ✓ VALID")
            else:
                logger.info(f"  QC: ✗ INVALID (no baseline RMSSD)")
        else:
            logger.info(f"  RMSSD during recovery: NO DATA ({len(recovery_rmssd_windows)} windows) ⚠")
            logger.info(f"  QC: ✗ INVALID (insufficient recovery data)")
            
            # Check why no data
            logger.info(f"\n  Diagnosis:")
            logger.info(f"    Latest RMSSD window: {valid_rmssd['t_center'].max():.1f}")
            logger.info(f"    Need data until: {recovery_end:.1f}")
            gap = recovery_end - valid_rmssd['t_center'].max()
            if gap > 0:
                logger.info(f"    Gap: {gap:.1f}s → PPG recording too short!")


def main():
    logger.info("="*80)
    logger.info("HRV RECOVERY ALIGNMENT DIAGNOSTIC")
    logger.info("="*80)
    logger.info("\nThis script verifies:")
    logger.info("  1. IBI extraction from PPG")
    logger.info("  2. RMSSD windowing")
    logger.info("  3. ADL activity timing")
    logger.info("  4. RMSSD during effort (baseline)")
    logger.info("  5. RMSSD after effort (recovery: 10-70s post-activity)")
    logger.info("  6. Delta RMSSD calculation")
    
    # Diagnose each subject
    for project, subject, ppg_file, adl_file in SUBJECTS:
        diagnose_subject(project, subject, ppg_file, adl_file)
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
