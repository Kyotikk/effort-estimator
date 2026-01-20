"""
Parser for SCAI app ADL timeline files.

Handles the custom format used in scai_app/ADLs_1.csv files.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parse_adl_timeline(filepath: str, session_start_time: Optional[float] = None) -> pd.DataFrame:
    """
    Parse ADL timeline from SCAI app CSV file.
    
    The file format is:
    - Row 0: User ID and recording start time
    - Row 1: Column headers (Time, ADLs, Effort)
    - Row 2+: Timestamps, activity names (Start/End pairs), Borg RPE values
    
    NOTE: ADL timestamps are in LOCAL time. We convert to relative seconds from session start.
    
    Args:
        filepath: Path to ADLs_1.csv file
        session_start_time: Unix timestamp of session start (if None, extracts from file)
        
    Returns:
        DataFrame with columns: adl_id, adl_name, start_time, end_time, duration_sec, borg_rpe
        (times are RELATIVE seconds from session start)
    """
    logger.info(f"Parsing ADL timeline from {filepath}")
    
    # Read raw file
    df_raw = pd.read_csv(filepath)
    
    # Extract session start time from row 0
    start_str = df_raw.iloc[0, 0]
    if 'Start of Recording:' in start_str:
        date_str = start_str.split('Start of Recording:')[1].strip()
        # Parse without timezone conversion - we'll use relative times
        adl_session_start = parse_timestamp(date_str, timezone_offset_hours=0)
        logger.info(f"ADL session start (local time): {date_str}")
    
    # Skip header rows (0 and 1), get actual data starting from row 2
    df_data = df_raw.iloc[2:].copy()
    df_data.columns = ['timestamp', 'activity', 'effort']
    
    # Parse timestamps (NOTE: SCAI app has 8-hour offset, actual time is -8 hours)
    df_data['local_time'] = df_data['timestamp'].apply(lambda x: parse_timestamp(x, timezone_offset_hours=+8))
    
    # Convert to seconds relative to session start  
    adl_session_start_corrected = parse_timestamp(date_str, timezone_offset_hours=+8)
    df_data['time_sec'] = df_data['local_time'] - adl_session_start_corrected
    
    # Separate Start and End events
    df_data['event_type'] = df_data['activity'].apply(lambda x: 'Start' if 'Start' in str(x) else 'End')
    df_data['adl_name'] = df_data['activity'].apply(lambda x: str(x).replace(' Start', '').replace(' End', ''))
    
    # Convert Borg RPE to numeric
    df_data['borg_rpe'] = pd.to_numeric(df_data['effort'], errors='coerce')
    
    # Pair Start and End events
    adl_records = []
    adl_id = 0
    
    i = 0
    while i < len(df_data):
        row = df_data.iloc[i]
        
        if row['event_type'] == 'Start':
            start_time = row['time_sec']
            adl_name = row['adl_name']
            
            # Find matching End event
            end_time = None
            borg_rpe = None
            
            # Look ahead for End event with same activity name
            for j in range(i + 1, len(df_data)):
                next_row = df_data.iloc[j]
                if next_row['adl_name'] == adl_name and next_row['event_type'] == 'End':
                    end_time = next_row['time_sec']
                    borg_rpe = next_row['borg_rpe']
                    break
            
            if end_time is not None:
                duration_sec = end_time - start_time
                
                adl_records.append({
                    'adl_id': adl_id,
                    'adl_name': adl_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_sec': duration_sec,
                    'borg_rpe': borg_rpe
                })
                adl_id += 1
        
        i += 1
    
    result_df = pd.DataFrame(adl_records)
    
    logger.info(f"Parsed {len(result_df)} ADL segments")
    logger.info(f"Total duration: {result_df['end_time'].max() / 60:.1f} minutes")
    logger.info(f"ADL names: {result_df['adl_name'].unique()[:10].tolist()}...")
    
    return result_df


def parse_timestamp(timestamp_str: str, timezone_offset_hours: int = 1) -> float:
    """
    Parse timestamp string to Unix time.
    
    The ADL timestamps are in local time (CET/CEST - Zurich time).
    We need to convert to UTC to match sensor timestamps.
    
    Format: DD-MM-YYYY-HH-MM-SS-mmm
    Example: 04-12-2025-17-46-22-075
    
    Args:
        timestamp_str: Timestamp string in local time
        timezone_offset_hours: Hours to subtract to convert to UTC (default 1 for CET in winter)
        
    Returns:
        Unix timestamp (seconds since epoch, UTC)
    """
    try:
        # Parse format: DD-MM-YYYY-HH-MM-SS-mmm
        parts = timestamp_str.strip().split('-')
        if len(parts) == 7:
            day, month, year, hour, minute, second, millisecond = parts
            # Create datetime in local time
            dt = datetime(int(year), int(month), int(day), 
                         int(hour), int(minute), int(second), 
                         int(millisecond) * 1000)
            # Convert to UTC by subtracting timezone offset
            # December in Zurich is CET (UTC+1)
            utc_timestamp = dt.timestamp() - (timezone_offset_hours * 3600)
            return utc_timestamp
        else:
            logger.warning(f"Unexpected timestamp format: {timestamp_str}")
            return np.nan
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return np.nan
