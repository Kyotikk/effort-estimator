#!/usr/bin/env python
"""
Compute RMSSD during activities and recovery slopes after activities using ADL bouts.

Inputs:
  --rmssd-csv: windowed RMSSD file (absolute timestamps in seconds)
  --adl-csv: ADLs file with columns [time, ADLs]

Outputs:
  CSV with per-activity metrics:
    activity, start_time, end_time, duration_s,
    n_activity_windows, rmssd_during_mean, rmssd_during_median,
    n_recovery_windows, recovery_slope_ms_per_s

Recovery slope is fitted on windows whose t_center is in [end + rec_start, end + rec_end].
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def parse_adl_bouts(adl_df: pd.DataFrame) -> List[Dict]:
    """
    Parse ADL start/end events into bouts.
    Expects columns: ['time', 'ADLs'] with strings ending in 'Start' or 'End'.
    """
    bouts = []
    open_bouts = {}
    for _, row in adl_df.iterrows():
        t = float(row['time'])
        label = str(row['ADLs']).strip()
        if label.endswith('Start'):
            activity = label.replace(' Start', '')
            open_bouts[activity] = t
        elif label.endswith('End'):
            activity = label.replace(' End', '')
            if activity in open_bouts:
                bouts.append({
                    'activity': activity,
                    'start_time': open_bouts.pop(activity),
                    'end_time': t
                })
            else:
                logger.warning(f"End without start for activity {activity} at {t}")
    if open_bouts:
        logger.warning(f"Unclosed bouts found: {list(open_bouts.keys())}")
    return bouts


def _fit_slope_ms_per_s(df: pd.DataFrame) -> float:
    """Return slope (ms/s) of rmssd vs t_center; NaN if <2 points."""
    if len(df) < 2:
        return np.nan
    x = df['t_center'].values
    y = df['rmssd'].values
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_recovery_slope(df_windows: pd.DataFrame, start: float, end: float,
                           rec_start: float, rec_end: float) -> Tuple[float, int]:
    """
    Fit slope of RMSSD (ms) over time in recovery window.
    Returns (slope_ms_per_s, n_points). If fewer than 2 points, slope is NaN.
    """
    mask = (df_windows['t_center'] >= end + rec_start) & (df_windows['t_center'] <= end + rec_end)
    rec_df = df_windows.loc[mask, ['t_center', 'rmssd']].dropna()
    n = len(rec_df)
    slope = _fit_slope_ms_per_s(rec_df)
    return slope, n


def compute_activity_rmssd(df_windows: pd.DataFrame, start: float, end: float) -> Tuple[float, float, float, int]:
    """
    Compute mean/median RMSSD and slope during the activity window.
    Returns (mean, median, slope_ms_per_s, n_points).
    """
    mask = (df_windows['t_center'] >= start) & (df_windows['t_center'] <= end)
    act_df = df_windows.loc[mask, ['t_center', 'rmssd']].dropna()
    if len(act_df) == 0:
        return np.nan, np.nan, np.nan, 0
    slope = _fit_slope_ms_per_s(act_df)
    return act_df['rmssd'].mean(), act_df['rmssd'].median(), slope, len(act_df)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-activity RMSSD and recovery slopes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/activity_rmssd_recovery.py \
    --rmssd-csv data/interim/rmssd_windows_real.csv \
    --adl-csv /Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv.gz \
    --output data/interim/activity_rmssd_metrics.csv \
    --rec-start 10 --rec-end 60
        """
    )
    parser.add_argument('--rmssd-csv', required=True, help='RMSSD windows CSV (absolute times)')
    parser.add_argument('--adl-csv', required=True, help='ADL file with time, ADLs columns')
    parser.add_argument('--output', required=True, help='Output CSV for per-activity metrics')
    parser.add_argument('--rec-start', type=float, default=10.0, help='Recovery window start (s after end) [default: 10]')
    parser.add_argument('--rec-end', type=float, default=60.0, help='Recovery window end (s after end) [default: 60]')
    parser.add_argument('--adl-offset-hours', type=float, default=0.0,
                        help='Apply a time offset (hours) to ADL timestamps, e.g., -8 to shift local+8h logs to UTC')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df_rmssd = pd.read_csv(args.rmssd_csv)
    # Ensure time columns exist
    required_cols = {'t_center', 'rmssd'}
    missing = required_cols - set(df_rmssd.columns)
    if missing:
        raise ValueError(f"Missing columns in RMSSD file: {missing}")

    df_adl = pd.read_csv(args.adl_csv)
    if not {'time', 'ADLs'}.issubset(df_adl.columns):
        raise ValueError("ADL file must contain columns: time, ADLs")

    if args.adl_offset_hours != 0:
        delta = args.adl_offset_hours * 3600.0
        df_adl['time'] = df_adl['time'] + delta
        logger.info(f"Shifted ADL times by {args.adl_offset_hours} h ({delta} s)")

    # Parse bouts
    bouts = parse_adl_bouts(df_adl)
    logger.info(f"Parsed {len(bouts)} activity bouts from ADL file")

    rows = []
    for b in bouts:
        start = b['start_time']
        end = b['end_time']
        duration = end - start
        rmssd_mean, rmssd_median, slope_act, n_act = compute_activity_rmssd(df_rmssd, start, end)
        slope_rec, n_rec = compute_recovery_slope(df_rmssd, start, end, args.rec_start, args.rec_end)
        rows.append({
            'activity': b['activity'],
            'start_time': start,
            'end_time': end,
            'duration_s': duration,
            'n_activity_windows': n_act,
            'rmssd_during_mean': rmssd_mean,
            'rmssd_during_median': rmssd_median,
            'activity_slope_ms_per_s': slope_act,
            'n_recovery_windows': n_rec,
            'recovery_slope_ms_per_s': slope_rec,
            'slope_recovery_minus_activity': (slope_rec - slope_act) if not np.isnan(slope_rec) and not np.isnan(slope_act) else np.nan
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Saved activity RMSSD metrics: {args.output}")

    # Quick summary
    if len(df_out) > 0:
        logger.info("\nSummary of recovery slopes (ms/s):")
        logger.info(df_out['recovery_slope_ms_per_s'].describe())


if __name__ == '__main__':
    main()
