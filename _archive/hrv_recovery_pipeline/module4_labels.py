"""
Module 4: Compute HRV Recovery Label per Bout

For each effort bout: compute RMSSD at end of effort, RMSSD during recovery, and recovery metrics.
Recovery metric options: Δ RMSSD at 60s post-effort, or RMSSD recovery slope.
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def compute_hrv_recovery_label(
    rmssd_df: pd.DataFrame,
    bout_t_start: float,
    bout_t_end: float,
    recovery_end_window_sec: float = 30.0,
    recovery_start_sec: float = 30.0,
    recovery_end_sec: float = 90.0,
    label_method: str = "delta",
    min_recovery_windows: int = 2,
) -> dict:
    """
    Compute HRV recovery label for a single effort bout.
    
    Args:
        rmssd_df: RMSSD windows dataframe [t_start, t_center, t_end, rmssd]
        bout_t_start: Bout start time (seconds)
        bout_t_end: Bout end time (seconds)
        recovery_end_window_sec: RMSSD window before bout_t_end (seconds) to compute end RMSSD
        recovery_start_sec: Recovery interval start after bout_t_end (seconds)
        recovery_end_sec: Recovery interval end after bout_t_end (seconds)
        label_method: "delta" (Δ RMSSD) or "slope" (recovery slope)
        min_recovery_windows: Minimum RMSSD windows in recovery interval
        
    Returns:
        label_dict: {
            'rmssd_end': RMSSD at effort end,
            'rmssd_recovery': RMSSD during recovery window,
            'delta_rmssd': RMSSD recovery change (label if method="delta"),
            'recovery_slope': RMSSD recovery slope (label if method="slope"),
            'qc_ok': bool (data quality OK),
            'note': str (reason for QC fail if applicable)
        }
    """
    result = {
        'rmssd_end': np.nan,
        'rmssd_recovery': np.nan,
        'delta_rmssd': np.nan,
        'recovery_slope': np.nan,
        'qc_ok': False,
        'note': '',
    }
    
    if rmssd_df.empty:
        result['note'] = "Empty RMSSD data"
        return result
    
    # RMSSD at effort end: average last window(s) before bout_t_end
    effort_end_start = bout_t_end - recovery_end_window_sec
    rmssd_end_mask = (rmssd_df['t_center'] >= effort_end_start) & (rmssd_df['t_center'] < bout_t_end)
    rmssd_end_values = rmssd_df.loc[rmssd_end_mask, 'rmssd'].dropna()
    
    if len(rmssd_end_values) == 0:
        result['note'] = "No RMSSD windows at effort end"
        return result
    
    rmssd_end = rmssd_end_values.mean()
    result['rmssd_end'] = rmssd_end
    
    # RMSSD during recovery: average windows in [t_end + recovery_start_sec, t_end + recovery_end_sec]
    recovery_t_start = bout_t_end + recovery_start_sec
    recovery_t_end = bout_t_end + recovery_end_sec
    rmssd_recovery_mask = (rmssd_df['t_center'] >= recovery_t_start) & (rmssd_df['t_center'] < recovery_t_end)
    rmssd_recovery_values = rmssd_df.loc[rmssd_recovery_mask, 'rmssd'].dropna()
    
    if len(rmssd_recovery_values) < min_recovery_windows:
        result['note'] = f"Too few recovery windows ({len(rmssd_recovery_values)} < {min_recovery_windows})"
        return result
    
    rmssd_recovery = rmssd_recovery_values.mean()
    result['rmssd_recovery'] = rmssd_recovery
    
    # Compute label
    if label_method == "delta":
        delta_rmssd = rmssd_recovery - rmssd_end
        result['delta_rmssd'] = delta_rmssd
        result['qc_ok'] = True
        
    elif label_method == "slope":
        # Linear regression: RMSSD recovery over time
        recovery_rmssd_df = rmssd_df[rmssd_recovery_mask].copy()
        
        if len(recovery_rmssd_df) < 2:
            result['note'] = "Too few recovery windows for slope"
            return result
        
        # Normalize time relative to recovery start
        recovery_rmssd_df['t_relative'] = recovery_rmssd_df['t_center'] - recovery_t_start
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                recovery_rmssd_df['t_relative'],
                recovery_rmssd_df['rmssd']
            )
            result['recovery_slope'] = slope
            result['qc_ok'] = True
        except Exception as e:
            result['note'] = f"Slope computation failed: {e}"
    
    else:
        result['note'] = f"Unknown label_method: {label_method}"
    
    return result


def compute_bout_labels(
    rmssd_df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    label_method: str = "delta",
    **compute_kwargs
) -> pd.DataFrame:
    """
    Compute HRV recovery labels for all effort bouts.
    
    Args:
        rmssd_df: RMSSD windows dataframe
        bouts_df: Effort bouts dataframe [bout_id, t_start, t_end, task_name]
        label_method: "delta" or "slope"
        **compute_kwargs: Arguments for compute_hrv_recovery_label
        
    Returns:
        labels_df: DataFrame with columns [bout_id, task_name, rmssd_end, 
                                          rmssd_recovery, delta_rmssd (or recovery_slope), qc_ok, note]
    """
    results = []
    
    for idx, bout in bouts_df.iterrows():
        label_dict = compute_hrv_recovery_label(
            rmssd_df,
            bout['t_start'],
            bout['t_end'],
            label_method=label_method,
            **compute_kwargs
        )
        
        label_dict['bout_id'] = bout['bout_id']
        label_dict['task_name'] = bout.get('task_name', 'unknown')
        
        results.append(label_dict)
    
    labels_df = pd.DataFrame(results)
    
    # QC summary
    n_total = len(labels_df)
    n_qc_ok = labels_df['qc_ok'].sum()
    pct_qc = 100.0 * n_qc_ok / n_total if n_total > 0 else 0
    
    logger.info(
        f"Computed labels for {n_total} bouts ({n_qc_ok} QC pass, {pct_qc:.1f}%)"
    )
    
    if n_qc_ok > 0:
        label_col = 'delta_rmssd' if label_method == "delta" else 'recovery_slope'
        if label_col in labels_df.columns:
            valid_labels = labels_df.loc[labels_df['qc_ok'], label_col]
            logger.info(
                f"  {label_col} range: [{valid_labels.min():.4f}, {valid_labels.max():.4f}]"
            )
    
    return labels_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    import sys
    if len(sys.argv) > 2:
        rmssd_path = sys.argv[1]
        bouts_path = sys.argv[2]
        
        rmssd_df = pd.read_csv(rmssd_path)
        bouts_df = pd.read_csv(bouts_path)
        
        labels_df = compute_bout_labels(rmssd_df, bouts_df)
        print(labels_df)
