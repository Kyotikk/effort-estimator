"""
RMSSD-based effort label computation from ECG-derived RR intervals.
This is the GROUND TRUTH for effort estimation, never used as input feature.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class RMSSDLabeler:
    """
    Computes physiological effort labels from ECG-derived RMSSD.
    """
    
    def __init__(self, window_size_sec: float = 60.0, 
                 overlap_sec: float = 30.0):
        """
        Args:
            window_size_sec: Window size for RMSSD computation (seconds)
            overlap_sec: Overlap between windows (seconds)
        """
        self.window_size_sec = window_size_sec
        self.overlap_sec = overlap_sec
    
    def compute_rmssd(self, rr_intervals: np.ndarray) -> float:
        """
        Compute Root Mean Square of Successive Differences (RMSSD).
        
        RMSSD reflects parasympathetic (vagal) activity:
        - Higher RMSSD = more relaxed / lower effort
        - Lower RMSSD = more stressed / higher effort
        
        Args:
            rr_intervals: Clean RR intervals in milliseconds
            
        Returns:
            RMSSD value (ms)
        """
        if len(rr_intervals) < 2:
            return np.nan
        
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        return rmssd
    
    def compute_ln_rmssd(self, rr_intervals: np.ndarray) -> float:
        """
        Compute natural log of RMSSD for better normality.
        
        Args:
            rr_intervals: Clean RR intervals in milliseconds
            
        Returns:
            ln(RMSSD)
        """
        rmssd = self.compute_rmssd(rr_intervals)
        if np.isnan(rmssd) or rmssd <= 0:
            return np.nan
        return np.log(rmssd)
    
    def compute_windowed_rmssd(self, rr_intervals: np.ndarray, 
                               rr_times: np.ndarray) -> pd.DataFrame:
        """
        Compute RMSSD over sliding windows.
        
        Args:
            rr_intervals: Clean RR intervals (ms)
            rr_times: Time stamps for each RR interval (seconds from start)
            
        Returns:
            DataFrame with columns: window_start, window_end, rmssd, ln_rmssd, n_beats
        """
        if len(rr_intervals) < 2:
            logger.warning("Not enough RR intervals for windowed RMSSD")
            return pd.DataFrame(columns=['window_start', 'window_end', 'rmssd', 'ln_rmssd', 'n_beats'])
        
        results = []
        
        start_time = rr_times[0]
        end_time = rr_times[-1]
        
        current_start = start_time
        step_size = self.window_size_sec - self.overlap_sec
        
        while current_start + self.window_size_sec <= end_time:
            current_end = current_start + self.window_size_sec
            
            # Select RR intervals in this window
            mask = (rr_times >= current_start) & (rr_times < current_end)
            window_rr = rr_intervals[mask]
            
            if len(window_rr) >= 2:
                rmssd = self.compute_rmssd(window_rr)
                ln_rmssd = np.log(rmssd) if rmssd > 0 else np.nan
                
                results.append({
                    'window_start': current_start,
                    'window_end': current_end,
                    'rmssd': rmssd,
                    'ln_rmssd': ln_rmssd,
                    'n_beats': len(window_rr)
                })
            
            current_start += step_size
        
        df = pd.DataFrame(results)
        logger.info(f"Computed RMSSD for {len(df)} windows")
        
        return df
    
    def compute_delta_ln_rmssd(self, baseline_rr: np.ndarray, 
                               task_rr: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute effort as change in ln(RMSSD) from baseline to task.
        
        Effort = ln(RMSSD_baseline) - ln(RMSSD_task)
        Higher effort → lower RMSSD during task → positive delta
        
        Args:
            baseline_rr: RR intervals during baseline/rest period
            task_rr: RR intervals during task/exercise period
            
        Returns:
            effort_label: ΔlnRMSSD (baseline - task)
            baseline_ln_rmssd: Baseline ln(RMSSD)
            task_ln_rmssd: Task ln(RMSSD)
        """
        baseline_ln = self.compute_ln_rmssd(baseline_rr)
        task_ln = self.compute_ln_rmssd(task_rr)
        
        if np.isnan(baseline_ln) or np.isnan(task_ln):
            return np.nan, baseline_ln, task_ln
        
        effort_label = baseline_ln - task_ln
        
        logger.info(f"ΔlnRMSSD effort: {effort_label:.4f} "
                   f"(baseline: {baseline_ln:.4f}, task: {task_ln:.4f})")
        
        return effort_label, baseline_ln, task_ln
    
    def compute_recovery_slope(self, exercise_rr: np.ndarray, 
                              recovery_rr: np.ndarray,
                              recovery_times: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute effort as RMSSD recovery slope after exercise.
        
        Effort = recovery slope (higher slope = faster recovery = lower effort)
        We return negative slope so that higher values = higher effort.
        
        Args:
            exercise_rr: RR intervals during exercise (for reference)
            recovery_rr: RR intervals during recovery period
            recovery_times: Time stamps for recovery RR intervals
            
        Returns:
            effort_label: Negative of recovery slope
            exercise_rmssd: RMSSD during exercise
            recovery_slope: Actual slope of RMSSD recovery
        """
        exercise_rmssd = self.compute_rmssd(exercise_rr)
        
        if len(recovery_rr) < 10:
            logger.warning("Not enough recovery data for slope computation")
            return np.nan, exercise_rmssd, np.nan
        
        # Compute RMSSD in recovery windows
        window_size = 30.0  # 30 second windows
        recovery_rmssd_values = []
        recovery_time_values = []
        
        for i in range(0, len(recovery_times) - 10, 10):
            window_mask = (recovery_times >= recovery_times[i]) & \
                         (recovery_times < recovery_times[i] + window_size)
            window_rr = recovery_rr[window_mask]
            
            if len(window_rr) >= 5:
                rmssd = self.compute_rmssd(window_rr)
                if not np.isnan(rmssd):
                    recovery_rmssd_values.append(rmssd)
                    recovery_time_values.append(recovery_times[i])
        
        if len(recovery_rmssd_values) < 3:
            logger.warning("Not enough valid RMSSD values for slope")
            return np.nan, exercise_rmssd, np.nan
        
        # Fit linear slope
        recovery_slope = np.polyfit(recovery_time_values, recovery_rmssd_values, 1)[0]
        
        # Return negative slope as effort (higher slope = better recovery = lower effort)
        effort_label = -recovery_slope
        
        logger.info(f"Recovery slope effort: {effort_label:.4f} "
                   f"(exercise RMSSD: {exercise_rmssd:.2f}, slope: {recovery_slope:.4f})")
        
        return effort_label, exercise_rmssd, recovery_slope
    
    def create_session_labels(self, rr_intervals: np.ndarray, 
                             rr_times: np.ndarray,
                             adl_segments: pd.DataFrame,
                             session_id: str) -> pd.DataFrame:
        """
        Create effort labels for ADL segments in a session.
        
        Args:
            rr_intervals: Clean RR intervals (ms)
            rr_times: Time stamps for RR intervals (seconds)
            adl_segments: DataFrame with ADL timing info (start_time, end_time, adl_id, phase)
            session_id: Identifier for this session
            
        Returns:
            DataFrame with effort labels per ADL
        """
        results = []
        
        for idx, adl_row in adl_segments.iterrows():
            adl_id = adl_row.get('adl_id', idx)
            start_time = adl_row['start_time']
            end_time = adl_row['end_time']
            phase = adl_row.get('phase', 'task')  # baseline, task, or recovery
            
            # Extract RR intervals for this segment
            mask = (rr_times >= start_time) & (rr_times < end_time)
            segment_rr = rr_intervals[mask]
            
            if len(segment_rr) >= 5:
                rmssd = self.compute_rmssd(segment_rr)
                ln_rmssd = self.compute_ln_rmssd(segment_rr)
                
                results.append({
                    'session_id': session_id,
                    'adl_id': adl_id,
                    'phase': phase,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_sec': end_time - start_time,
                    'n_beats': len(segment_rr),
                    'rmssd': rmssd,
                    'ln_rmssd': ln_rmssd,
                    'mean_rr': np.mean(segment_rr),
                    'std_rr': np.std(segment_rr)
                })
            else:
                logger.warning(f"Insufficient RR data for ADL {adl_id} ({len(segment_rr)} beats)")
        
        df = pd.DataFrame(results)
        logger.info(f"Created labels for {len(df)} ADL segments in session {session_id}")
        
        return df
