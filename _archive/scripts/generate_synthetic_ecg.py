#!/usr/bin/env python
"""
Generate synthetic ECG data for testing the ECG → RMSSD pipeline.

Usage:
    python scripts/generate_synthetic_ecg.py \\
        --output-dir data/raw \\
        --duration 300 \\
        --sampling-rate 250 \\
        --sessions 3
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_ecg(
    duration_s=300,
    sampling_rate_hz=250,
    heart_rate_bpm=70,
    heart_rate_variability_bpm=5,
    noise_level=0.05,
    session_id="session_001"
):
    """
    Generate synthetic ECG with realistic R-peak morphology.
    
    Parameters
    ----------
    duration_s : float
        Duration of ECG recording (seconds)
    sampling_rate_hz : int
        Sampling rate (Hz)
    heart_rate_bpm : float
        Mean heart rate (beats per minute)
    heart_rate_variability_bpm : float
        Standard deviation of heart rate (bpm)
    noise_level : float
        Gaussian noise standard deviation (as fraction of signal)
    session_id : str
        Session identifier
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 't' (seconds) and 'ecg' (normalized voltage)
    """
    t = np.arange(0, duration_s, 1/sampling_rate_hz)
    n_samples = len(t)
    
    # Generate RR intervals with physiological variation
    # HR ± HRV with some autocorrelation (realistic)
    hr_trace = heart_rate_bpm + heart_rate_variability_bpm * np.sin(2 * np.pi * t / 60)  # ~1 min cycle
    hr_trace += np.random.normal(0, heart_rate_variability_bpm/3, len(t))  # Add noise
    hr_trace = np.clip(hr_trace, 40, 160)  # Keep realistic
    
    # Convert HR to RR intervals
    rr_intervals_s = 60 / hr_trace
    peak_times = np.concatenate([[0], np.cumsum(rr_intervals_s[:-1])])
    peak_times = peak_times[peak_times < duration_s]
    
    # Generate ECG with R-peak morphology
    ecg = np.zeros(n_samples)
    peak_indices = np.round(peak_times * sampling_rate_hz).astype(int)
    
    for peak_idx in peak_indices:
        if peak_idx < n_samples:
            # R-peak: positive deflection (QRS complex approximation)
            window_len = 30  # samples (~120 ms at 250 Hz)
            start_idx = max(0, peak_idx - window_len // 3)
            end_idx = min(n_samples, peak_idx + 2 * window_len // 3)
            actual_window_len = end_idx - start_idx
            
            if actual_window_len > 1:
                # Gaussian-like peak
                peak_window = np.exp(-(np.arange(actual_window_len) - (peak_idx - start_idx))**2 / 50)
                ecg[start_idx:end_idx] += peak_window * 100
    
    # Add baseline wandering (low-freq component)
    baseline = 20 * np.sin(2 * np.pi * t / 30)  # ~30 sec cycle
    ecg += baseline
    
    # Add noise
    ecg += np.random.normal(0, noise_level * ecg.max(), n_samples)
    
    # Normalize to 0-100 mV range
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-6) * 100
    
    df = pd.DataFrame({
        't': t,
        'ecg': ecg
    })
    
    logger.info(f"Generated ECG: {len(df)} samples ({duration_s}s @ {sampling_rate_hz}Hz)")
    logger.info(f"  Mean HR: {np.mean(hr_trace):.1f} bpm")
    logger.info(f"  HR range: {hr_trace.min():.1f} - {hr_trace.max():.1f} bpm")
    logger.info(f"  ECG range: {ecg.min():.1f} - {ecg.max():.1f} mV")
    logger.info(f"  Detected peaks: {len(peak_indices)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ECG for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 session (5 minutes)
  python scripts/generate_synthetic_ecg.py --output-dir data/raw --duration 300

  # Generate 3 sessions for testing
  python scripts/generate_synthetic_ecg.py --output-dir data/raw --sessions 3

  # Custom heart rate and duration
  python scripts/generate_synthetic_ecg.py \\
    --output-dir data/raw \\
    --duration 600 \\
    --heart-rate 85 \\
    --sessions 1
        """
    )
    
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory for ECG CSVs [default: data/raw]')
    parser.add_argument('--duration', type=float, default=300,
                       help='ECG duration (seconds) [default: 300]')
    parser.add_argument('--sampling-rate', type=int, default=250,
                       help='Sampling rate (Hz) [default: 250]')
    parser.add_argument('--heart-rate', type=float, default=70,
                       help='Mean heart rate (bpm) [default: 70]')
    parser.add_argument('--heart-rate-variability', type=float, default=5,
                       help='Heart rate std dev (bpm) [default: 5]')
    parser.add_argument('--sessions', type=int, default=1,
                       help='Number of sessions to generate [default: 1]')
    parser.add_argument('--noise', type=float, default=0.05,
                       help='Noise level [default: 0.05]')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.sessions} synthetic ECG session(s)")
    logger.info(f"  Duration: {args.duration}s")
    logger.info(f"  Sampling: {args.sampling_rate}Hz")
    logger.info(f"  Heart rate: {args.heart_rate} ± {args.heart_rate_variability} bpm")
    logger.info(f"  Output: {output_dir}")
    logger.info("")
    
    for session_num in range(args.sessions):
        session_id = f"session_{session_num + 1:03d}"
        
        # Add some variation between sessions
        hr = args.heart_rate + np.random.normal(0, 5)  # ±5 bpm variation
        
        df_ecg = generate_synthetic_ecg(
            duration_s=args.duration,
            sampling_rate_hz=args.sampling_rate,
            heart_rate_bpm=hr,
            heart_rate_variability_bpm=args.heart_rate_variability,
            noise_level=args.noise,
            session_id=session_id
        )
        
        output_file = output_dir / f"ecg_{session_id}.csv"
        df_ecg.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}\n")
    
    logger.info("="*70)
    logger.info("NEXT STEPS: Run Stage 1 (ECG → RR)")
    logger.info("="*70)
    
    first_session_file = output_dir / "ecg_session_001.csv"
    if first_session_file.exists():
        logger.info(f"\npython scripts/ecg_to_rr.py \\")
        logger.info(f"  --ecg-csv {first_session_file} \\")
        logger.info(f"  --output-rr data/interim/rr_session_001.csv \\")
        logger.info(f"  --sampling-rate {args.sampling_rate} \\")
        logger.info(f"  --session-id session_001 \\")
        logger.info(f"  --verbose")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
