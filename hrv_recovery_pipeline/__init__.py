"""
HRV Recovery Estimation Pipeline

Alternative to Borg effort labels: Uses PPG-derived RMSSD recovery as training target.
Effort bouts are defined from ADL intervals or IMU intensity thresholding.
Features X extracted from IMU+EDA+PPG during effort; label y = HRV recovery post-effort.
"""

__version__ = "0.1.0"
