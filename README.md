# effort-estimator
An Effort estimator based on physiological features extracted from wearable and nearable sensors.
The regressor takes in sensor data and puts out an effort score based on the Borg CR10 scale.

## Overview
The pipeline includes:
* Preprocessing
* Feature computation and extraction based on Tifex-py
* Redundant feature removal (overlap and non-numerics)
* Feature validation using PCA
* PC loading analysis
* Feature selection
* Regressor fitting
* Regressor evaluation

## Input
Raw gzipped .csv sensor data files.

## Output
The regressor returns a number based on the Borg CR10 scale of perceived exertion.