#!/usr/bin/env python3
"""
Run HRV recovery pipeline for all 9 datasets.

Generates config files and runs pipeline for:
- parsingsim3: sim_healthy3, sim_elderly3, sim_severe3
- parsingsim4: sim_healthy4, sim_elderly4, sim_severe4
- parsingsim5: sim_healthy5, sim_elderly5, sim_severe5
"""

import sys
import subprocess
import yaml
from pathlib import Path


# Base configuration template
BASE_CONFIG = {
    'dataset': {
        'name': 'PLACEHOLDER',
        'base_path': 'PLACEHOLDER',
        'imu_path': 'corsano_wrist_acc_16/2025-MM-DD.csv',
        'ppg_green_path': 'corsano_wrist_ppg2_green_6/2025-MM-DD.csv',
        'ppg_infra_path': 'corsano_wrist_ppg2_infra_red_22/2025-MM-DD.csv',
        'ppg_red_path': 'corsano_wrist_ppg2_red_30/2025-MM-DD.csv',
        'eda_path': 'corsano_wrist_eda_8/2025-MM-DD.csv',
        'adl_path': 'ADLs_1.csv.gz',
        'rr_path': 'vivalnk_vv330_heart_rate/data_1.csv.gz',
    },
    'output': {
        'base_dir': 'PLACEHOLDER',
        'run_name': 'PLACEHOLDER',
    },
    'preprocessing': {
        'imu_fs': 50,
        'ppg_fs': 32,
        'eda_fs': 8,
    },
    'windowing': {
        'window_length_s': 10.0,
        'overlap_percent': 70,
    },
    'targets': {
        'adl_path': 'ADLs_1.csv.gz',
        'adl_offset_hours': 0,
        'rr_path': 'vivalnk_vv330_heart_rate/data_1.csv.gz',
        'hr_col': 'hr',
        'compute_hrv_recovery': True,
    },
    'feature_selection': {
        'correlation_threshold': 0.90,
        'top_n': 100,
    },
}


# Dataset specifications
DATASETS = [
    {'patient': 'parsingsim3', 'condition': 'sim_healthy3', 'date': '2025-12-04'},
    {'patient': 'parsingsim3', 'condition': 'sim_elderly3', 'date': '2025-12-04'},
    {'patient': 'parsingsim3', 'condition': 'sim_severe3', 'date': '2025-12-04'},
    {'patient': 'parsingsim4', 'condition': 'sim_healthy4', 'date': '2025-12-04'},
    {'patient': 'parsingsim4', 'condition': 'sim_elderly4', 'date': '2025-12-04'},
    {'patient': 'parsingsim4', 'condition': 'sim_severe4', 'date': '2025-12-04'},
    {'patient': 'parsingsim5', 'condition': 'sim_healthy5', 'date': '2025-12-05'},
    {'patient': 'parsingsim5', 'condition': 'sim_elderly5', 'date': '2025-12-05'},
    {'patient': 'parsingsim5', 'condition': 'sim_severe5', 'date': '2025-12-05'},
]


def create_config(patient, condition, date):
    """Create config file for a dataset."""
    config = BASE_CONFIG.copy()
    
    # Update paths
    base_path = f'/Users/pascalschlegel/data/interim/{patient}/{condition}'
    
    config['dataset']['name'] = f'{patient}_{condition}'
    config['dataset']['base_path'] = base_path
    
    # Update dates in all paths
    for key in config['dataset']:
        if isinstance(config['dataset'][key], str) and '2025-MM-DD' in config['dataset'][key]:
            config['dataset'][key] = config['dataset'][key].replace('2025-MM-DD', date)
    
    config['output']['base_dir'] = f'{base_path}/effort_estimation_output'
    config['output']['run_name'] = f'{patient}_{condition}'
    
    # Save config
    config_file = Path('config') / f'pipeline_{condition}.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Created {config_file}")
    return config_file


def run_pipeline(config_file, patient, condition):
    """Run pipeline for a dataset."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {patient}/{condition}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            ['.venv/bin/python', 'run_pipeline.py', str(config_file)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ SUCCESS: {patient}/{condition}")
            return True
        else:
            print(f"  ✗ FAILED: {patient}/{condition}")
            print(f"    Error: {result.stderr[-500:]}")  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT: {patient}/{condition} (>10 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {patient}/{condition} - {str(e)}")
        return False


def main():
    """Main execution."""
    print(f"{'='*70}")
    print(f"BATCH PIPELINE EXECUTION")
    print(f"{'='*70}")
    print(f"\nProcessing {len(DATASETS)} datasets:")
    for ds in DATASETS:
        print(f"  - {ds['patient']}/{ds['condition']}")
    
    # Create all configs
    print(f"\n{'='*70}")
    print(f"CREATING CONFIG FILES")
    print(f"{'='*70}")
    
    configs = []
    for ds in DATASETS:
        config_file = create_config(ds['patient'], ds['condition'], ds['date'])
        configs.append((config_file, ds['patient'], ds['condition']))
    
    # Run pipelines
    print(f"\n{'='*70}")
    print(f"RUNNING PIPELINES")
    print(f"{'='*70}")
    
    results = []
    for config_file, patient, condition in configs:
        success = run_pipeline(config_file, patient, condition)
        results.append((patient, condition, success))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    print(f"\nTotal: {len(results)} datasets")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed datasets:")
        for patient, condition, success in results:
            if not success:
                print(f"  - {patient}/{condition}")
    
    if successful > 0:
        print(f"\n{'='*70}")
        print(f"NEXT STEP:")
        print(f"{'='*70}")
        print(f"Run: .venv/bin/python train_hrv_recovery_multidataset.py")
        print(f"This will combine all {successful} datasets and train a model")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
