"""
Extract HRV Recovery Labels for All Working Subjects

Processes 4 subjects with overlapping PPG+ADL data:
- parsingsim3/sim_elderly3 (32 bouts)
- parsingsim3/sim_healthy3 (31 bouts)
- parsingsim4/sim_elderly4 (27 bouts)
- parsingsim5/sim_severe5 (28 bouts)

Total: ~98 valid training samples
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
from hrv_recovery_pipeline.module4_labels import compute_bout_labels
from hrv_recovery_pipeline.module5_features import aggregate_windowed_features, extract_hr_from_ibi
from hrv_recovery_pipeline.module6_training import evaluate_models
from sklearn.feature_selection import SelectKBest, f_regression

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("/Users/pascalschlegel/data/interim")
TIMEZONE_OFFSET_SEC = 8 * 3600
FS_PPG = 128

# Subjects with overlapping PPG and ADL data
SUBJECTS = [
    ("parsingsim3", "sim_elderly3", "2025-12-04.csv", "ADLs_1.csv"),
    ("parsingsim3", "sim_healthy3", "2025-12-04.csv.gz", "ADLs_1.csv.gz"),
    ("parsingsim4", "sim_elderly4", "2025-12-05.csv.gz", "ADLs_1.csv.gz"),
    ("parsingsim5", "sim_severe5", "2025-12-05.csv.gz", "ADLs_1.csv.gz"),
]


def extract_hrv_labels(project, subject_name, ppg_file, adl_file):
    """Extract HRV recovery labels for one subject"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {project}/{subject_name}")
    logger.info(f"{'='*70}")
    
    subject_dir = DATA_ROOT / project / subject_name
    ppg_path = subject_dir / "corsano_wrist_ppg2_green_6" / ppg_file
    adl_path = subject_dir / "scai_app" / adl_file
    features_path = subject_dir / "effort_estimation_output" / f"{project}_{subject_name}" / "fused_features_10.0s.csv"
    
    if not ppg_path.exists():
        logger.error(f"  ✗ PPG not found: {ppg_path}")
        return None
    if not adl_path.exists():
        logger.error(f"  ✗ ADL not found: {adl_path}")
        return None
    if not features_path.exists():
        logger.warning(f"  ⚠ Features not found: {features_path}")
        logger.warning(f"  → Run pipeline first for this subject")
        return None
    
    try:
        # Module 1: Extract IBIs
        logger.info("Module 1: Extracting IBIs...")
        compression = 'gzip' if ppg_path.suffix == '.gz' else None
        ppg_df = pd.read_csv(ppg_path, compression=compression)
        
        ibi_df = extract_ibi_timeseries(
            ppg_df,
            value_col='value',
            time_col='time',
            fs=FS_PPG,
            distance_ms=300
        )
        
        # Apply timezone correction ONCE to IBI timestamps
        ibi_df['t'] += TIMEZONE_OFFSET_SEC
        logger.info(f"  ✓ {len(ibi_df)} IBIs extracted")
        
        # Module 2: Compute RMSSD windows (timestamps inherit from IBI, don't correct again)
        logger.info("Module 2: Computing RMSSD...")
        rmssd_df = compute_rmssd_windows(
            ibi_df,
            window_len_sec=60.0,
            step_sec=10.0,
            min_beats=10
        )
        
        valid_count = rmssd_df['rmssd'].notna().sum()
        logger.info(f"  ✓ {len(rmssd_df)} windows ({valid_count} valid)")
        
        # Module 3: Parse ADL bouts
        logger.info("Module 3: Parsing ADL bouts...")
        bouts_df = parse_adl_intervals(adl_path, format='auto')
        
        if bouts_df.empty:
            logger.error("  ✗ No bouts parsed")
            return None
        
        if 'bout_id' not in bouts_df.columns:
            bouts_df['bout_id'] = range(len(bouts_df))
        
        logger.info(f"  ✓ {len(bouts_df)} bouts parsed")
        
        # Module 4: Compute HRV recovery labels
        logger.info("Module 4: Computing HRV recovery labels...")
        labels_df = compute_bout_labels(
            rmssd_df,
            bouts_df,
            label_method="delta",
            recovery_end_window_sec=30.0,
            recovery_start_sec=10.0,
            recovery_end_sec=70.0,
            min_recovery_windows=2
        )
        
        valid_labels = labels_df[labels_df['qc_ok']].copy()
        logger.info(f"  ✓ {len(valid_labels)}/{len(labels_df)} valid labels ({100*len(valid_labels)/len(labels_df):.0f}%)")
        
        # Module 5: Load existing features and aggregate per bout
        logger.info("Module 5: Aggregating features...")
        features_df = pd.read_csv(features_path)
        
        # Apply timezone correction to features
        features_df['t_start'] += TIMEZONE_OFFSET_SEC
        features_df['t_center'] += TIMEZONE_OFFSET_SEC
        features_df['t_end'] += TIMEZONE_OFFSET_SEC
        
        bout_features = aggregate_windowed_features(
            features_df,
            bouts_df,
            time_col="t_center",
            method="mean",
            prefix=""
        )
        
        # Add HR features during effort
        hr_features_list = []
        for _, bout in bouts_df.iterrows():
            hr_feats = extract_hr_from_ibi(ibi_df, bout['t_start'], bout['t_end'])
            hr_feats['bout_id'] = bout['bout_id']
            hr_features_list.append(hr_feats)
        
        hr_features_df = pd.DataFrame(hr_features_list)
        bout_features = bout_features.merge(hr_features_df, on='bout_id', how='left')
        
        logger.info(f"  ✓ {len(bout_features.columns)-1} features per bout")
        
        # Build model table
        model_table = bouts_df[['bout_id', 't_start', 't_end', 'duration_sec', 'task_name']].merge(
            bout_features, on='bout_id', how='left'
        ).merge(
            labels_df[['bout_id', 'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'qc_ok']],
            on='bout_id', how='left'
        )
        
        if 'effort' in bouts_df.columns:
            model_table = model_table.merge(
                bouts_df[['bout_id', 'effort']], on='bout_id', how='left'
            )
        
        model_table['subject_id'] = f"{project}_{subject_name}"
        model_table = model_table[model_table['qc_ok'] == True].copy()
        
        logger.info(f"  ✓ {len(model_table)} valid rows in model table")
        
        return model_table
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    logger.info("="*70)
    logger.info("MULTI-SUBJECT HRV RECOVERY PIPELINE")
    logger.info("="*70)
    logger.info(f"\nProcessing {len(SUBJECTS)} subjects with overlapping PPG+ADL data")
    
    all_tables = []
    
    for project, subject, ppg_file, adl_file in SUBJECTS:
        model_table = extract_hrv_labels(project, subject, ppg_file, adl_file)
        if model_table is not None and len(model_table) > 0:
            all_tables.append(model_table)
    
    if len(all_tables) == 0:
        logger.error("\n✗ No subjects processed successfully!")
        return
    
    # Combine all tables
    logger.info("\n" + "="*70)
    logger.info("COMBINING ALL SUBJECTS")
    logger.info("="*70)
    
    combined_table = pd.concat(all_tables, ignore_index=True)
    
    logger.info(f"\n✓ Combined dataset:")
    logger.info(f"  Total rows: {len(combined_table)}")
    logger.info(f"  Subjects: {combined_table['subject_id'].nunique()}")
    logger.info(f"  Feature columns: {len(combined_table.columns)}")
    
    for subj_id, count in combined_table['subject_id'].value_counts().items():
        logger.info(f"    {subj_id}: {count} bouts")
    
    # Save
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    combined_path = output_dir / "hrv_recovery_all_subjects.csv"
    combined_table.to_csv(combined_path, index=False)
    logger.info(f"\n✓ Saved: {combined_path}")
    
    # Feature selection
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION")
    logger.info("="*70)
    
    exclude_cols = [
        'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
        'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
        'qc_ok', 'effort', 'subject_id'
    ]
    
    feature_cols = [c for c in combined_table.columns if c not in exclude_cols]
    
    X_df = combined_table[feature_cols].apply(pd.to_numeric, errors='coerce')
    X_df = X_df.fillna(X_df.median()).fillna(0)
    X = X_df.values.astype(np.float64)
    y = combined_table['delta_rmssd'].values
    
    # Feature selection with SelectKBest
    logger.info(f"\n✓ Feature Selection (SelectKBest with F-test)")
    logger.info(f"  Initial features: {len(feature_cols)}")
    
    K = min(15, len(feature_cols))
    selector = SelectKBest(f_regression, k=K)
    selector.fit(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    
    logger.info(f"\n✓ Selected top {K} features:")
    for i, feat in enumerate(selected_features, 1):
        score = selector.scores_[selected_indices[i-1]]
        logger.info(f"  {i}. {feat} (F-score: {score:.2f})")
    
    keep_cols = exclude_cols + selected_features
    reduced_table = combined_table[[c for c in keep_cols if c in combined_table.columns]].copy()
    
    reduced_path = output_dir / "hrv_recovery_reduced.csv"
    reduced_table.to_csv(reduced_path, index=False)
    logger.info(f"\n✓ Saved reduced table: {reduced_path}")
    
    # Train models
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODELS")
    logger.info("="*70)
    
    results = evaluate_models(
        reduced_table,
        target_col="delta_rmssd",
        output_dir=output_dir
    )
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    
    logger.info(f"\nDataset:")
    logger.info(f"  Samples: {results['n_samples']}")
    logger.info(f"  Features: {results['n_features']}")
    logger.info(f"  Ratio: 1:{results['n_samples']/results['n_features']:.1f}")
    logger.info(f"  Subjects: {len(all_tables)}")
    
    if results.get('elasticnet'):
        metrics = results['elasticnet']['metrics']
        logger.info(f"\nElasticNet:")
        logger.info(f"  Test MAE: {metrics['mae_test']:.4f}")
        logger.info(f"  Test R²:  {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            logger.info(f"  Pearson r: {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
    
    if results.get('xgboost'):
        metrics = results['xgboost']['metrics']
        logger.info(f"\nXGBoost:")
        logger.info(f"  Test MAE: {metrics['mae_test']:.4f}")
        logger.info(f"  Test R²:  {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            logger.info(f"  Pearson r: {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
    
    logger.info("\n" + "="*70)
    logger.info("✓ MULTI-SUBJECT PIPELINE COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
