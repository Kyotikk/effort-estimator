"""
Simplified Multi-Subject HRV Recovery Pipeline

Start with elderly3 (which has all data ready), then show how to extend to others.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add hrv_recovery_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module6_training import evaluate_models
from sklearn.feature_selection import SelectKBest, f_regression

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*70)
    logger.info("SIMPLIFIED MULTI-SUBJECT HRV RECOVERY PIPELINE")
    logger.info("="*70)
    logger.info("\nStarting with elderly3 which has complete data...")
    
    # Load the elderly3 model table we already created
    output_dir = Path("./output")
    elderly3_table_path = output_dir / "test_model_table_valid.csv"
    
    if not elderly3_table_path.exists():
        logger.error(f"elderly3 table not found: {elderly3_table_path}")
        logger.info("\nPlease run: .venv/bin/python test_module6_training.py first")
        return
    
    elderly3_table = pd.read_csv(elderly3_table_path)
    logger.info(f"\n✓ Loaded elderly3 data: {len(elderly3_table)} rows")
    
    # Add subject ID
    elderly3_table['subject_id'] = 'parsingsim3_sim_elderly3'
    
    # For now, this is our multi-subject dataset (just elderly3)
    combined_table = elderly3_table.copy()
    
    logger.info(f"\n✓ Combined dataset:")
    logger.info(f"  Total rows: {len(combined_table)}")
    logger.info(f"  Subjects: {combined_table['subject_id'].nunique()}")
    logger.info(f"  Total features: {len(combined_table.columns)}")
    
    # Save combined table
    combined_path = output_dir / "multisub_model_table_elderly3_only.csv"
    combined_table.to_csv(combined_path, index=False)
    logger.info(f"\n✓ Saved: {combined_path}")
    
    # Feature selection: reduce to top K features
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION")
    logger.info("="*70)
    
    # Prepare data
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
    
    # Select top 10 features using F-test
    K = 10
    selector = SelectKBest(f_regression, k=K)
    selector.fit(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    
    logger.info(f"\n✓ Selected top {K} features:")
    for i, feat in enumerate(selected_features, 1):
        score = selector.scores_[selected_indices[i-1]]
        logger.info(f"  {i}. {feat} (F-score: {score:.2f})")
    
    # Create reduced table
    keep_cols = exclude_cols + selected_features
    reduced_table = combined_table[[c for c in keep_cols if c in combined_table.columns]].copy()
    
    reduced_path = output_dir / "multisub_model_table_reduced_elderly3.csv"
    reduced_table.to_csv(reduced_path, index=False)
    logger.info(f"\n✓ Saved reduced table: {reduced_path}")
    
    # Train final model
    logger.info("\n" + "="*70)
    logger.info("TRAINING WITH FEATURE SELECTION")
    logger.info("="*70)
    
    results = evaluate_models(
        reduced_table,
        target_col="delta_rmssd",
        output_dir=output_dir
    )
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\nDataset:")
    logger.info(f"  Samples: {results['n_samples']}")
    logger.info(f"  Features: {results['n_features']}")
    logger.info(f"  Feature-to-sample ratio: 1:{results['n_samples']/results['n_features']:.1f}")
    
    if results.get('elasticnet'):
        metrics = results['elasticnet']['metrics']
        logger.info(f"\nElasticNet Performance:")
        logger.info(f"  Test MAE: {metrics['mae_test']:.4f}")
        logger.info(f"  Test R²:  {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            logger.info(f"  Pearson r: {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
    
    if results.get('xgboost'):
        metrics = results['xgboost']['metrics']
        logger.info(f"\nXGBoost Performance:")
        logger.info(f"  Test MAE: {metrics['mae_test']:.4f}")
        logger.info(f"  Test R²:  {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            logger.info(f"  Pearson r: {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS TO EXPAND")
    logger.info("="*70)
    logger.info("\nTo add more subjects:")
    logger.info("1. Run existing pipeline on healthy3 and severe3:")
    logger.info("   cd /Users/pascalschlegel/effort-estimator")
    logger.info("   # Modify config to point to healthy3")
    logger.info("   .venv/bin/python run_pipeline.py")
    logger.info("")
    logger.info("2. For each subject, run modules 1-5 to create model tables")
    logger.info("3. Combine all model tables and retrain")
    logger.info("4. Expected improvement with 3x data: ~81 samples vs 27")
    
    logger.info("\n" + "="*70)
    logger.info("✓ PIPELINE COMPLETE (ELDERLY3 BASELINE)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
