"""
Test Module 6: Train baseline models to predict HRV recovery
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrv_recovery_pipeline.module6_training import evaluate_models, prepare_model_data

if __name__ == "__main__":
    # Load model table with valid labels
    model_table_path = Path("./output/test_model_table_valid.csv")
    model_table = pd.read_csv(model_table_path)
    
    print("="*70)
    print("HRV RECOVERY ESTIMATION - Module 6: Training")
    print("="*70)
    
    print(f"\n✓ Loaded model table: {len(model_table)} rows, {len(model_table.columns)} columns")
    print(f"  Target: delta_rmssd")
    print(f"  Target range: [{model_table['delta_rmssd'].min():.4f}, {model_table['delta_rmssd'].max():.4f}]")
    print(f"  Target mean ± std: {model_table['delta_rmssd'].mean():.4f} ± {model_table['delta_rmssd'].std():.4f}")
    
    # Check feature completeness
    exclude_cols = [
        'bout_id', 't_start', 't_end', 'duration_sec', 'task_name',
        'rmssd_end', 'rmssd_recovery', 'delta_rmssd', 'recovery_slope',
        'qc_ok', 'effort'
    ]
    feature_cols = [c for c in model_table.columns if c not in exclude_cols]
    print(f"\n  Feature columns: {len(feature_cols)}")
    
    # Warning if too few samples
    if len(model_table) < 20:
        print(f"\n⚠ WARNING: Only {len(model_table)} samples - results may not be reliable")
        print(f"  Consider collecting more data or using simpler models")
    
    # Prepare data
    print("\n" + "-"*70)
    print("Preparing data...")
    print("-"*70)
    
    X, y, feature_names = prepare_model_data(
        model_table,
        target_col="delta_rmssd",
        drop_qc_fail=True
    )
    
    print(f"✓ Data prepared: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Train models
    print("\n" + "-"*70)
    print("Training models...")
    print("-"*70)
    
    output_dir = Path("./output")
    results = evaluate_models(
        model_table,
        target_col="delta_rmssd",
        output_dir=output_dir
    )
    
    # Display results
    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    
    print(f"\n✓ Trained on {results['n_samples']} samples with {results['n_features']} features")
    
    if results.get('elasticnet'):
        print("\n--- ElasticNet ---")
        metrics = results['elasticnet']['metrics']
        print(f"  Train MAE:  {metrics['mae_train']:.4f}")
        print(f"  Test MAE:   {metrics['mae_test']:.4f}")
        print(f"  Train R²:   {metrics['r2_train']:.4f}")
        print(f"  Test R²:    {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            print(f"  Pearson r:  {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
        
        # Show prediction scatter
        if len(results['elasticnet']['y_test']) > 0:
            y_test = results['elasticnet']['y_test']
            y_pred = results['elasticnet']['model'].predict(results['elasticnet']['X_test'])
            print(f"\n  Sample predictions (actual vs predicted):")
            for i in range(min(5, len(y_test))):
                print(f"    {y_test[i]:.4f} → {y_pred[i]:.4f} (error: {abs(y_test[i]-y_pred[i]):.4f})")
    
    if results.get('xgboost'):
        print("\n--- XGBoost ---")
        metrics = results['xgboost']['metrics']
        print(f"  Train MAE:  {metrics['mae_train']:.4f}")
        print(f"  Test MAE:   {metrics['mae_test']:.4f}")
        print(f"  Train R²:   {metrics['r2_train']:.4f}")
        print(f"  Test R²:    {metrics['r2_test']:.4f}")
        if 'r_pearson' in metrics:
            print(f"  Pearson r:  {metrics['r_pearson']:.4f} (p={metrics['p_pearson']:.4f})")
        
        # Feature importance (top 10)
        if hasattr(results['xgboost']['model'], 'feature_importances_'):
            importances = results['xgboost']['model'].feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print(f"\n  Top 10 feature importances:")
            for idx in top_indices:
                print(f"    {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Model comparison
    if results.get('elasticnet') and results.get('xgboost'):
        print("\n" + "-"*70)
        print("Model Comparison (Test Set)")
        print("-"*70)
        
        en_mae = results['elasticnet']['metrics']['mae_test']
        xgb_mae = results['xgboost']['metrics']['mae_test']
        en_r2 = results['elasticnet']['metrics']['r2_test']
        xgb_r2 = results['xgboost']['metrics']['r2_test']
        
        print(f"  MAE:  ElasticNet={en_mae:.4f}, XGBoost={xgb_mae:.4f}")
        print(f"  R²:   ElasticNet={en_r2:.4f}, XGBoost={xgb_r2:.4f}")
        
        if xgb_mae < en_mae:
            print(f"\n  → XGBoost performs better (MAE {(en_mae-xgb_mae)/en_mae*100:.1f}% lower)")
        else:
            print(f"\n  → ElasticNet performs better (MAE {(xgb_mae-en_mae)/xgb_mae*100:.1f}% lower)")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    baseline_std = model_table['delta_rmssd'].std()
    print(f"\nBaseline (predicting mean): MAE ≈ {0.8*baseline_std:.4f} (std={baseline_std:.4f})")
    
    if results.get('elasticnet'):
        mae = results['elasticnet']['metrics']['mae_test']
        r2 = results['elasticnet']['metrics']['r2_test']
        
        if mae < 0.8 * baseline_std:
            print(f"✓ Models beat baseline (MAE={mae:.4f} < {0.8*baseline_std:.4f})")
        else:
            print(f"⚠ Models struggling to beat baseline (MAE={mae:.4f} vs {0.8*baseline_std:.4f})")
        
        if r2 > 0.1:
            print(f"✓ Models explain {r2*100:.1f}% of variance")
        else:
            print(f"⚠ Low R² ({r2:.4f}) - consider more data or better features")
    
    # Limitations
    print(f"\n⚠ Limitations:")
    print(f"  • Only {results['n_samples']} training samples")
    print(f"  • Single-subject data (elderly3)")
    print(f"  • Timezone correction applied manually")
    print(f"  • Small test set (20% of {results['n_samples']} samples)")
    
    print(f"\n✓ Training summary saved to: {output_dir / 'training_summary_delta_rmssd.txt'}")
    print(f"\n✓ Module 6 complete!")
