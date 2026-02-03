#!/usr/bin/env python3
"""
DATA-DRIVEN EFFORT ESTIMATION PIPELINE
=======================================

This pipeline automatically discovers which features/modalities work best
for the given data. It doesn't assume IMU is best - it tests and finds out.

Approach:
1. Test each modality separately (IMU, PPG, EDA)
2. Test combinations with regularization
3. Automatically select the best performing approach
4. Train final model with winning configuration

This works for ANY data - if subjects 6-20 have better PPG quality,
it will automatically discover that and use PPG instead.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import json
warnings.filterwarnings('ignore')


class DataDrivenEffortPipeline:
    """
    Automatically finds the best features and model for effort estimation.
    No hardcoded assumptions - learns from the data.
    
    Two modes:
    1. Modality-level: Test IMU vs PPG vs EDA vs combinations
    2. Granular: Find optimal individual features (slower but more precise)
    """
    
    def __init__(self, cal_fraction=0.2, random_state=42, granular=False):
        self.cal_fraction = cal_fraction
        self.random_state = random_state
        self.granular = granular  # If True, do individual feature selection
        self.best_config = None
        self.best_features = None
        self.results_log = []
        
    def get_feature_sets(self, df):
        """Extract different feature sets from data"""
        all_cols = df.columns.tolist()
        
        # Identify features by modality
        imu_features = [c for c in all_cols if 'acc' in c.lower() or 'gyro' in c.lower()]
        ppg_features = [c for c in all_cols if 'ppg' in c.lower()]
        eda_features = [c for c in all_cols if 'eda' in c.lower()]
        
        # All numeric features (excluding metadata)
        skip_cols = {'t_center', 't_start', 't_end', 'borg', 'subject', 'activity_label', 
                     'window_id', 'n_samples', 'win_sec', 'modality', 'valid'}
        all_features = [c for c in all_cols 
                        if c not in skip_cols 
                        and not c.startswith('Unnamed')
                        and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        feature_sets = {
            'IMU': imu_features,
            'PPG': ppg_features,
            'EDA': eda_features,
            'IMU+PPG': imu_features + ppg_features,
            'IMU+EDA': imu_features + eda_features,
            'PPG+EDA': ppg_features + eda_features,
            'ALL': all_features,
        }
        
        # Remove empty sets
        feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}
        
        return feature_sets
    
    def get_models(self):
        """Define models to test"""
        return {
            'Ridge': Ridge(alpha=1.0),
            'Ridge_strong': Ridge(alpha=10.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'RF_d4': RandomForestRegressor(n_estimators=100, max_depth=4, 
                                           random_state=self.random_state, n_jobs=-1),
            'RF_d6': RandomForestRegressor(n_estimators=100, max_depth=6, 
                                           random_state=self.random_state, n_jobs=-1),
            'GB_d4': GradientBoostingRegressor(n_estimators=100, max_depth=4, 
                                               random_state=self.random_state),
            'SVR': SVR(kernel='rbf', C=1.0),
        }
    
    def evaluate_config(self, df, features, model):
        """
        Evaluate a feature+model configuration using LOSO with calibration.
        Returns per-subject r (the honest metric).
        """
        np.random.seed(self.random_state)
        
        subjects = sorted(df['subject'].unique())
        
        # Clean features
        valid_features = [f for f in features if f in df.columns 
                          and df[f].notna().mean() > 0.5 
                          and df[f].std() > 1e-10]
        
        if len(valid_features) == 0:
            return {'per_subject_r': float('nan'), 'per_subject': {}}
        
        per_subject = {}
        all_preds, all_true = [], []
        
        for test_sub in subjects:
            train_df = df[df['subject'] != test_sub]
            test_df = df[df['subject'] == test_sub]
            
            n_test = len(test_df)
            n_cal = max(5, int(n_test * self.cal_fraction))
            idx = np.random.permutation(n_test)
            cal_idx = idx[:n_cal]
            eval_idx = idx[n_cal:]
            
            if len(eval_idx) < 5:
                continue
            
            X_train = train_df[valid_features].values
            y_train = train_df['borg'].values
            X_test = test_df[valid_features].values
            y_test = test_df['borg'].values
            
            # Impute + Scale
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(imputer.fit_transform(X_train))
            X_test = scaler.transform(imputer.transform(X_test))
            
            # Clone model for fresh training
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            # Calibrate
            cal_offset = y_test[cal_idx].mean() - y_pred[cal_idx].mean()
            y_pred_cal = y_pred + cal_offset
            
            # Evaluate
            r, _ = pearsonr(y_pred_cal[eval_idx], y_test[eval_idx])
            mae = np.mean(np.abs(y_pred_cal[eval_idx] - y_test[eval_idx]))
            
            per_subject[test_sub] = {'r': r, 'mae': mae}
            all_preds.extend(y_pred_cal[eval_idx])
            all_true.extend(y_test[eval_idx])
        
        if len(per_subject) == 0:
            return {'per_subject_r': float('nan'), 'per_subject': {}}
        
        per_subject_r = np.mean([m['r'] for m in per_subject.values()])
        pooled_r, _ = pearsonr(all_preds, all_true) if len(all_preds) > 0 else (float('nan'), 0)
        avg_mae = np.mean([m['mae'] for m in per_subject.values()])
        
        return {
            'per_subject_r': per_subject_r,
            'pooled_r': pooled_r,
            'mae': avg_mae,
            'per_subject': per_subject,
            'n_features': len(valid_features)
        }
    
    def find_best_configuration(self, df, verbose=True):
        """
        Test all combinations of features and models, find the best one.
        This is the data-driven discovery step.
        """
        df = df.dropna(subset=['borg']).copy()
        
        feature_sets = self.get_feature_sets(df)
        models = self.get_models()
        
        if verbose:
            print("="*80)
            print("DATA-DRIVEN CONFIGURATION SEARCH")
            print("="*80)
            print(f"\nTesting {len(feature_sets)} feature sets × {len(models)} models = {len(feature_sets)*len(models)} configurations")
            print(f"\nFeature sets:")
            for name, features in feature_sets.items():
                print(f"  {name}: {len(features)} features")
        
        results = []
        
        for feat_name, features in feature_sets.items():
            for model_name, model in models.items():
                result = self.evaluate_config(df, features, model)
                result['features'] = feat_name
                result['model'] = model_name
                results.append(result)
                
                if verbose and not np.isnan(result['per_subject_r']):
                    print(f"  {feat_name:>10} + {model_name:<12}: per-sub r = {result['per_subject_r']:.3f}")
        
        # Find best by per-subject r
        valid_results = [r for r in results if not np.isnan(r['per_subject_r'])]
        if len(valid_results) == 0:
            raise ValueError("No valid configurations found!")
        
        best = max(valid_results, key=lambda x: x['per_subject_r'])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"BEST CONFIGURATION FOUND:")
            print(f"{'='*80}")
            print(f"  Features: {best['features']} ({best['n_features']} features)")
            print(f"  Model:    {best['model']}")
            print(f"  Per-subject r = {best['per_subject_r']:.3f}")
            print(f"  Pooled r = {best['pooled_r']:.3f}")
            print(f"  MAE = {best['mae']:.2f}")
            print(f"\n  Per-subject breakdown:")
            for subj, metrics in best['per_subject'].items():
                print(f"    {subj}: r = {metrics['r']:.3f}, MAE = {metrics['mae']:.2f}")
        
        self.best_config = best
        self.results_log = results
        
        # If granular mode, also do individual feature selection
        if self.granular:
            self.best_features = self.greedy_feature_selection(df, verbose=verbose)
        else:
            feature_sets = self.get_feature_sets(df)
            self.best_features = feature_sets[best['features']]
        
        return best
    
    def greedy_feature_selection(self, df, max_features=30, verbose=True):
        """
        Greedy forward selection: find best individual features.
        Adds one feature at a time if it improves per-subject r.
        """
        df = df.dropna(subset=['borg']).copy()
        
        # Get all valid features
        skip_cols = {'t_center', 't_start', 't_end', 'borg', 'subject', 'activity_label', 
                     'window_id', 'n_samples', 'win_sec', 'modality', 'valid'}
        all_features = [c for c in df.columns 
                        if c not in skip_cols 
                        and not c.startswith('Unnamed')
                        and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                        and df[c].notna().mean() > 0.5
                        and df[c].std() > 1e-10]
        
        if verbose:
            print(f"\n{'='*80}")
            print("GRANULAR FEATURE SELECTION (Greedy Forward)")
            print(f"{'='*80}")
            print(f"Searching through {len(all_features)} candidate features...")
        
        # Get RF importance to prioritize search order
        X = df[all_features].values
        y = df['borg'].values
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_clean = scaler.fit_transform(imputer.fit_transform(X))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, 
                                   random_state=self.random_state, n_jobs=-1)
        rf.fit(X_clean, y)
        
        importance_order = np.argsort(rf.feature_importances_)[::-1]
        sorted_features = [all_features[i] for i in importance_order]
        
        # Greedy selection
        selected = []
        current_r = 0
        model = RandomForestRegressor(n_estimators=50, max_depth=6, 
                                      random_state=self.random_state, n_jobs=-1)
        
        for feat in sorted_features[:100]:  # Only try top 100 by importance
            if len(selected) >= max_features:
                break
            
            test_features = selected + [feat]
            result = self.evaluate_config(df, test_features, model)
            
            if result['per_subject_r'] > current_r + 0.005:  # Improvement threshold
                selected.append(feat)
                current_r = result['per_subject_r']
                
                if verbose:
                    mod = self._get_modality(feat)
                    print(f"  + {feat:<45} [{mod}] → r = {current_r:.3f}")
        
        if verbose:
            mix = {}
            for f in selected:
                mod = self._get_modality(f)
                mix[mod] = mix.get(mod, 0) + 1
            print(f"\nSelected {len(selected)} features: {mix}")
            print(f"Final per-subject r = {current_r:.3f}")
        
        return selected
    
    def _get_modality(self, feature):
        """Get modality of a feature"""
        if 'acc' in feature.lower() or 'gyro' in feature.lower():
            return 'IMU'
        elif 'ppg' in feature.lower():
            return 'PPG'
        elif 'eda' in feature.lower():
            return 'EDA'
        return 'Other'
    
    def train_final_model(self, df, output_dir=None):
        """
        Train the final model using the best discovered configuration.
        """
        if self.best_config is None:
            raise ValueError("Run find_best_configuration() first!")
        
        df = df.dropna(subset=['borg']).copy()
        models = self.get_models()
        
        # Use granular features if available, otherwise use modality-level
        if self.best_features is not None:
            valid_features = self.best_features
        else:
            feature_sets = self.get_feature_sets(df)
            best_features = feature_sets[self.best_config['features']]
            valid_features = [f for f in best_features if f in df.columns 
                              and df[f].notna().mean() > 0.5 
                              and df[f].std() > 1e-10]
        
        best_model = models[self.best_config['model']]
        
        # Prepare data
        X = df[valid_features].values
        y = df['borg'].values
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_clean = scaler.fit_transform(imputer.fit_transform(X))
        
        # Train final model on all data
        from sklearn.base import clone
        final_model = clone(best_model)
        final_model.fit(X_clean, y)
        
        # Save if output_dir provided
        if output_dir:
            import joblib
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(final_model, output_dir / 'model.joblib')
            joblib.dump(scaler, output_dir / 'scaler.joblib')
            joblib.dump(imputer, output_dir / 'imputer.joblib')
            
            # Save configuration
            config = {
                'features': self.best_config['features'],
                'feature_list': valid_features,
                'model': self.best_config['model'],
                'per_subject_r': self.best_config['per_subject_r'],
                'cal_fraction': self.cal_fraction,
                'granular': self.granular
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\nSaved model to {output_dir}/")
        
        return final_model, scaler, imputer, valid_features


def run_data_driven_pipeline(fused_csv_path, output_dir=None, cal_fraction=0.2):
    """
    Main function to run the data-driven pipeline.
    
    Args:
        fused_csv_path: Path to fused_aligned CSV (or list of paths)
        output_dir: Where to save model (optional)
        cal_fraction: Calibration fraction (default 0.2 = 20%)
    """
    # Load data
    if isinstance(fused_csv_path, list):
        dfs = []
        for i, path in enumerate(fused_csv_path):
            df = pd.read_csv(path)
            if 'subject' not in df.columns:
                df['subject'] = f'subject_{i}'
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.read_csv(fused_csv_path)
    
    print(f"Loaded {len(df_all)} samples from {df_all['subject'].nunique()} subjects")
    
    # Run pipeline
    pipeline = DataDrivenEffortPipeline(cal_fraction=cal_fraction)
    best_config = pipeline.find_best_configuration(df_all, verbose=True)
    
    # Train final model
    if output_dir:
        pipeline.train_final_model(df_all, output_dir)
    
    return pipeline


if __name__ == "__main__":
    import sys
    
    # Example: load from multiple subject files
    all_dfs = []
    for i in [1,2,3,4,5]:
        path = Path(f'/Users/pascalschlegel/data/interim/parsingsim{i}/sim_elderly{i}/effort_estimation_output/elderly_sim_elderly{i}/fused_aligned_5.0s.csv')
        if path.exists():
            df = pd.read_csv(path)
            df['subject'] = f'elderly{i}'
            all_dfs.append(df)
    
    if len(all_dfs) == 0:
        print("No data found!")
        sys.exit(1)
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Check for --granular flag
    granular = '--granular' in sys.argv
    
    print(f"\nMode: {'GRANULAR (individual features)' if granular else 'MODALITY-LEVEL (IMU/PPG/EDA groups)'}")
    
    # Run data-driven pipeline
    pipeline = DataDrivenEffortPipeline(cal_fraction=0.2, granular=granular)
    best = pipeline.find_best_configuration(df_all)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if granular and pipeline.best_features:
        # Show granular feature breakdown
        mix = {}
        for f in pipeline.best_features:
            mod = pipeline._get_modality(f)
            mix[mod] = mix.get(mod, 0) + 1
        
        print(f"""
GRANULAR FEATURE SELECTION RESULT:
──────────────────────────────────
  Selected {len(pipeline.best_features)} individual features
  Modality mix: {mix}
  
  Best features:""")
        for f in pipeline.best_features[:10]:
            mod = pipeline._get_modality(f)
            print(f"    [{mod:>3}] {f}")
        if len(pipeline.best_features) > 10:
            print(f"    ... and {len(pipeline.best_features) - 10} more")
    
    print(f"""
BEST CONFIGURATION:
───────────────────
  Modality group: {best['features']}
  Model: {best['model']}
  Per-subject r = {best['per_subject_r']:.3f}

USAGE:
──────
  # Modality-level (fast):
  python ml/data_driven_pipeline.py
  
  # Granular feature selection (slower, finds optimal feature mix):
  python ml/data_driven_pipeline.py --granular

With new data (subjects 6-20), run again - it will automatically
discover if different features/modalities work better!
""")
    pipeline = DataDrivenEffortPipeline(cal_fraction=0.2)
    best = pipeline.find_best_configuration(df_all)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
The pipeline automatically discovered that:
  → Best features: {best['features']}
  → Best model: {best['model']}
  → Per-subject r = {best['per_subject_r']:.3f}

This is DATA-DRIVEN - with different data (e.g., subjects 6-20), 
it might discover that PPG or EDA works better!

To use with new data:
  pipeline = DataDrivenEffortPipeline()
  best = pipeline.find_best_configuration(your_new_data)
  pipeline.train_final_model(your_new_data, 'output/')
""")
