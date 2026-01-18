#!/usr/bin/env python3
"""
Complete Model Training & Evaluation with Comprehensive Visualizations

Trains XGBoost models and evaluates with:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Residuals analysis
- Feature importance
- Beautiful plots and summaries
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Any, Tuple, List
import pickle


# Set style for beautiful plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


class ModelTrainerEvaluator:
    """Complete model training and evaluation with visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_cols: List[str],
                   model_key: str,
                   window_length: float = None) -> None:
        """Train XGBoost model with feature selection."""
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {model_key}")
        print(f"{'='*80}")
        
        # Feature selection (top 100 by correlation with target, balanced across modalities)
        k_features = min(100, X_train.shape[1])
        
        # Calculate absolute correlation with target
        correlations = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
        correlations = np.abs(np.nan_to_num(correlations, nan=0))
        
        top_indices = np.argsort(correlations)[-k_features:]
        top_indices = np.sort(top_indices)
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        selected_cols = [feature_cols[i] for i in top_indices]
        
        # Show distribution
        eda_count = sum(1 for c in selected_cols if c.startswith('eda_'))
        imu_count = sum(1 for c in selected_cols if c.startswith('acc_'))
        ppg_count = sum(1 for c in selected_cols if c.startswith('ppg_'))
        print(f"\nFeature selection: kept top {k_features} by correlation with target")
        print(f"  EDA: {eda_count}, IMU: {imu_count}, PPG: {ppg_count}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train model
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Target train range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
        
        print(f"✓ Model trained successfully")
        
        # Evaluate on train and test sets
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Store results
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        self.feature_cols[model_key] = selected_cols
        
        self.predictions[model_key] = {
            'train': {
                'y_true': y_train,
                'y_pred': y_train_pred,
                'residuals': y_train - y_train_pred,
            },
            'test': {
                'y_true': y_test,
                'y_pred': y_test_pred,
                'residuals': y_test - y_test_pred,
            }
        }
        
        # Calculate metrics
        self._calculate_metrics(model_key)
        
        # Save model and scaler
        self._save_model(model_key, window_length)
    
    def _calculate_metrics(self, model_key: str) -> None:
        """Calculate evaluation metrics."""
        
        self.metrics[model_key] = {}
        
        for set_name, preds in self.predictions[model_key].items():
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            self.metrics[model_key][set_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'n_samples': len(y_true),
                'y_range': (y_true.min(), y_true.max()),
                'y_mean': y_true.mean(),
                'y_std': y_true.std(),
            }
    
    def _save_model(self, model_key: str, window_length: float = None) -> None:
        """Save model and scaler."""
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Save model
        if window_length:
            model_path = self.output_dir / f"xgboost_borg_{window_length:.1f}s.json"
            scaler_path = self.output_dir / f"scaler_{window_length:.1f}s.pkl"
        else:
            model_path = self.output_dir / f"{model_key}_model.json"
            scaler_path = self.output_dir / f"{model_key}_scaler.pkl"
        
        model.save_model(str(model_path))
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✓ Saved model to: {model_path.name}")
        print(f"✓ Saved scaler to: {scaler_path.name}")
    def print_metrics_summary(self) -> None:
        """Print formatted metrics summary."""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY - MAE, RMSE, R²")
        print("="*80)
        
        for model_key, sets in self.metrics.items():
            print(f"\n📊 Model: {model_key}")
            print("-" * 80)
            
            for set_name, metrics in sets.items():
                print(f"\n  {set_name.upper()} SET:")
                print(f"    Samples:     {metrics['n_samples']}")
                print(f"    Y range:     [{metrics['y_range'][0]:.2f}, {metrics['y_range'][1]:.2f}]")
                print(f"    Y mean ± std: {metrics['y_mean']:.2f} ± {metrics['y_std']:.2f}")
                print(f"    ┌─ MAE:  {metrics['mae']:.4f}")
                print(f"    ├─ RMSE: {metrics['rmse']:.4f}")
                print(f"    ├─ MAPE: {metrics['mape']:.2f}%")
                print(f"    └─ R²:   {metrics['r2']:.4f}")
    def plot_predictions_vs_true(self, model_key: str, figsize: Tuple[int, int] = (16, 5)) -> None:
        """Plot predictions vs true values with regression lines."""
        
        sets = self.predictions[model_key]
        n_sets = len(sets)
        
        fig, axes = plt.subplots(1, n_sets, figsize=figsize)
        if n_sets == 1:
            axes = [axes]
        
        for ax, (set_name, preds) in zip(axes, sets.items()):
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='k', linewidth=0.5, c=y_true, cmap='viridis')
            
            # Perfect prediction line
            y_min, y_max = y_true.min(), y_true.max()
            ax.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2.5, label='Perfect prediction', alpha=0.7)
            
            # Regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(y_min, y_max, 100)
            ax.plot(x_line, p(x_line), 'g-', lw=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', alpha=0.7)
            
            # Metrics
            metrics = self.metrics[model_key][set_name]
            r2 = metrics['r2']
            mae = metrics['mae']
            rmse = metrics['rmse']
            
            title = f"{set_name.upper()} SET (n={metrics['n_samples']})\n"
            title += f"R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('True Borg Rating', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Borg Rating', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plot_path = self.output_dir / f"01_predictions_vs_true_{model_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
    
    def plot_residuals(self, model_key: str, figsize: Tuple[int, int] = (16, 5)) -> None:
        """Plot residuals analysis."""
        
        sets = self.predictions[model_key]
        n_sets = len(sets)
        
        fig, axes = plt.subplots(1, n_sets, figsize=figsize)
        if n_sets == 1:
            axes = [axes]
        
        for ax, (set_name, preds) in zip(axes, sets.items()):
            y_pred = preds['y_pred']
            residuals = preds['residuals']
            
            # Scatter plot
            scatter = ax.scatter(y_pred, residuals, alpha=0.6, s=60, c=np.abs(residuals), 
                               cmap='RdYlGn_r', edgecolors='k', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', lw=2.5, alpha=0.8)
            
            # Add ±1 std bands
            std_residuals = np.std(residuals)
            ax.fill_between([y_pred.min(), y_pred.max()], 
                           -std_residuals, std_residuals, 
                           alpha=0.2, color='green', label=f'±1σ ({std_residuals:.4f})')
            
            metrics = self.metrics[model_key][set_name]
            mae = metrics['mae']
            rmse = metrics['rmse']
            
            title = f"{set_name.upper()} RESIDUALS (n={metrics['n_samples']})\n"
            title += f"MAE = {mae:.4f} | RMSE = {rmse:.4f}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Borg Rating', fontsize=11, fontweight='bold')
            ax.set_ylabel('Residuals (True - Pred)', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='|Residual|')
        
        plt.tight_layout()
        plot_path = self.output_dir / f"02_residuals_{model_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
    
    def plot_residuals_histogram(self, model_key: str, figsize: Tuple[int, int] = (16, 5)) -> None:
        """Plot residuals distribution."""
        
        sets = self.predictions[model_key]
        n_sets = len(sets)
        
        fig, axes = plt.subplots(1, n_sets, figsize=figsize)
        if n_sets == 1:
            axes = [axes]
        
        for ax, (set_name, preds) in zip(axes, sets.items()):
            residuals = preds['residuals']
            
            # Histogram with KDE
            ax.hist(residuals, bins=25, alpha=0.7, edgecolor='black', density=True, color='skyblue')
            
            # KDE
            from scipy import stats
            kde = stats.gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 150)
            ax.plot(x_range, kde(x_range), 'b-', lw=2.5, label='KDE')
            
            # Normal distribution for comparison
            mu, sigma = residuals.mean(), residuals.std()
            x_normal = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
            ax.plot(x_normal, stats.norm.pdf(x_normal, mu, sigma), 'g--', lw=2.5, label='Normal fit')
            
            ax.axvline(x=0, color='r', linestyle='-', lw=1.5, alpha=0.8, label='Zero')
            
            title = f"{set_name.upper()} RESIDUALS DISTRIBUTION\n"
            title += f"μ={mu:.4f}, σ={sigma:.4f} (n={len(residuals)})"
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Residuals', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = self.output_dir / f"03_residuals_histogram_{model_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
    
    def plot_feature_importance(self, model_key: str, top_n: int = 20, figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot top N feature importances."""
        
        model = self.models[model_key]
        importances = model.feature_importances_
        feature_names = self.feature_cols[model_key]
        
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        bars = ax.barh(range(top_n), top_importances, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f"Top {top_n} Feature Importances - {model_key}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            ax.text(importance, i, f' {importance:.4f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / f"04_feature_importance_{model_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
    
    def plot_metrics_bars(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """Plot metrics comparison across all models."""
        
        if not self.metrics:
            print("No metrics to compare")
            return
        
        # Prepare data
        model_names = list(self.metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('mae', 'Mean Absolute Error (MAE)', 'skyblue'),
            ('rmse', 'Root Mean Squared Error (RMSE)', 'lightcoral'),
            ('r2', 'R² Score', 'lightgreen'),
            ('mape', 'Mean Absolute Percentage Error (MAPE %)', 'lightyellow'),
        ]
        
        for ax, (metric_name, title, color) in zip(axes, metrics_to_plot):
            train_vals = [self.metrics[m].get('train', {}).get(metric_name, 0) for m in model_names]
            test_vals = [self.metrics[m].get('test', {}).get(metric_name, 0) for m in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color=color, alpha=0.5, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel(metric_name.upper(), fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_path = self.output_dir / "05_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
    
    def save_metrics_json(self) -> None:
        """Save metrics to JSON file."""
        
        # Convert numpy types to native Python types
        metrics_serializable = {}
        for model_key, sets in self.metrics.items():
            metrics_serializable[model_key] = {}
            for set_name, metrics in sets.items():
                metrics_serializable[model_key][set_name] = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) 
                         else (tuple(float(x) for x in v) if isinstance(v, tuple) else v))
                    for k, v in metrics.items()
                }
        
        output_path = self.output_dir / "evaluation_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"✓ Saved metrics: {output_path.name}")
    
    def generate_full_report(self, model_key: str) -> None:
        """Generate complete evaluation report with all plots."""
        
        print(f"\n{'='*80}")
        print(f"GENERATING FULL EVALUATION REPORT: {model_key}")
        print(f"{'='*80}")
        
        # Generate plots
        print(f"\n📊 Generating visualizations...")
        self.plot_predictions_vs_true(model_key)
        self.plot_residuals(model_key)
        self.plot_residuals_histogram(model_key)
        self.plot_feature_importance(model_key, top_n=20)
        
        print(f"✓ Full report generated!")


def main():
    """Main training and evaluation script."""
    
    # Configuration
    fused_data_path = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv")
    output_dir = Path("/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/xgboost_models")
    
    # Check if data exists
    if not fused_data_path.exists():
        print(f"❌ Data file not found: {fused_data_path}")
        return
    
    # Initialize trainer/evaluator
    evaluator = ModelTrainerEvaluator(str(output_dir))
    
    # Load data
    print(f"\n{'='*80}")
    print(f"LOADING DATA")
    print(f"{'='*80}")
    print(f"\n📂 Loading fused data from: {fused_data_path.name}")
    
    df = pd.read_csv(fused_data_path)
    df_labeled = df.dropna(subset=["borg"]).copy()
    
    print(f"   Total samples: {len(df)}")
    print(f"   Labeled samples: {len(df_labeled)}")
    print(f"   Borg range: [{df_labeled['borg'].min():.2f}, {df_labeled['borg'].max():.2f}]")
    
    # Prepare features - exclude all metadata and numbered variants
    skip_cols = {
        "window_id", "start_idx", "end_idx", "valid",
        "t_start", "t_center", "t_end", "n_samples", "win_sec",
        "modality", "subject", "borg",
    }
    
    def is_metadata(col):
        if col in skip_cols:
            return True
        # Exclude all _r variants (single and numbered like _r.1, _r.2, etc.)
        if col.endswith("_r") or any(col.endswith(f"_r.{i}") for i in range(1, 10)):
            return True
        return False
    
    feature_cols = [col for col in df_labeled.columns if not is_metadata(col)]
    
    X = df_labeled[feature_cols].values
    y = df_labeled["borg"].values
    
    print(f"\n📊 Features (no metadata): {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    print(f"   Y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n🔀 Train-test split:")
    print(f"   Train: {len(X_train)} samples, Y range [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Test: {len(X_test)} samples, Y range [{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Train model
    evaluator.train_model(X_train, y_train, X_test, y_test, feature_cols, "10.0s", window_length=10.0)
    
    # Print summary
    evaluator.print_metrics_summary()
    
    # Generate plots
    print(f"\n{'='*80}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    evaluator.generate_full_report("10.0s")
    evaluator.plot_metrics_bars()
    
    # Save metrics
    evaluator.save_metrics_json()
    
    print(f"\n{'='*80}")
    print(f"✓ TRAINING & EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📂 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
