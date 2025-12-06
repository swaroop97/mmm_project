"""
Model Monitoring Module

Tracks model performance, detects drift, and triggers retraining.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor MMM model performance and data quality.
    """
    
    def __init__(
        self,
        model,
        metrics_history_path: Optional[str] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            model: Trained MMM model
            metrics_history_path: Path to save metrics history
        """
        self.model = model
        self.metrics_history_path = metrics_history_path or 'data/monitoring/metrics_history.json'
        self.metrics_history = self._load_history()
        self.logger = logging.getLogger(__name__)
    
    def _load_history(self) -> List[Dict]:
        """Load metrics history from file."""
        path = Path(self.metrics_history_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save metrics history to file."""
        path = Path(self.metrics_history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate model performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            mean_absolute_percentage_error
        )
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred)
        }
    
    def monitor_performance(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'revenue',
        window_days: int = 30
    ) -> Dict:
        """
        Monitor model performance on recent data.
        
        Args:
            df: Input data
            date_column: Name of date column
            value_column: Name of target column
            window_days: Number of days to evaluate
            
        Returns:
            Dictionary with performance metrics
        """
        self.logger.info("Monitoring model performance")
        
        # Filter to recent window
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        cutoff_date = df[date_column].max() - timedelta(days=window_days)
        recent_df = df[df[date_column] >= cutoff_date].copy()
        
        if len(recent_df) == 0:
            self.logger.warning("No recent data found")
            return {}
        
        # Make predictions
        y_true = recent_df[value_column].values
        y_pred = self.model.predict(recent_df, date_column)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(y_true, y_pred)
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['window_days'] = window_days
        metrics['n_samples'] = len(recent_df)
        
        # Add to history
        self.metrics_history.append(metrics)
        self._save_history()
        
        self.logger.info(f"Performance metrics: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        
        return metrics
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect data drift between current and reference data.
        
        Args:
            current_data: Current data
            reference_data: Reference (training) data
            columns: Columns to check (if None, checks all numeric)
            
        Returns:
            Dictionary with drift detection results
        """
        self.logger.info("Detecting data drift")
        
        from scipy import stats
        
        if columns is None:
            columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_results = {}
        
        for col in columns:
            if col not in current_data.columns or col not in reference_data.columns:
                continue
            
            current_values = current_data[col].dropna()
            reference_values = reference_data[col].dropna()
            
            if len(current_values) == 0 or len(reference_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(reference_values, current_values)
            
            # Mean shift
            mean_shift = current_values.mean() - reference_values.mean()
            mean_shift_pct = (mean_shift / reference_values.mean()) * 100 if reference_values.mean() != 0 else 0
            
            # Variance shift
            var_shift = current_values.var() - reference_values.var()
            var_shift_pct = (var_shift / reference_values.var()) * 100 if reference_values.var() != 0 else 0
            
            drift_results[col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'drift_detected': ks_pvalue < 0.05,  # Significant drift
                'mean_shift': mean_shift,
                'mean_shift_pct': mean_shift_pct,
                'var_shift': var_shift,
                'var_shift_pct': var_shift_pct
            }
        
        # Overall drift flag
        drift_detected = any(
            result['drift_detected']
            for result in drift_results.values()
        )
        
        return {
            'drift_detected': drift_detected,
            'columns': drift_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_retraining_trigger(
        self,
        performance_threshold: float = 0.7,
        drift_threshold: float = 0.3
    ) -> Tuple[bool, str]:
        """
        Check if model should be retrained.
        
        Args:
            performance_threshold: Minimum R² to maintain
            drift_threshold: Maximum allowed drift
            
        Returns:
            (should_retrain, reason)
        """
        if not self.metrics_history:
            return False, "No metrics history"
        
        # Check recent performance
        recent_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        if recent_metrics.get('r2', 1.0) < performance_threshold:
            return True, f"Low performance: R² = {recent_metrics.get('r2', 0):.4f}"
        
        # Check for performance degradation
        if len(self.metrics_history) >= 2:
            current_r2 = recent_metrics.get('r2', 1.0)
            previous_r2 = self.metrics_history[-2].get('r2', 1.0)
            
            if current_r2 < previous_r2 - 0.1:  # 10% drop
                return True, f"Performance degradation: {previous_r2:.4f} -> {current_r2:.4f}"
        
        return False, "No retraining needed"
    
    def generate_report(self) -> str:
        """
        Generate monitoring report.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("MODEL MONITORING REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.metrics_history:
            report.append("\nNo metrics history available.")
            return "\n".join(report)
        
        # Recent metrics
        recent = self.metrics_history[-1]
        report.append("\nRecent Performance Metrics:")
        report.append(f"  R² Score: {recent.get('r2', 0):.4f}")
        report.append(f"  RMSE: {recent.get('rmse', 0):.2f}")
        report.append(f"  MAE: {recent.get('mae', 0):.2f}")
        report.append(f"  MAPE: {recent.get('mape', 0):.2f}%")
        
        # Retraining check
        should_retrain, reason = self.check_retraining_trigger()
        report.append(f"\nRetraining Status: {'REQUIRED' if should_retrain else 'NOT REQUIRED'}")
        if should_retrain:
            report.append(f"  Reason: {reason}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would use a trained model
    # For demo, create a simple mock
    from ..modeling.mmm_model import MMMModel
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(10000, 2000, len(dates)),
        'spend_tv': np.random.uniform(1000, 5000, len(dates)),
        'spend_digital': np.random.uniform(500, 3000, len(dates))
    })
    
    # Train model
    model = MMMModel(media_channels=['TV', 'Digital'])
    model.train(df)
    
    # Monitor
    monitor = ModelMonitor(model)
    metrics = monitor.monitor_performance(df)
    
    print(monitor.generate_report())

