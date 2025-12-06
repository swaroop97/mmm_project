"""
Data Validation Module for MMM Pipeline

Implements data quality checks, validation rules, and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for MMM data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_completeness(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        expected_frequency: str = 'D'
    ) -> Tuple[bool, Dict]:
        """
        Check data completeness.
        
        Args:
            df: DataFrame to validate
            date_column: Name of date column
            expected_frequency: Expected frequency ('D', 'W', 'M')
            
        Returns:
            (is_valid, issues_dict)
        """
        self.logger.info("Validating data completeness")
        
        issues = []
        
        # Check date range continuity
        dates = pd.to_datetime(df[date_column])
        expected_dates = pd.date_range(
            dates.min(),
            dates.max(),
            freq=expected_frequency
        )
        
        missing_dates = set(expected_dates) - set(dates)
        if missing_dates:
            issues.append({
                'type': 'missing_dates',
                'count': len(missing_dates),
                'dates': sorted(list(missing_dates))[:10]  # First 10
            })
        
        # Check for duplicate dates
        duplicates = dates.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'duplicate_dates',
                'count': duplicates
            })
        
        is_valid = len(issues) == 0
        
        return is_valid, {'issues': issues}
    
    def validate_numeric_ranges(
        self,
        df: pd.DataFrame,
        column_ranges: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, Dict]:
        """
        Validate numeric columns are within expected ranges.
        
        Args:
            df: DataFrame to validate
            column_ranges: Dict mapping column names to (min, max) tuples
            
        Returns:
            (is_valid, issues_dict)
        """
        self.logger.info("Validating numeric ranges")
        
        issues = []
        
        for column, (min_val, max_val) in column_ranges.items():
            if column not in df.columns:
                issues.append({
                    'type': 'missing_column',
                    'column': column
                })
                continue
            
            out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
            
            if out_of_range > 0:
                issues.append({
                    'type': 'out_of_range',
                    'column': column,
                    'count': out_of_range,
                    'min': min_val,
                    'max': max_val,
                    'actual_min': df[column].min(),
                    'actual_max': df[column].max()
                })
        
        is_valid = len(issues) == 0
        
        return is_valid, {'issues': issues}
    
    def validate_no_negative_values(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[bool, Dict]:
        """
        Check for negative values in specified columns.
        
        Args:
            df: DataFrame to validate
            columns: List of columns to check
            
        Returns:
            (is_valid, issues_dict)
        """
        self.logger.info("Validating no negative values")
        
        issues = []
        
        for column in columns:
            if column not in df.columns:
                continue
            
            negative_count = (df[column] < 0).sum()
            
            if negative_count > 0:
                issues.append({
                    'type': 'negative_values',
                    'column': column,
                    'count': negative_count,
                    'min_value': df[column].min()
                })
        
        is_valid = len(issues) == 0
        
        return is_valid, {'issues': issues}
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies in a column using statistical methods.
        
        Args:
            df: DataFrame to analyze
            column: Column name
            method: Method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for detection
            
        Returns:
            DataFrame with anomaly flags
        """
        self.logger.info(f"Detecting anomalies in {column} using {method}")
        
        result_df = df.copy()
        result_df[f'{column}_anomaly'] = 0
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].fillna(df[column].median())))
            result_df[f'{column}_anomaly'] = (z_scores > threshold).astype(int)
        
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            result_df[f'{column}_anomaly'] = (
                (df[column] < lower_bound) | (df[column] > upper_bound)
            ).astype(int)
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(contamination=0.1, random_state=42)
            anomalies = model.fit_predict(df[[column]].fillna(df[column].median()).values)
            result_df[f'{column}_anomaly'] = (anomalies == -1).astype(int)
        
        anomaly_count = result_df[f'{column}_anomaly'].sum()
        self.logger.info(f"Detected {anomaly_count} anomalies in {column}")
        
        return result_df
    
    def validate_all(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        spend_columns: Optional[List[str]] = None,
        value_column: str = 'revenue'
    ) -> Dict:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            date_column: Name of date column
            spend_columns: List of media spend column names
            value_column: Name of revenue/sales column
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Running comprehensive validation")
        
        results = {
            'is_valid': True,
            'checks': {}
        }
        
        # Completeness check
        is_complete, completeness_issues = self.validate_completeness(df, date_column)
        results['checks']['completeness'] = {
            'passed': is_complete,
            'issues': completeness_issues
        }
        
        if not is_complete:
            results['is_valid'] = False
        
        # Negative values check
        columns_to_check = [value_column]
        if spend_columns:
            columns_to_check.extend(spend_columns)
        
        is_no_negative, negative_issues = self.validate_no_negative_values(
            df, columns_to_check
        )
        results['checks']['no_negative'] = {
            'passed': is_no_negative,
            'issues': negative_issues
        }
        
        if not is_no_negative:
            results['is_valid'] = False
        
        # Range validation for revenue
        if value_column in df.columns:
            revenue_min = df[value_column].quantile(0.01)
            revenue_max = df[value_column].quantile(0.99)
            
            is_in_range, range_issues = self.validate_numeric_ranges(
                df,
                {value_column: (revenue_min * 0.5, revenue_max * 2.0)}
            )
            results['checks']['ranges'] = {
                'passed': is_in_range,
                'issues': range_issues
            }
        
        # Anomaly detection
        if value_column in df.columns:
            df_with_anomalies = self.detect_anomalies(df, value_column, method='zscore')
            anomaly_count = df_with_anomalies[f'{value_column}_anomaly'].sum()
            
            results['checks']['anomalies'] = {
                'passed': anomaly_count == 0,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / len(df)
            }
        
        self.validation_results = results
        
        return results
    
    def get_validation_report(self) -> str:
        """
        Generate human-readable validation report.
        
        Returns:
            Validation report string
        """
        if not self.validation_results:
            return "No validation results available"
        
        report = []
        report.append("=" * 70)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"\nOverall Status: {'PASSED' if self.validation_results['is_valid'] else 'FAILED'}")
        report.append("\nDetailed Checks:")
        
        for check_name, check_result in self.validation_results['checks'].items():
            status = "PASSED" if check_result['passed'] else "FAILED"
            report.append(f"\n  {check_name.upper()}: {status}")
            
            if 'issues' in check_result and check_result['issues']:
                for issue in check_result['issues']:
                    report.append(f"    - {issue}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(10000, 2000, len(dates)),
        'spend_tv': np.random.uniform(1000, 5000, len(dates)),
        'spend_digital': np.random.uniform(500, 3000, len(dates))
    })
    
    # Validate
    validator = DataValidator()
    results = validator.validate_all(
        df,
        spend_columns=['spend_tv', 'spend_digital']
    )
    
    print(validator.get_validation_report())

