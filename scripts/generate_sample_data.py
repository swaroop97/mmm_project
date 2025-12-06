"""
Generate Sample Data for MMM Pipeline

Creates synthetic marketing mix data for demonstration and testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "data/raw/sample_data.csv",
    seed: int = 42
):
    """
    Generate synthetic MMM data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Output file path
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    logger.info(f"Generating {n_days} days of sample data...")
    
    # Media channels
    channels = ['TV', 'Radio', 'Digital', 'Social', 'Search', 'Display']
    
    # Generate media spend with seasonality and trends
    data = {'date': dates}
    
    for channel in channels:
        # Base spend with channel-specific patterns
        base_spend = {
            'TV': 5000,
            'Radio': 2000,
            'Digital': 3000,
            'Social': 1500,
            'Search': 2500,
            'Display': 1000
        }[channel]
        
        # Add seasonality (higher in Q4)
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + np.pi/2)
        
        # Add trend
        trend = 1 + 0.001 * np.arange(n_days) / n_days
        
        # Add random variation
        noise = np.random.normal(1, 0.2, n_days)
        
        spend = base_spend * seasonal * trend * noise
        spend = np.maximum(spend, 0)  # Ensure non-negative
        
        data[f'spend_{channel}'] = spend
    
    # Generate revenue (dependent on media spend + external factors)
    df = pd.DataFrame(data)
    
    # External factors
    df['gdp_growth'] = np.random.normal(0.02, 0.01, n_days)  # Monthly GDP growth
    df['unemployment'] = np.random.normal(0.05, 0.01, n_days)  # Unemployment rate
    df['inflation'] = np.random.normal(0.03, 0.005, n_days)  # Inflation rate
    
    # Holiday flags
    df['holiday_flag'] = 0
    # Add some holiday periods
    for year in [2023, 2024]:
        # Christmas period
        df.loc[(df['date'] >= f'{year}-12-20') & (df['date'] <= f'{year}-12-31'), 'holiday_flag'] = 1
        # New Year
        df.loc[(df['date'] >= f'{year+1}-01-01') & (df['date'] <= f'{year+1}-01-05'), 'holiday_flag'] = 1
        # Black Friday
        df.loc[df['date'] == f'{year}-11-24', 'holiday_flag'] = 1
    
    # Time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Generate revenue with realistic relationships
    # Base revenue
    base_revenue = 50000
    
    # Media contribution (with diminishing returns)
    media_contribution = 0
    for channel in channels:
        spend = df[f'spend_{channel}']
        # Hill saturation function
        alpha = 0.5 + np.random.uniform(-0.2, 0.2)
        gamma = 0.5 + np.random.uniform(-0.2, 0.2)
        saturation = (spend ** alpha) / (spend ** alpha + gamma * 1000)
        
        # Channel-specific coefficients
        coef = {
            'TV': 15,
            'Radio': 8,
            'Digital': 12,
            'Social': 10,
            'Search': 14,
            'Display': 6
        }[channel]
        
        media_contribution += coef * saturation * 100
    
    # External factor effects
    external_effect = (
        5000 * df['gdp_growth'] +
        -2000 * df['unemployment'] +
        -1000 * df['inflation'] +
        3000 * df['holiday_flag']
    )
    
    # Trend
    trend = 1 + 0.0005 * np.arange(n_days) / n_days
    
    # Random noise
    noise = np.random.normal(0, 5000, n_days)
    
    # Total revenue
    df['revenue'] = (base_revenue + media_contribution + external_effect) * trend + noise
    df['revenue'] = np.maximum(df['revenue'], 0)  # Ensure non-negative
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to {output_path}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Revenue range: ${df['revenue'].min():,.2f} to ${df['revenue'].max():,.2f}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample MMM data")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/raw/sample_data.csv",
                       help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    generate_sample_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        seed=args.seed
    )

