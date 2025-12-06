"""
Data Collection Module for MMM Pipeline

Handles data ingestion from multiple sources:
- Sales data (internal systems)
- Media spend data (ad platforms)
- External factors (economic indicators)
- Competitive data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Base class for data collection from various sources.
    """
    
    def __init__(self, source_name: str, config: Optional[Dict] = None):
        """
        Initialize data collector.
        
        Args:
            source_name: Name of data source
            config: Configuration dictionary
        """
        self.source_name = source_name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
    
    def collect(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Collect data for date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with collected data
        """
        raise NotImplementedError("Subclasses must implement collect()")
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate collected data.
        
        Args:
            df: Data to validate
            
        Returns:
            True if valid
        """
        raise NotImplementedError("Subclasses must implement validate()")


class SalesDataCollector(DataCollector):
    """
    Collect sales/revenue data from internal systems.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("sales", config)
        self.date_column = self.config.get('date_column', 'date')
        self.value_column = self.config.get('value_column', 'revenue')
        self.product_column = self.config.get('product_column', 'product')
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Collect sales data.
        
        Args:
            start_date: Start date
            end_date: End date
            product: Optional product filter
            
        Returns:
            DataFrame with sales data
        """
        self.logger.info(f"Collecting sales data from {start_date} to {end_date}")
        
        # In production, this would query a database
        # For demo, generate synthetic data
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        for date in dates:
            # Simulate sales with trend and seasonality
            base_sales = 10000
            trend = (date - start_date).days * 5
            seasonality = 1000 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, 500)
            
            revenue = base_sales + trend + seasonality + noise
            
            data.append({
                self.date_column: date,
                self.value_column: max(0, revenue),
                self.product_column: product or 'default'
            })
        
        df = pd.DataFrame(data)
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate sales data."""
        required_columns = [self.date_column, self.value_column]
        
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False
        
        if df[self.value_column].min() < 0:
            self.logger.warning("Negative sales values found")
        
        if df[self.date_column].isnull().any():
            self.logger.error("Missing dates found")
            return False
        
        return True


class MediaSpendCollector(DataCollector):
    """
    Collect media spend data from advertising platforms.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("media_spend", config)
        self.channels = self.config.get('channels', [
            'TV', 'Radio', 'Digital', 'Social', 'Search', 'Display'
        ])
        self.date_column = self.config.get('date_column', 'date')
        self.spend_column = self.config.get('spend_column', 'spend')
        self.channel_column = self.config.get('channel_column', 'channel')
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        channels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect media spend data.
        
        Args:
            start_date: Start date
            end_date: End date
            channels: Optional channel filter
            
        Returns:
            DataFrame with media spend data
        """
        self.logger.info(f"Collecting media spend from {start_date} to {end_date}")
        
        channels = channels or self.channels
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        for date in dates:
            for channel in channels:
                # Simulate media spend with campaigns
                base_spend = np.random.uniform(1000, 10000)
                campaign_boost = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) * 5000
                spend = base_spend + campaign_boost
                
                data.append({
                    self.date_column: date,
                    self.channel_column: channel,
                    self.spend_column: spend,
                    'impressions': spend * np.random.uniform(10, 50),
                    'clicks': spend * np.random.uniform(0.1, 1.0)
                })
        
        df = pd.DataFrame(data)
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate media spend data."""
        required_columns = [self.date_column, self.channel_column, self.spend_column]
        
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False
        
        if df[self.spend_column].min() < 0:
            self.logger.error("Negative spend values found")
            return False
        
        return True


class ExternalFactorsCollector(DataCollector):
    """
    Collect external factors (economic indicators, seasonality, etc.).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("external_factors", config)
        self.factors = self.config.get('factors', [
            'gdp_growth', 'unemployment', 'inflation', 'holiday_flag'
        ])
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect external factors.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with external factors
        """
        self.logger.info(f"Collecting external factors from {start_date} to {end_date}")
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        for date in dates:
            row = {'date': date}
            
            # GDP growth (quarterly, interpolated)
            row['gdp_growth'] = np.random.normal(2.5, 0.5)
            
            # Unemployment rate
            row['unemployment'] = np.random.normal(5.0, 0.3)
            
            # Inflation
            row['inflation'] = np.random.normal(2.0, 0.2)
            
            # Holiday flag
            row['holiday_flag'] = 1 if date.month == 12 or date.month == 1 else 0
            
            # Day of week
            row['day_of_week'] = date.dayofweek
            row['is_weekend'] = 1 if date.dayofweek >= 5 else 0
            
            # Month
            row['month'] = date.month
            row['quarter'] = date.quarter
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate external factors."""
        if 'date' not in df.columns:
            self.logger.error("Missing date column")
            return False
        
        return True


class DataAggregator:
    """
    Aggregate data from multiple collectors.
    """
    
    def __init__(self, collectors: List[DataCollector]):
        """
        Initialize aggregator with list of collectors.
        
        Args:
            collectors: List of data collector instances
        """
        self.collectors = collectors
        self.logger = logging.getLogger(__name__)
    
    def collect_all(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for collectors
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        self.logger.info(f"Collecting data from {len(self.collectors)} sources")
        
        data = {}
        
        for collector in self.collectors:
            try:
                df = collector.collect(start_date, end_date, **kwargs)
                
                # Validate
                if collector.validate(df):
                    data[collector.source_name] = df
                    self.logger.info(f"Successfully collected {len(df)} rows from {collector.source_name}")
                else:
                    self.logger.error(f"Validation failed for {collector.source_name}")
            
            except Exception as e:
                self.logger.error(f"Error collecting from {collector.source_name}: {e}")
        
        return data
    
    def merge_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Merge data from multiple sources.
        
        Args:
            data_dict: Dictionary of DataFrames
            date_column: Date column name
            
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging data from multiple sources")
        
        # Start with sales data
        if 'sales' in data_dict:
            merged = data_dict['sales'].copy()
        else:
            raise ValueError("Sales data required for merging")
        
        # Merge media spend (pivot channels)
        if 'media_spend' in data_dict:
            media = data_dict['media_spend']
            media_pivot = media.pivot_table(
                index=date_column,
                columns='channel',
                values='spend',
                aggfunc='sum'
            ).fillna(0)
            
            # Rename columns
            media_pivot.columns = [f'spend_{col.lower()}' for col in media_pivot.columns]
            merged = merged.merge(media_pivot, left_on=date_column, right_index=True, how='left')
            merged[media_pivot.columns] = merged[media_pivot.columns].fillna(0)
        
        # Merge external factors
        if 'external_factors' in data_dict:
            factors = data_dict['external_factors']
            merged = merged.merge(factors, on=date_column, how='left')
        
        # Sort by date
        merged = merged.sort_values(date_column).reset_index(drop=True)
        
        self.logger.info(f"Merged data shape: {merged.shape}")
        
        return merged


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize collectors
    sales_collector = SalesDataCollector()
    media_collector = MediaSpendCollector()
    external_collector = ExternalFactorsCollector()
    
    # Aggregate
    aggregator = DataAggregator([sales_collector, media_collector, external_collector])
    
    # Collect data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data_dict = aggregator.collect_all(start_date, end_date)
    
    # Merge
    merged_data = aggregator.merge_data(data_dict)
    
    print(f"\nMerged data shape: {merged_data.shape}")
    print(f"\nColumns: {merged_data.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(merged_data.head())

