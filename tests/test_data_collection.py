"""
Tests for data collection module.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.data_collection import (
    SalesDataCollector,
    MediaSpendCollector,
    DataAggregator,
    DataValidator
)


def test_sales_data_collector():
    """Test sales data collection."""
    collector = SalesDataCollector()
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    df = collector.collect(start_date, end_date)
    
    assert len(df) > 0
    assert 'date' in df.columns
    assert 'revenue' in df.columns
    assert collector.validate(df)


def test_media_spend_collector():
    """Test media spend collection."""
    collector = MediaSpendCollector()
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    df = collector.collect(start_date, end_date)
    
    assert len(df) > 0
    assert 'date' in df.columns
    assert 'channel' in df.columns
    assert 'spend' in df.columns
    assert collector.validate(df)


def test_data_aggregator():
    """Test data aggregation."""
    sales_collector = SalesDataCollector()
    media_collector = MediaSpendCollector()
    
    aggregator = DataAggregator([sales_collector, media_collector])
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    data_dict = aggregator.collect_all(start_date, end_date)
    
    assert 'sales' in data_dict
    assert 'media_spend' in data_dict
    
    merged = aggregator.merge_data(data_dict)
    assert len(merged) > 0
    assert 'revenue' in merged.columns


def test_data_validator():
    """Test data validation."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'revenue': [10000] * 30,
        'spend_tv': [1000] * 30
    })
    
    validator = DataValidator()
    results = validator.validate_all(df, spend_columns=['spend_tv'])
    
    assert 'is_valid' in results
    assert 'checks' in results

