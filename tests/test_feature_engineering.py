"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import AdstockTransformer, SaturationTransformer


def test_adstock_transformer():
    """Test adstock transformation."""
    transformer = AdstockTransformer(method='geometric', decay_rate=0.5)
    
    # Create sample data
    data = pd.DataFrame({
        'spend': [100, 0, 0, 0, 0]
    })
    
    transformed = transformer.transform(data['spend'])
    
    assert len(transformed) == len(data)
    assert transformed[0] == 100  # First value unchanged
    assert transformed[1] < transformed[0]  # Decay applied
    assert all(transformed >= 0)  # Non-negative


def test_saturation_transformer():
    """Test saturation transformation."""
    transformer = SaturationTransformer(method='hill', alpha=0.5, gamma=0.5)
    
    # Create sample data
    spend = np.array([100, 200, 300, 400, 500])
    
    transformed = transformer.transform(spend)
    
    assert len(transformed) == len(spend)
    assert all(transformed >= 0)
    assert all(transformed <= 1)  # Hill function outputs [0, 1]
    
    # Check diminishing returns
    increments = np.diff(transformed)
    assert all(increments >= 0)  # Monotonically increasing
    # Later increments should be smaller (diminishing returns)
    assert increments[-1] < increments[0]


def test_adstock_saturation_pipeline():
    """Test combined adstock and saturation."""
    adstock = AdstockTransformer(method='geometric', decay_rate=0.6)
    saturation = SaturationTransformer(method='hill', alpha=0.5, gamma=0.5)
    
    spend = pd.Series([100, 50, 0, 0, 0])
    
    # Apply adstock first
    adstocked = adstock.transform(spend)
    
    # Then saturation
    saturated = saturation.transform(adstocked)
    
    assert len(saturated) == len(spend)
    assert all(saturated >= 0)
    assert all(saturated <= 1)

