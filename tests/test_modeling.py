"""
Tests for modeling module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.modeling import MMMModel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(50000, 10000, len(dates)),
        'spend_TV': np.random.normal(5000, 1000, len(dates)),
        'spend_Digital': np.random.normal(3000, 500, len(dates)),
        'spend_Social': np.random.normal(1500, 300, len(dates)),
        'gdp_growth': np.random.normal(0.02, 0.01, len(dates)),
        'unemployment': np.random.normal(0.05, 0.01, len(dates)),
        'holiday_flag': np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
    })
    
    return data


def test_mmm_model_initialization():
    """Test MMM model initialization."""
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment'],
        adstock_params={'TV': 0.6, 'Digital': 0.4, 'Social': 0.3},
        saturation_params={
            'TV': {'method': 'hill', 'alpha': 0.5, 'gamma': 0.5}
        }
    )
    
    assert model.media_channels == ['TV', 'Digital', 'Social']
    assert model.external_factors == ['gdp_growth', 'unemployment']
    assert model.model is None  # Not trained yet


def test_mmm_model_training(sample_data):
    """Test MMM model training."""
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment', 'holiday_flag'],
        adstock_params={'TV': 0.6, 'Digital': 0.4, 'Social': 0.3}
    )
    
    model.train(sample_data)
    
    assert model.model is not None
    assert model.is_trained


def test_mmm_model_prediction(sample_data):
    """Test MMM model prediction."""
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment', 'holiday_flag']
    )
    
    model.train(sample_data)
    
    predictions = model.predict(sample_data)
    
    assert len(predictions) == len(sample_data)
    assert all(predictions >= 0)  # Revenue should be non-negative


def test_mmm_model_evaluation(sample_data):
    """Test MMM model evaluation."""
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment', 'holiday_flag']
    )
    
    model.train(sample_data)
    
    metrics = model.evaluate(sample_data)
    
    assert 'r2_score' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 0 <= metrics['r2_score'] <= 1  # R² should be in [0, 1]


def test_channel_roi(sample_data):
    """Test channel ROI calculation."""
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment', 'holiday_flag']
    )
    
    model.train(sample_data)
    
    roi_df = model.get_channel_roi(sample_data)
    
    assert 'channel' in roi_df.columns
    assert 'roi' in roi_df.columns
    assert len(roi_df) == len(model.media_channels)

