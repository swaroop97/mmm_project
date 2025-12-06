"""
Tests for optimization module.
"""

import pytest
import pandas as pd
import numpy as np
from src.modeling import MMMModel
from src.optimization import BudgetOptimizer


@pytest.fixture
def trained_model():
    """Create a trained model for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(50000, 10000, len(dates)),
        'spend_TV': np.random.normal(5000, 1000, len(dates)),
        'spend_Digital': np.random.normal(3000, 500, len(dates)),
        'spend_Social': np.random.normal(1500, 300, len(dates)),
        'gdp_growth': np.random.normal(0.02, 0.01, len(dates)),
        'unemployment': np.random.normal(0.05, 0.01, len(dates))
    })
    
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        external_factors=['gdp_growth', 'unemployment']
    )
    
    model.train(data)
    return model


def test_budget_optimizer_initialization(trained_model):
    """Test budget optimizer initialization."""
    optimizer = BudgetOptimizer(trained_model, method='scipy', objective='roi')
    
    assert optimizer.mmm_model == trained_model
    assert optimizer.method == 'scipy'
    assert optimizer.objective == 'roi'


def test_budget_optimization(trained_model):
    """Test budget optimization."""
    optimizer = BudgetOptimizer(trained_model, method='scipy')
    
    base_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-12-31')],
        'gdp_growth': [0.02],
        'unemployment': [0.05]
    })
    
    optimal_budget = optimizer.optimize(
        total_budget=100000,
        channels=['TV', 'Digital', 'Social'],
        base_data=base_data
    )
    
    assert len(optimal_budget) == 3
    assert all(spend >= 0 for spend in optimal_budget.values())
    
    # Check budget constraint
    total_allocated = sum(optimal_budget.values())
    assert abs(total_allocated - 100000) < 100  # Allow small numerical error


def test_budget_optimization_with_constraints(trained_model):
    """Test budget optimization with constraints."""
    optimizer = BudgetOptimizer(trained_model, method='scipy')
    
    base_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-12-31')],
        'gdp_growth': [0.02],
        'unemployment': [0.05]
    })
    
    optimal_budget = optimizer.optimize(
        total_budget=100000,
        channels=['TV', 'Digital', 'Social'],
        constraints={
            'TV_min': 20000,
            'Digital_min': 10000,
            'Social_min': 5000
        },
        base_data=base_data
    )
    
    assert optimal_budget['TV'] >= 20000
    assert optimal_budget['Digital'] >= 10000
    assert optimal_budget['Social'] >= 5000


def test_scenario_comparison(trained_model):
    """Test scenario comparison."""
    optimizer = BudgetOptimizer(trained_model, method='scipy')
    
    base_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-12-31')],
        'gdp_growth': [0.02],
        'unemployment': [0.05]
    })
    
    scenarios_df = optimizer.compare_scenarios(
        total_budget=100000,
        channels=['TV', 'Digital', 'Social'],
        base_data=base_data
    )
    
    assert 'scenario' in scenarios_df.columns
    assert 'total_revenue' in scenarios_df.columns
    assert len(scenarios_df) > 0

