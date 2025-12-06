"""
Example End-to-End MMM Pipeline

Demonstrates the complete workflow:
1. Data collection
2. Data validation
3. Feature engineering
4. Model training
5. Budget optimization
6. Monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data_collection import (
    SalesDataCollector,
    MediaSpendCollector,
    ExternalFactorsCollector,
    DataAggregator,
    DataValidator
)
from src.modeling import MMMModel
from src.optimization import BudgetOptimizer
from src.monitoring import ModelMonitor
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run complete MMM pipeline."""
    
    logger.info("=" * 70)
    logger.info("MMM PIPELINE - END-TO-END EXAMPLE")
    logger.info("=" * 70)
    
    # 1. Data Collection
    logger.info("\n[1/6] Collecting data...")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    sales_collector = SalesDataCollector()
    media_collector = MediaSpendCollector()
    external_collector = ExternalFactorsCollector()
    
    aggregator = DataAggregator([sales_collector, media_collector, external_collector])
    data_dict = aggregator.collect_all(start_date, end_date)
    merged_data = aggregator.merge_data(data_dict)
    
    logger.info(f"Collected {len(merged_data)} records")
    
    # 2. Data Validation
    logger.info("\n[2/6] Validating data...")
    
    validator = DataValidator()
    spend_columns = [col for col in merged_data.columns if col.startswith('spend_')]
    validation_results = validator.validate_all(
        merged_data,
        spend_columns=spend_columns
    )
    
    if not validation_results['is_valid']:
        logger.warning("Data validation found issues:")
        print(validator.get_validation_report())
    else:
        logger.info("Data validation passed")
    
    # 3. Feature Engineering (handled in model)
    logger.info("\n[3/6] Feature engineering (adstock & saturation)...")
    
    # 4. Model Training
    logger.info("\n[4/6] Training MMM model...")
    
    model = MMMModel(
        media_channels=['TV', 'Radio', 'Digital', 'Social', 'Search', 'Display'],
        external_factors=['gdp_growth', 'unemployment', 'holiday_flag'],
        adstock_params={
            'TV': 0.6,
            'Radio': 0.5,
            'Digital': 0.4,
            'Social': 0.3,
            'Search': 0.2,
            'Display': 0.3
        },
        saturation_params={
            'TV': {'method': 'hill', 'alpha': 0.5, 'gamma': 0.5},
            'Digital': {'method': 'hill', 'alpha': 0.7, 'gamma': 0.3},
            'Social': {'method': 'hill', 'alpha': 0.6, 'gamma': 0.4}
        }
    )
    
    model.train(merged_data)
    
    # Get channel ROI
    roi_df = model.get_channel_roi(merged_data)
    logger.info("\nChannel ROI Analysis:")
    print(roi_df.to_string(index=False))
    
    # 5. Budget Optimization
    logger.info("\n[5/6] Optimizing budget allocation...")
    
    optimizer = BudgetOptimizer(model, method='scipy')
    
    optimal_budget = optimizer.optimize(
        total_budget=1000000,
        channels=['TV', 'Digital', 'Social', 'Search'],
        constraints={
            'TV_min': 10000,
            'Digital_min': 5000,
            'Social_min': 2000,
            'Search_min': 3000
        },
        base_data=merged_data.tail(1)  # Use recent data as base
    )
    
    logger.info("\nOptimal Budget Allocation:")
    for channel, spend in optimal_budget.items():
        logger.info(f"  {channel}: ${spend:,.2f}")
    
    # Compare scenarios
    scenarios_df = optimizer.compare_scenarios(
        total_budget=1000000,
        channels=['TV', 'Digital', 'Social', 'Search'],
        base_data=merged_data.tail(1)
    )
    
    logger.info("\nScenario Comparison:")
    print(scenarios_df.to_string(index=False))
    
    # 6. Monitoring
    logger.info("\n[6/6] Setting up monitoring...")
    
    monitor = ModelMonitor(model)
    metrics = monitor.monitor_performance(merged_data, window_days=30)
    
    logger.info("\nMonitoring Report:")
    print(monitor.generate_report())
    
    # Check retraining trigger
    should_retrain, reason = monitor.check_retraining_trigger()
    if should_retrain:
        logger.warning(f"Retraining recommended: {reason}")
    else:
        logger.info("Model performance is acceptable")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

