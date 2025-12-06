"""
Budget Optimization Script

Optimizes media budget allocation using trained MMM model.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import joblib
import json

from src.optimization import BudgetOptimizer
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_budget(
    model_path: str,
    config_path: str,
    data_path: str,
    total_budget: float,
    horizon: int = 12,
    output_path: str = "data/results/optimization_results.csv"
):
    """
    Optimize media budget allocation.
    
    Args:
        model_path: Path to trained MMM model
        config_path: Path to optimization configuration YAML
        data_path: Path to base data for optimization
        total_budget: Total budget to allocate
        horizon: Forecast horizon in months
        output_path: Path to save optimization results
    """
    logger.info("=" * 70)
    logger.info("BUDGET OPTIMIZATION")
    logger.info("=" * 70)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    opt_config = config.get('optimization', {})
    
    # Load base data
    logger.info(f"Loading base data from {data_path}")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    base_data = data.tail(1)  # Use most recent data point
    
    # Initialize optimizer
    method = opt_config.get('method', 'scipy')
    objective = opt_config.get('objective', 'roi')
    
    logger.info(f"Initializing optimizer (method={method}, objective={objective})")
    optimizer = BudgetOptimizer(model, method=method, objective=objective)
    
    # Get channels from model
    channels = model.media_channels
    
    # Get constraints from config
    channel_constraints = opt_config.get('channel_constraints', {})
    constraints = {}
    for channel in channels:
        if channel in channel_constraints:
            ch_config = channel_constraints[channel]
            if 'min_spend' in ch_config:
                constraints[f'{channel}_min'] = ch_config['min_spend']
            if 'max_spend' in ch_config:
                constraints[f'{channel}_max'] = ch_config['max_spend']
    
    # Optimize
    logger.info(f"Optimizing budget allocation (total=${total_budget:,.2f})...")
    optimal_budget = optimizer.optimize(
        total_budget=total_budget,
        channels=channels,
        constraints=constraints,
        base_data=base_data
    )
    
    logger.info("\nOptimal Budget Allocation:")
    total_allocated = 0
    for channel, spend in optimal_budget.items():
        logger.info(f"  {channel}: ${spend:,.2f} ({spend/total_budget*100:.1f}%)")
        total_allocated += spend
    logger.info(f"  Total: ${total_allocated:,.2f}")
    
    # Compare scenarios
    logger.info("\nComparing scenarios...")
    scenarios_df = optimizer.compare_scenarios(
        total_budget=total_budget,
        channels=channels,
        base_data=base_data
    )
    
    logger.info("\nScenario Comparison:")
    print(scenarios_df.to_string(index=False))
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save optimal allocation
    results_df = pd.DataFrame([
        {'channel': k, 'optimal_spend': v, 'pct_of_total': v/total_budget*100}
        for k, v in optimal_budget.items()
    ])
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    # Save scenarios
    scenarios_path = output_path.parent / "scenarios_comparison.csv"
    scenarios_df.to_csv(scenarios_path, index=False)
    logger.info(f"Scenarios comparison saved to {scenarios_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    
    return optimal_budget, scenarios_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize media budget")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained MMM model")
    parser.add_argument("--config", type=str, default="config/optimization_config.yaml",
                       help="Path to optimization configuration YAML")
    parser.add_argument("--data", type=str, default="data/raw/sample_data.csv",
                       help="Path to base data CSV")
    parser.add_argument("--budget", type=float, required=True,
                       help="Total budget to allocate")
    parser.add_argument("--horizon", type=int, default=12,
                       help="Forecast horizon in months")
    parser.add_argument("--output", type=str, default="data/results/optimization_results.csv",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    optimize_budget(
        model_path=args.model,
        config_path=args.config,
        data_path=args.data,
        total_budget=args.budget,
        horizon=args.horizon,
        output_path=args.output
    )

