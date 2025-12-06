"""
Media Budget Optimization Engine

Optimizes media budget allocation across channels to maximize ROI
subject to business constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy.optimize import minimize, differential_evolution
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BudgetOptimizer:
    """
    Optimize media budget allocation using constrained optimization.
    """
    
    def __init__(
        self,
        mmm_model,
        method: str = 'scipy',
        objective: str = 'roi'
    ):
        """
        Initialize budget optimizer.
        
        Args:
            mmm_model: Trained MMM model
            method: Optimization method ('scipy', 'pulp', 'differential_evolution')
            objective: Objective function ('roi', 'revenue', 'profit')
        """
        self.mmm_model = mmm_model
        self.method = method
        self.objective = objective
        self.logger = logging.getLogger(__name__)
    
    def _predict_revenue_from_spend(
        self,
        channel_spend: Dict[str, float],
        base_data: pd.DataFrame
    ) -> float:
        """
        Predict revenue from channel spend allocation.
        
        Args:
            channel_spend: Dict mapping channel names to spend amounts
            base_data: Base DataFrame with other features
            
        Returns:
            Predicted revenue
        """
        # Create prediction DataFrame
        pred_df = base_data.copy()
        
        # Update spend columns
        for channel, spend in channel_spend.items():
            spend_col = f'spend_{channel.lower()}'
            if spend_col in pred_df.columns:
                pred_df[spend_col] = spend
            else:
                # Add if missing
                pred_df[spend_col] = spend
        
        # Predict
        predictions = self.mmm_model.predict(pred_df)
        
        return predictions.sum()
    
    def optimize_scipy(
        self,
        total_budget: float,
        channels: List[str],
        constraints: Optional[Dict] = None,
        base_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Optimize using scipy.optimize.
        
        Args:
            total_budget: Total budget to allocate
            channels: List of channel names
            constraints: Dict with constraints (min_spend, max_spend per channel)
            base_data: Base DataFrame for predictions
            
        Returns:
            Dict mapping channels to optimal spend
        """
        self.logger.info(f"Optimizing budget allocation using scipy (total: ${total_budget:,.0f})")
        
        constraints_dict = constraints or {}
        n_channels = len(channels)
        
        # Initial guess: equal allocation
        x0 = np.array([total_budget / n_channels] * n_channels)
        
        # Objective function: maximize revenue (minimize negative revenue)
        def objective(x):
            channel_spend = dict(zip(channels, x))
            revenue = self._predict_revenue_from_spend(channel_spend, base_data)
            return -revenue  # Minimize negative = maximize
        
        # Constraints
        constraint_list = []
        
        # Budget constraint: sum equals total
        constraint_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - total_budget
        })
        
        # Min/max per channel
        for i, channel in enumerate(channels):
            min_spend = constraints_dict.get(f'{channel}_min', 0)
            max_spend = constraints_dict.get(f'{channel}_max', total_budget)
            
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, idx=i, min_val=min_spend: x[idx] - min_val
            })
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, idx=i, max_val=max_spend: max_val - x[idx]
            })
        
        # Bounds: non-negative
        bounds = [(0, total_budget) for _ in channels]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_spend = dict(zip(channels, result.x))
            predicted_revenue = -result.fun
            
            self.logger.info(f"Optimization successful. Predicted revenue: ${predicted_revenue:,.0f}")
            
            return optimal_spend
        else:
            self.logger.warning("Optimization failed, using equal allocation")
            return {ch: total_budget / n_channels for ch in channels}
    
    def optimize_pulp(
        self,
        total_budget: float,
        channels: List[str],
        constraints: Optional[Dict] = None,
        base_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Optimize using PuLP (linear programming).
        
        Args:
            total_budget: Total budget to allocate
            channels: List of channel names
            constraints: Dict with constraints
            base_data: Base DataFrame for predictions
            
        Returns:
            Dict mapping channels to optimal spend
        """
        self.logger.info(f"Optimizing budget allocation using PuLP (total: ${total_budget:,.0f})")
        
        # Create problem
        prob = LpProblem("Budget_Optimization", LpMaximize)
        
        # Decision variables
        spend_vars = {}
        for channel in channels:
            min_spend = constraints.get(f'{channel}_min', 0) if constraints else 0
            max_spend = constraints.get(f'{channel}_max', total_budget) if constraints else total_budget
            spend_vars[channel] = LpVariable(
                f"spend_{channel}",
                lowBound=min_spend,
                upBound=max_spend,
                cat='Continuous'
            )
        
        # Objective: maximize revenue (simplified linear approximation)
        # In practice, would need to linearize the MMM response function
        # For demo, use ROI coefficients from model
        objective_terms = []
        for channel in channels:
            # Get coefficient from model (simplified)
            coeff = self.mmm_model.coefficients.get(
                f'spend_{channel.lower()}',
                0.1  # Default ROI
            )
            objective_terms.append(coeff * spend_vars[channel])
        
        prob += lpSum(objective_terms)
        
        # Budget constraint
        prob += lpSum([spend_vars[ch] for ch in channels]) == total_budget
        
        # Solve
        prob.solve()
        
        # Extract solution
        optimal_spend = {}
        for channel in channels:
            optimal_spend[channel] = value(spend_vars[channel])
        
        self.logger.info("Optimization completed")
        
        return optimal_spend
    
    def optimize(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        constraints: Optional[Dict] = None,
        base_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Optimize budget allocation.
        
        Args:
            total_budget: Total budget to allocate
            channels: List of channel names (if None, uses model channels)
            constraints: Dict with constraints
            base_data: Base DataFrame for predictions
            
        Returns:
            Dict mapping channels to optimal spend
        """
        channels = channels or self.mmm_model.media_channels
        
        if base_data is None:
            # Create minimal base data
            base_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=1, freq='D')
            })
        
        if self.method == 'scipy':
            return self.optimize_scipy(total_budget, channels, constraints, base_data)
        elif self.method == 'pulp':
            return self.optimize_pulp(total_budget, channels, constraints, base_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compare_scenarios(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        constraints: Optional[Dict] = None,
        base_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compare different budget allocation scenarios.
        
        Args:
            total_budget: Total budget
            channels: List of channels
            constraints: Constraints dict
            base_data: Base DataFrame
            
        Returns:
            DataFrame comparing scenarios
        """
        channels = channels or self.mmm_model.media_channels
        
        scenarios = {}
        
        # Current allocation (equal)
        equal_allocation = {ch: total_budget / len(channels) for ch in channels}
        scenarios['Equal Allocation'] = equal_allocation
        
        # Optimized allocation
        optimal_allocation = self.optimize(total_budget, channels, constraints, base_data)
        scenarios['Optimized'] = optimal_allocation
        
        # Compare
        results = []
        for scenario_name, allocation in scenarios.items():
            revenue = self._predict_revenue_from_spend(allocation, base_data)
            total_spend = sum(allocation.values())
            roi = revenue / total_spend if total_spend > 0 else 0
            
            results.append({
                'scenario': scenario_name,
                'total_revenue': revenue,
                'total_spend': total_spend,
                'roi': roi,
                **{f'{ch}_spend': allocation.get(ch, 0) for ch in channels}
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would use a trained MMM model
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
    
    # Optimize
    optimizer = BudgetOptimizer(model, method='scipy')
    
    optimal = optimizer.optimize(
        total_budget=1000000,
        channels=['TV', 'Digital'],
        constraints={'TV_min': 10000, 'Digital_min': 5000}
    )
    
    print("\nOptimal Budget Allocation:")
    for channel, spend in optimal.items():
        print(f"  {channel}: ${spend:,.2f}")

