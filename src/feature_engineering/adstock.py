"""
Adstock Transformation Module

Implements adstock (carryover) effects for marketing channels.
Adstock models how advertising effects decay over time.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.signal import convolve
import logging

logger = logging.getLogger(__name__)


class AdstockTransformer:
    """
    Transform media spend using adstock (carryover) effects.
    
    Adstock models the carryover effect of advertising:
    - Immediate effect from current period
    - Decaying effect from previous periods
    """
    
    def __init__(
        self,
        decay_rate: float = 0.5,
        max_lag: int = 4,
        method: str = 'geometric'
    ):
        """
        Initialize adstock transformer.
        
        Args:
            decay_rate: Decay rate (0-1). Higher = slower decay
            max_lag: Maximum number of periods for carryover
            method: Method ('geometric', 'weibull', 'exponential')
        """
        if not 0 < decay_rate < 1:
            raise ValueError("decay_rate must be between 0 and 1")
        
        self.decay_rate = decay_rate
        self.max_lag = max_lag
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def _geometric_weights(self) -> np.ndarray:
        """
        Generate geometric decay weights.
        
        Returns:
            Array of weights
        """
        weights = np.array([self.decay_rate ** i for i in range(self.max_lag + 1)])
        return weights / weights.sum()  # Normalize
    
    def _weibull_weights(self, shape: float = 1.0) -> np.ndarray:
        """
        Generate Weibull decay weights.
        
        Args:
            shape: Weibull shape parameter
            
        Returns:
            Array of weights
        """
        from scipy.stats import weibull_min
        
        scale = self.max_lag / np.log(1 / self.decay_rate)
        x = np.arange(self.max_lag + 1)
        
        weights = weibull_min.pdf(x, shape, scale=scale)
        return weights / weights.sum()  # Normalize
    
    def _exponential_weights(self) -> np.ndarray:
        """
        Generate exponential decay weights.
        
        Returns:
            Array of weights
        """
        lambda_param = -np.log(self.decay_rate)
        weights = np.array([np.exp(-lambda_param * i) for i in range(self.max_lag + 1)])
        return weights / weights.sum()  # Normalize
    
    def get_weights(self) -> np.ndarray:
        """
        Get adstock weights based on method.
        
        Returns:
            Array of weights
        """
        if self.method == 'geometric':
            return self._geometric_weights()
        elif self.method == 'weibull':
            return self._weibull_weights()
        elif self.method == 'exponential':
            return self._exponential_weights()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def transform(
        self,
        spend: Union[np.ndarray, pd.Series],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply adstock transformation to spend data.
        
        Args:
            spend: Media spend time series
            weights: Optional custom weights (if None, uses method weights)
            
        Returns:
            Adstock-transformed spend
        """
        if weights is None:
            weights = self.get_weights()
        
        # Convert to numpy array
        if isinstance(spend, pd.Series):
            spend = spend.values
        
        # Apply convolution (moving weighted average)
        adstock = convolve(spend, weights, mode='same')
        
        return adstock
    
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        spend_columns: list,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Apply adstock to multiple columns in DataFrame.
        
        Args:
            df: DataFrame with spend data
            spend_columns: List of column names to transform
            date_column: Name of date column (for sorting)
            
        Returns:
            DataFrame with adstock-transformed columns
        """
        result_df = df.copy()
        
        # Sort by date
        if date_column in result_df.columns:
            result_df = result_df.sort_values(date_column).reset_index(drop=True)
        
        weights = self.get_weights()
        
        for col in spend_columns:
            if col not in result_df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
            
            adstock_col = f"{col}_adstock"
            result_df[adstock_col] = self.transform(result_df[col], weights)
            
            self.logger.info(f"Applied adstock to {col}, created {adstock_col}")
        
        return result_df


def estimate_adstock_decay(
    spend: np.ndarray,
    response: np.ndarray,
    max_decay: float = 0.9,
    min_decay: float = 0.1
) -> float:
    """
    Estimate optimal adstock decay rate from data.
    
    Uses correlation between lagged spend and response.
    
    Args:
        spend: Media spend time series
        response: Response variable (sales, conversions, etc.)
        max_decay: Maximum decay rate to test
        min_decay: Minimum decay rate to test
        
    Returns:
        Estimated optimal decay rate
    """
    from scipy.optimize import minimize_scalar
    
    def objective(decay):
        """Objective: maximize correlation with response."""
        transformer = AdstockTransformer(decay_rate=decay)
        adstock_spend = transformer.transform(spend)
        correlation = np.corrcoef(adstock_spend, response)[0, 1]
        return -correlation  # Minimize negative correlation
    
    result = minimize_scalar(
        objective,
        bounds=(min_decay, max_decay),
        method='bounded'
    )
    
    return result.x


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample spend data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    spend = np.random.uniform(1000, 5000, len(dates))
    
    # Apply adstock
    transformer = AdstockTransformer(decay_rate=0.6, max_lag=4)
    adstock_spend = transformer.transform(spend)
    
    # Compare
    df = pd.DataFrame({
        'date': dates,
        'original_spend': spend,
        'adstock_spend': adstock_spend
    })
    
    print("\nAdstock Transformation Example:")
    print(df.head(10))
    print(f"\nOriginal spend sum: {spend.sum():.2f}")
    print(f"Adstock spend sum: {adstock_spend.sum():.2f}")
    print(f"\nWeights used: {transformer.get_weights()}")

