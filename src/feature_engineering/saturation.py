"""
Saturation Curve Module

Implements saturation (diminishing returns) curves for marketing channels.
Models how incremental impact decreases as spend increases.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class SaturationTransformer:
    """
    Transform media spend using saturation curves.
    
    Models diminishing returns: each additional dollar has less impact.
    """
    
    def __init__(
        self,
        method: str = 'hill',
        alpha: float = 0.5,
        gamma: float = 0.5
    ):
        """
        Initialize saturation transformer.
        
        Args:
            method: Method ('hill', 'exponential', 'log')
            alpha: Saturation parameter (for Hill function)
            gamma: Shape parameter (for Hill function)
        """
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.logger = logging.getLogger(__name__)
    
    def _hill_function(self, x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
        """
        Hill saturation function.
        
        f(x) = (x^alpha) / (gamma^alpha + x^alpha)
        
        Args:
            x: Input values
            alpha: Saturation parameter
            gamma: Half-saturation point
            
        Returns:
            Saturated values
        """
        return (x ** alpha) / (gamma ** alpha + x ** alpha)
    
    def _exponential_saturation(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        Exponential saturation function.
        
        f(x) = 1 - exp(-alpha * x)
        
        Args:
            x: Input values
            alpha: Saturation rate
            
        Returns:
            Saturated values
        """
        return 1 - np.exp(-alpha * x)
    
    def _log_saturation(self, x: np.ndarray) -> np.ndarray:
        """
        Logarithmic saturation function.
        
        f(x) = log(1 + x)
        
        Args:
            x: Input values
            
        Returns:
            Saturated values
        """
        return np.log1p(x)
    
    def transform(
        self,
        spend: Union[np.ndarray, pd.Series],
        max_spend: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply saturation transformation.
        
        Args:
            spend: Media spend values
            max_spend: Maximum spend for normalization (if None, uses max of data)
            
        Returns:
            Saturated spend values
        """
        # Convert to numpy
        if isinstance(spend, pd.Series):
            spend = spend.values
        
        # Normalize if max_spend provided
        if max_spend is None:
            max_spend = spend.max()
        
        if max_spend == 0:
            return np.zeros_like(spend)
        
        normalized_spend = spend / max_spend
        
        # Apply saturation
        if self.method == 'hill':
            saturated = self._hill_function(normalized_spend, self.alpha, self.gamma)
            # Scale back to original range
            saturated = saturated * max_spend
        
        elif self.method == 'exponential':
            saturated = self._exponential_saturation(normalized_spend, self.alpha)
            saturated = saturated * max_spend
        
        elif self.method == 'log':
            saturated = self._log_saturation(normalized_spend)
            saturated = saturated * max_spend
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return saturated
    
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        spend_columns: list,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Apply saturation to multiple columns.
        
        Args:
            df: DataFrame with spend data
            spend_columns: List of column names to transform
            date_column: Name of date column
            
        Returns:
            DataFrame with saturated columns
        """
        result_df = df.copy()
        
        for col in spend_columns:
            if col not in result_df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
            
            max_spend = result_df[col].max()
            saturated_col = f"{col}_saturated"
            result_df[saturated_col] = self.transform(result_df[col], max_spend)
            
            self.logger.info(f"Applied saturation to {col}, created {saturated_col}")
        
        return result_df


def estimate_saturation_parameters(
    spend: np.ndarray,
    response: np.ndarray,
    method: str = 'hill'
) -> dict:
    """
    Estimate saturation parameters from data.
    
    Args:
        spend: Media spend time series
        response: Response variable
        method: Saturation method
        
    Returns:
        Dictionary with estimated parameters
    """
    if method == 'hill':
        def hill_func(x, alpha, gamma):
            max_x = x.max()
            norm_x = x / max_x
            return (norm_x ** alpha) / (gamma ** alpha + norm_x ** alpha) * max_x
        
        try:
            popt, _ = curve_fit(
                hill_func,
                spend,
                response,
                bounds=([0.1, 0.1], [2.0, 1.0]),
                maxfev=1000
            )
            return {'alpha': popt[0], 'gamma': popt[1]}
        except:
            return {'alpha': 0.5, 'gamma': 0.5}  # Default
    
    elif method == 'exponential':
        def exp_func(x, alpha):
            max_x = x.max()
            norm_x = x / max_x
            return (1 - np.exp(-alpha * norm_x)) * max_x
        
        try:
            popt, _ = curve_fit(exp_func, spend, response, bounds=(0.1, 10.0))
            return {'alpha': popt[0]}
        except:
            return {'alpha': 1.0}  # Default
    
    return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    spend = np.linspace(0, 10000, 100)
    
    # Apply saturation
    transformer = SaturationTransformer(method='hill', alpha=0.5, gamma=0.5)
    saturated = transformer.transform(spend)
    
    # Compare
    df = pd.DataFrame({
        'spend': spend,
        'saturated_spend': saturated,
        'incremental': np.diff(np.concatenate([[0], saturated]))
    })
    
    print("\nSaturation Transformation Example:")
    print(df.head(10))
    print(f"\nOriginal spend max: {spend.max():.2f}")
    print(f"Saturated spend max: {saturated.max():.2f}")
    print(f"\nDiminishing returns visible in incremental column")

