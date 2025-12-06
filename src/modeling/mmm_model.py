"""
Marketing Mix Modeling (MMM) Core Module

Implements Bayesian MMM for measuring marketing effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class MMMModel:
    """
    Marketing Mix Model using Bayesian regression.
    
    Models sales as a function of:
    - Media channels (with adstock and saturation)
    - External factors
    - Base sales
    """
    
    def __init__(
        self,
        media_channels: List[str],
        external_factors: Optional[List[str]] = None,
        adstock_params: Optional[Dict[str, float]] = None,
        saturation_params: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize MMM model.
        
        Args:
            media_channels: List of media channel names
            external_factors: List of external factor names
            adstock_params: Dict mapping channels to decay rates
            saturation_params: Dict mapping channels to saturation params
        """
        self.media_channels = media_channels
        self.external_factors = external_factors or []
        self.adstock_params = adstock_params or {}
        self.saturation_params = saturation_params or {}
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.coefficients = {}
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'revenue'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of target variable
            
        Returns:
            (features_df, target_series)
        """
        self.logger.info("Preparing features for MMM")
        
        features_list = []
        
        # Add media channels (with adstock and saturation if specified)
        for channel in self.media_channels:
            spend_col = f'spend_{channel.lower()}'
            
            if spend_col not in df.columns:
                self.logger.warning(f"Channel {channel} not found in data")
                continue
            
            # Apply adstock if specified
            if channel in self.adstock_params:
                from ..feature_engineering.adstock import AdstockTransformer
                decay = self.adstock_params[channel]
                adstock = AdstockTransformer(decay_rate=decay)
                adstock_col = f'{spend_col}_adstock'
                df[adstock_col] = adstock.transform(df[spend_col])
                spend_col = adstock_col
            
            # Apply saturation if specified
            if channel in self.saturation_params:
                from ..feature_engineering.saturation import SaturationTransformer
                sat_params = self.saturation_params[channel]
                sat = SaturationTransformer(**sat_params)
                sat_col = f'{spend_col}_saturated'
                df[sat_col] = sat.transform(df[spend_col])
                spend_col = sat_col
            
            features_list.append(spend_col)
        
        # Add external factors
        for factor in self.external_factors:
            if factor in df.columns:
                features_list.append(factor)
        
        # Add time-based features
        if date_column in df.columns:
            df['day_of_week'] = pd.to_datetime(df[date_column]).dt.dayofweek
            df['month'] = pd.to_datetime(df[date_column]).dt.month
            df['quarter'] = pd.to_datetime(df[date_column]).dt.quarter
            features_list.extend(['day_of_week', 'month', 'quarter'])
        
        # Create features DataFrame
        features_df = df[features_list].copy()
        self.feature_names = features_list
        
        # Target variable
        if value_column not in df.columns:
            raise ValueError(f"Target column {value_column} not found")
        
        target = df[value_column].copy()
        
        self.logger.info(f"Prepared {len(features_list)} features")
        
        return features_df, target
    
    def train(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'revenue'
    ) -> 'MMMModel':
        """
        Train the MMM model.
        
        Args:
            df: Training data
            date_column: Name of date column
            value_column: Name of target variable
            
        Returns:
            Self for chaining
        """
        self.logger.info("Training MMM model")
        
        # Prepare features
        X, y = self.prepare_features(df, date_column, value_column)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Bayesian Ridge regression
        self.model = BayesianRidge(
            n_iter=300,
            compute_score=True,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        
        self.model.fit(X_scaled, y)
        
        # Extract coefficients
        self.coefficients = dict(zip(self.feature_names, self.model.coef_))
        
        # Calculate R-squared
        y_pred = self.model.predict(X_scaled)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
        self.logger.info(f"Model trained. R² = {r2:.4f}")
        self.logger.info(f"Number of features: {len(self.feature_names)}")
        
        return self
    
    def predict(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            df: Input data
            date_column: Name of date column
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features (same as training)
        X, _ = self.prepare_features(df, date_column, value_column='revenue')
        
        # Ensure same features as training
        X = X[[col for col in self.feature_names if col in X.columns]]
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def decompose_contributions(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Decompose sales into channel contributions.
        
        Args:
            df: Input data
            date_column: Name of date column
            
        Returns:
            DataFrame with contributions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X, _ = self.prepare_features(df, date_column, value_column='revenue')
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X_scaled = self.scaler.transform(X)
        
        # Calculate contributions
        contributions = {}
        
        for i, feature in enumerate(self.feature_names):
            if feature in X.columns:
                contributions[feature] = X_scaled[:, i] * self.coefficients[feature]
        
        contributions_df = pd.DataFrame(contributions)
        
        # Add date if available
        if date_column in df.columns:
            contributions_df[date_column] = df[date_column].values
        
        # Add total predicted
        contributions_df['predicted_revenue'] = self.predict(df, date_column)
        
        # Add base (intercept)
        contributions_df['base'] = self.model.intercept_
        
        return contributions_df
    
    def get_channel_roi(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Calculate ROI for each channel.
        
        Args:
            df: Input data
            date_column: Name of date column
            
        Returns:
            DataFrame with ROI metrics
        """
        contributions = self.decompose_contributions(df, date_column)
        
        roi_data = []
        
        for channel in self.media_channels:
            spend_col = f'spend_{channel.lower()}'
            
            if spend_col not in df.columns:
                continue
            
            # Find contribution column
            contrib_col = None
            for col in contributions.columns:
                if channel.lower() in col.lower() and 'spend' in col.lower():
                    contrib_col = col
                    break
            
            if contrib_col is None:
                continue
            
            total_spend = df[spend_col].sum()
            total_contribution = contributions[contrib_col].sum()
            
            if total_spend > 0:
                roi = total_contribution / total_spend
                roi_data.append({
                    'channel': channel,
                    'total_spend': total_spend,
                    'total_contribution': total_contribution,
                    'roi': roi,
                    'incremental_revenue': total_contribution
                })
        
        return pd.DataFrame(roi_data)
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'coefficients': self.coefficients,
            'media_channels': self.media_channels,
            'external_factors': self.external_factors,
            'adstock_params': self.adstock_params,
            'saturation_params': self.saturation_params
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MMMModel':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            media_channels=model_data['media_channels'],
            external_factors=model_data['external_factors'],
            adstock_params=model_data['adstock_params'],
            saturation_params=model_data['saturation_params']
        )
        
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.coefficients = model_data['coefficients']
        
        return instance


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(10000, 2000, len(dates)),
        'spend_tv': np.random.uniform(1000, 5000, len(dates)),
        'spend_digital': np.random.uniform(500, 3000, len(dates)),
        'spend_social': np.random.uniform(200, 1500, len(dates))
    })
    
    # Initialize model
    model = MMMModel(
        media_channels=['TV', 'Digital', 'Social'],
        adstock_params={'TV': 0.6, 'Digital': 0.5, 'Social': 0.4},
        saturation_params={
            'TV': {'method': 'hill', 'alpha': 0.5, 'gamma': 0.5},
            'Digital': {'method': 'hill', 'alpha': 0.6, 'gamma': 0.4}
        }
    )
    
    # Train
    model.train(df)
    
    # Get ROI
    roi_df = model.get_channel_roi(df)
    print("\nChannel ROI:")
    print(roi_df)

