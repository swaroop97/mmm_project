"""
Model Training Script

Trains MMM model from configuration file.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime

from src.modeling import MMMModel
from src.utils.config import load_config
from src.data_collection import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(config_path: str, data_path: str, output_dir: str = "models"):
    """
    Train MMM model.
    
    Args:
        config_path: Path to model configuration YAML
        data_path: Path to training data CSV
        output_dir: Directory to save trained model
    """
    logger.info("=" * 70)
    logger.info("MMM MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    model_config = config.get('model', {})
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Validate data
    logger.info("Validating data...")
    validator = DataValidator()
    spend_columns = [col for col in data.columns if col.startswith('spend_')]
    validation_results = validator.validate_all(data, spend_columns=spend_columns)
    
    if not validation_results['is_valid']:
        logger.warning("Data validation found issues:")
        print(validator.get_validation_report())
        logger.warning("Continuing with training despite validation issues...")
    
    # Extract configuration
    media_channels = model_config.get('media_channels', [])
    external_factors = model_config.get('external_factors', [])
    
    adstock_config = model_config.get('adstock', {})
    adstock_params = adstock_config.get('decay_rates', {})
    
    saturation_config = model_config.get('saturation', {})
    saturation_params = {}
    for channel in media_channels:
        if channel in saturation_config.get('parameters', {}):
            saturation_params[channel] = saturation_config['parameters'][channel]
    
    # Initialize model
    logger.info("Initializing MMM model...")
    model = MMMModel(
        media_channels=media_channels,
        external_factors=external_factors,
        adstock_params=adstock_params,
        saturation_params=saturation_params
    )
    
    # Train model
    logger.info("Training model...")
    model.train(data)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model.evaluate(data)
    
    logger.info("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Get channel ROI
    roi_df = model.get_channel_roi(data)
    logger.info("\nChannel ROI Analysis:")
    print(roi_df.to_string(index=False))
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"mmm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics_path = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MMM model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="Path to model configuration YAML")
    parser.add_argument("--data", type=str, default="data/raw/sample_data.csv",
                       help="Path to training data CSV")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained model")
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir
    )

