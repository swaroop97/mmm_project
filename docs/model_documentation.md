# MMM Model Documentation

## Model Overview

The Marketing Mix Model (MMM) measures the incremental impact of marketing channels on sales/revenue using Bayesian regression.

## Model Equation

```
Revenue(t) = Base + Σ(Channel_Contribution_i(t)) + External_Factors(t) + ε(t)
```

Where:
- `Base`: Baseline sales (intercept)
- `Channel_Contribution_i(t)`: Contribution from channel i at time t
- `External_Factors(t)`: Impact of external factors
- `ε(t)`: Error term

## Channel Contribution

Each channel's contribution is calculated as:

```
Channel_Contribution(t) = β_i × Adstock(Spend_i(t)) × Saturation(Spend_i(t))
```

Where:
- `β_i`: Channel coefficient (learned from data)
- `Adstock()`: Carryover effect transformation
- `Saturation()`: Diminishing returns transformation

## Adstock Transformation

Models how advertising effects decay over time:

**Geometric Adstock**:
```
Adstock(t) = Σ(λ^k × Spend(t-k)) for k = 0 to max_lag
```

Where `λ` is the decay rate (0-1).

## Saturation Transformation

Models diminishing returns:

**Hill Function**:
```
Saturation(x) = (x^α) / (γ^α + x^α)
```

Where:
- `α`: Saturation parameter
- `γ`: Half-saturation point

## Model Training

1. **Feature Preparation**:
   - Apply adstock to media spend
   - Apply saturation curves
   - Add external factors
   - Add time-based features

2. **Bayesian Regression**:
   - Uses Bayesian Ridge regression
   - Provides uncertainty estimates
   - Regularizes coefficients

3. **Evaluation**:
   - R² score
   - RMSE
   - MAE
   - MAPE

## Model Outputs

1. **Channel Coefficients**: Impact of each channel
2. **ROI**: Return on investment per channel
3. **Contributions**: Decomposed sales by channel
4. **Predictions**: Future revenue forecasts

## Model Assumptions

1. **Linearity**: Linear relationship between transformed spend and revenue
2. **Additivity**: Channel effects are additive
3. **Stationarity**: Relationships stable over time
4. **No Interaction**: Channels don't interact (can be extended)

## Limitations

1. **Attribution**: Cannot measure individual customer-level attribution
2. **Causality**: Correlation-based, not causal
3. **Time Period**: Requires sufficient historical data
4. **External Factors**: May miss unobserved factors

## Model Validation

- **Holdout Testing**: Test on unseen data
- **Cross-Validation**: Time-series cross-validation
- **Residual Analysis**: Check for patterns in errors
- **Coefficient Stability**: Monitor over time

## Model Maintenance

- **Retraining**: Quarterly or when performance degrades
- **Monitoring**: Continuous performance tracking
- **Drift Detection**: Alert on data distribution changes
- **Version Control**: Track model versions

