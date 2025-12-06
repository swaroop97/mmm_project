# Budget Optimization Guide

## Overview

The budget optimization engine allocates media budgets across channels to maximize ROI, revenue, or profit subject to business constraints.

## Optimization Problem

**Objective**: Maximize ROI (or revenue/profit)

**Subject to**:
- Total budget constraint: `Σ(spend_i) = Total_Budget`
- Channel minimums: `spend_i ≥ min_spend_i`
- Channel maximums: `spend_i ≤ max_spend_i`
- Percentage constraints: `min_pct_i ≤ spend_i / Total_Budget ≤ max_pct_i`

## Optimization Methods

### 1. Scipy (Sequential Least Squares Programming)

**Pros**:
- Handles non-linear objectives
- Supports constraints
- Fast for small-medium problems

**Cons**:
- May get stuck in local optima
- Requires good initial guess

### 2. PuLP (Linear Programming)

**Pros**:
- Guaranteed global optimum (for linear problems)
- Fast
- Good for large problems

**Cons**:
- Requires linearization of MMM response function
- Less accurate for non-linear relationships

### 3. Differential Evolution

**Pros**:
- Global optimization
- No gradient required
- Good for complex landscapes

**Cons**:
- Slower
- More iterations needed

## Usage Example

```python
from src.optimization import BudgetOptimizer

optimizer = BudgetOptimizer(model, method='scipy')

optimal = optimizer.optimize(
    total_budget=1000000,
    channels=['TV', 'Digital', 'Social'],
    constraints={
        'TV_min': 10000,
        'TV_max': 500000,
        'Digital_min': 5000,
        'Social_min': 2000
    }
)
```

## Constraint Types

### Absolute Constraints
- `channel_min`: Minimum spend for channel
- `channel_max`: Maximum spend for channel

### Percentage Constraints
- `channel_min_pct`: Minimum percentage of total budget
- `channel_max_pct`: Maximum percentage of total budget

## Scenario Planning

Compare multiple scenarios:

```python
scenarios = optimizer.compare_scenarios(
    total_budget=1000000,
    channels=['TV', 'Digital', 'Social']
)
```

Scenarios include:
- Equal allocation
- Optimized allocation
- Custom scenarios

## Best Practices

1. **Start with reasonable constraints**: Don't over-constrain
2. **Validate results**: Check if optimal allocation makes business sense
3. **Test sensitivity**: Vary constraints to see impact
4. **Consider uncertainty**: Use confidence intervals from Bayesian model
5. **Update regularly**: Re-optimize as data and model change

## Common Issues

1. **Infeasible solution**: Constraints too tight
2. **Poor performance**: Model may need retraining
3. **Unrealistic allocation**: Check model coefficients

