# MMM Pipeline Architecture

## Overview

This document describes the architecture of the Marketing Mix Modeling (MMM) production pipeline.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Sales   │  │  Media   │  │ External │  │  Other   │  │
│  │  Data    │  │  Spend   │  │ Factors  │  │  Sources  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Collection & Validation                    │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   Collectors     │  │   Validators     │                │
│  │  - Sales         │  │  - Completeness  │                │
│  │  - Media         │  │  - Ranges        │                │
│  │  - External      │  │  - Anomalies     │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Adstock    │  │  Saturation  │  │  Time-based  │     │
│  │  (Carryover) │  │  (Returns)   │  │  Features    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      MMM Modeling                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Bayesian Regression Model                    │   │
│  │  - Channel contributions                             │   │
│  │  - Incrementality measurement                        │   │
│  │  - Uncertainty quantification                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Forecasting │  │ Optimization │  │  Monitoring  │
│   Engine     │  │   Engine     │  │  & Alerts   │
└─────────────┘  └─────────────┘  └─────────────┘
```

## Component Details

### 1. Data Collection Layer

**Purpose**: Ingest data from multiple sources

**Components**:
- `SalesDataCollector`: Collects sales/revenue data
- `MediaSpendCollector`: Collects media spend by channel
- `ExternalFactorsCollector`: Collects external factors (economic, seasonal)
- `DataAggregator`: Merges data from multiple sources

**Key Features**:
- Modular design for easy extension
- Handles missing data gracefully
- Supports multiple data formats

### 2. Data Validation Layer

**Purpose**: Ensure data quality and detect anomalies

**Components**:
- `DataValidator`: Comprehensive validation checks
  - Completeness checks
  - Range validation
  - Anomaly detection (Z-score, IQR, Isolation Forest)

**Key Features**:
- Automated quality checks
- Anomaly detection
- Validation reports

### 3. Feature Engineering Layer

**Purpose**: Transform raw data into model features

**Components**:
- `AdstockTransformer`: Models carryover effects
- `SaturationTransformer`: Models diminishing returns

**Key Features**:
- Multiple adstock methods (geometric, Weibull, exponential)
- Multiple saturation curves (Hill, exponential, log)
- Parameter estimation from data

### 4. Modeling Layer

**Purpose**: Build and train MMM models

**Components**:
- `MMMModel`: Core MMM implementation
  - Bayesian Ridge regression
  - Channel contribution decomposition
  - ROI calculation

**Key Features**:
- Bayesian inference for uncertainty
- Automatic feature preparation
- Model persistence

### 5. Optimization Layer

**Purpose**: Optimize media budget allocation

**Components**:
- `BudgetOptimizer`: Constrained optimization
  - Multiple optimization methods (scipy, PuLP)
  - Scenario comparison
  - Constraint handling

**Key Features**:
- Multiple optimization algorithms
- Business constraint support
- Scenario planning

### 6. Monitoring Layer

**Purpose**: Track model performance and detect issues

**Components**:
- `ModelMonitor`: Performance tracking
  - Performance metrics (R², RMSE, MAE, MAPE)
  - Data drift detection
  - Retraining triggers

**Key Features**:
- Automated monitoring
- Drift detection
- Retraining recommendations

### 7. AWS Integration Layer

**Purpose**: Cloud storage and compute integration

**Components**:
- `S3Handler`: S3 operations
  - Data upload/download
  - Model artifact storage

**Key Features**:
- Seamless S3 integration
- Model versioning support

## Data Flow

1. **Collection**: Data collected from various sources
2. **Validation**: Data quality checks performed
3. **Feature Engineering**: Adstock and saturation applied
4. **Training**: Model trained on historical data
5. **Optimization**: Budget allocation optimized
6. **Monitoring**: Performance tracked continuously
7. **Retraining**: Model retrained when needed

## Scalability

- **Modular Design**: Components can be scaled independently
- **Configuration-Driven**: YAML configs for easy customization
- **Cloud-Ready**: AWS integration for scalable storage/compute
- **Multi-Product**: Designed to handle multiple products/LoBs

## Extensibility

- Easy to add new data sources
- Pluggable feature engineering methods
- Customizable optimization objectives
- Flexible monitoring rules

