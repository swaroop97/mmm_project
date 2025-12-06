# AWS Deployment Guide

This guide covers deploying the MMM pipeline to AWS infrastructure.

## Overview

The MMM pipeline can be deployed on AWS using:
- **S3**: Data storage and model artifacts
- **SageMaker**: Model training and inference
- **Redshift**: Data warehouse for historical data
- **Lambda/ECS**: API endpoints for real-time inference
- **EventBridge**: Scheduled retraining jobs

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured (`aws configure`)
3. Python 3.9+ with dependencies installed
4. Docker (for containerized deployment)

## S3 Setup

### 1. Create S3 Bucket

```bash
aws s3 mb s3://mmm-data-bucket --region us-east-1
```

### 2. Configure Bucket Structure

```
s3://mmm-data-bucket/
├── raw/              # Raw input data
├── processed/        # Processed features
├── models/           # Trained model artifacts
├── results/          # Optimization results
└── monitoring/       # Monitoring metrics
```

### 3. Upload Data

```python
from src.aws import S3Handler

s3 = S3Handler(bucket_name='mmm-data-bucket')
s3.upload_dataframe(df, 'raw/sales_data.csv')
```

## SageMaker Deployment

### 1. Prepare Training Script

The training script should be compatible with SageMaker's training framework:

```python
# train_sagemaker.py
import argparse
import os
import joblib
from src.modeling import MMMModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(f'{args.train}/data.csv')
    
    # Train model
    model = MMMModel(...)
    model.train(data)
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.pkl'))
```

### 2. Create SageMaker Training Job

```python
import sagemaker
from sagemaker.sklearn import SKLearn

role = 'arn:aws:iam::ACCOUNT:role/SageMakerRole'
sagemaker_session = sagemaker.Session()

estimator = SKLearn(
    entry_point='train_sagemaker.py',
    role=role,
    instance_type='ml.m5.xlarge',
    framework_version='0.24-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

estimator.fit({'training': 's3://mmm-data-bucket/processed/training/'})
```

### 3. Deploy Model Endpoint

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Test endpoint
result = predictor.predict(data)
```

## Redshift Integration

### 1. Connect to Redshift

```python
import redshift_connector

conn = redshift_connector.connect(
    host='your-cluster.region.redshift.amazonaws.com',
    database='mmm_db',
    user='admin',
    password='password',
    port=5439
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM sales_data LIMIT 10")
```

### 2. Load Data from Redshift

```python
import pandas as pd

query = """
    SELECT date, revenue, spend_tv, spend_digital
    FROM marketing_data
    WHERE date >= '2023-01-01'
"""

df = pd.read_sql(query, conn)
```

## Lambda Function for Inference

### 1. Create Lambda Function

```python
# lambda_function.py
import json
import joblib
import boto3
import pandas as pd

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Load model from S3
    model_path = '/tmp/model.pkl'
    s3.download_file('mmm-data-bucket', 'models/latest_model.pkl', model_path)
    model = joblib.load(model_path)
    
    # Parse input
    data = pd.DataFrame([event['data']])
    
    # Predict
    prediction = model.predict(data)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction.tolist()})
    }
```

### 2. Deploy Lambda

```bash
zip -r lambda_function.zip lambda_function.py
aws lambda create-function \
    --function-name mmm-inference \
    --runtime python3.9 \
    --role arn:aws:iam::ACCOUNT:role/lambda-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda_function.zip
```

## Scheduled Retraining

### 1. Create EventBridge Rule

```bash
aws events put-rule \
    --name mmm-retrain-schedule \
    --schedule-expression "rate(30 days)" \
    --state ENABLED
```

### 2. Add Lambda Target

```bash
aws events put-targets \
    --rule mmm-retrain-schedule \
    --targets "Id"="1","Arn"="arn:aws:lambda:REGION:ACCOUNT:function:retrain-model"
```

## Monitoring with CloudWatch

### 1. Log Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='MMM/Pipeline',
    MetricData=[
        {
            'MetricName': 'ModelR2Score',
            'Value': r2_score,
            'Unit': 'None'
        }
    ]
)
```

### 2. Set Up Alarms

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name mmm-low-r2 \
    --alarm-description "Alert when R² drops below 0.7" \
    --metric-name ModelR2Score \
    --namespace MMM/Pipeline \
    --statistic Average \
    --period 3600 \
    --threshold 0.7 \
    --comparison-operator LessThanThreshold
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "example_pipeline.py"]
```

### 2. Build and Push to ECR

```bash
docker build -t mmm-pipeline .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag mmm-pipeline:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mmm-pipeline:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mmm-pipeline:latest
```

## Best Practices

1. **Security**: Use IAM roles with least privilege
2. **Cost Optimization**: Use spot instances for training
3. **Monitoring**: Set up CloudWatch dashboards
4. **Versioning**: Tag model artifacts with versions
5. **Backup**: Regularly backup model artifacts and data
6. **Testing**: Test deployments in staging environment first

## Troubleshooting

### Common Issues

1. **Permission Errors**: Check IAM role permissions
2. **Timeout Errors**: Increase Lambda timeout or use ECS
3. **Memory Issues**: Increase instance size or optimize code
4. **Data Access**: Verify S3 bucket policies and VPC configurations

## Next Steps

- Set up CI/CD pipeline with GitHub Actions or AWS CodePipeline
- Implement A/B testing for model versions
- Add data quality monitoring
- Create automated alerting system

