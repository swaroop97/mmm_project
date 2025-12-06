"""
AWS S3 Integration Module

Handles data storage and retrieval from S3.
"""

import boto3
import pandas as pd
from typing import Optional, List
from pathlib import Path
import logging
from io import BytesIO, StringIO
import json

logger = logging.getLogger(__name__)


class S3Handler:
    """
    Handle S3 operations for MMM pipeline.
    """
    
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = 'us-east-1'
    ):
        """
        Initialize S3 handler.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key (if None, uses default credentials)
            aws_secret_access_key: AWS secret key (if None, uses default credentials)
            region_name: AWS region
        """
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # Use default credentials
            self.s3_client = boto3.client('s3', region_name=region_name)
        
        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Connected to S3 bucket: {bucket_name}")
        except Exception as e:
            self.logger.warning(f"Could not verify bucket {bucket_name}: {e}")
    
    def upload_file(
        self,
        local_path: str,
        s3_key: str
    ) -> bool:
        """
        Upload file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            self.logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading to S3: {e}")
            return False
    
    def download_file(
        self,
        s3_key: str,
        local_path: str
    ) -> bool:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            
        Returns:
            True if successful
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            self.logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading from S3: {e}")
            return False
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        s3_key: str,
        format: str = 'csv'
    ) -> bool:
        """
        Upload DataFrame to S3.
        
        Args:
            df: DataFrame to upload
            s3_key: S3 object key
            format: File format ('csv', 'parquet', 'json')
            
        Returns:
            True if successful
        """
        try:
            buffer = BytesIO()
            
            if format == 'csv':
                df.to_csv(buffer, index=False)
            elif format == 'parquet':
                df.to_parquet(buffer, index=False)
            elif format == 'json':
                df.to_json(buffer, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            
            self.logger.info(f"Uploaded DataFrame to s3://{self.bucket_name}/{s3_key}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error uploading DataFrame to S3: {e}")
            return False
    
    def download_dataframe(
        self,
        s3_key: str,
        format: str = 'csv'
    ) -> Optional[pd.DataFrame]:
        """
        Download DataFrame from S3.
        
        Args:
            s3_key: S3 object key
            format: File format ('csv', 'parquet', 'json')
            
        Returns:
            DataFrame or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            buffer = BytesIO(response['Body'].read())
            
            if format == 'csv':
                df = pd.read_csv(buffer)
            elif format == 'parquet':
                df = pd.read_parquet(buffer)
            elif format == 'json':
                df = pd.read_json(buffer, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Downloaded DataFrame from s3://{self.bucket_name}/{s3_key}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error downloading DataFrame from S3: {e}")
            return None
    
    def list_files(
        self,
        prefix: str = '',
        suffix: Optional[str] = None
    ) -> List[str]:
        """
        List files in S3 bucket.
        
        Args:
            prefix: Key prefix to filter
            suffix: Optional suffix to filter
            
        Returns:
            List of S3 keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if suffix is None or key.endswith(suffix):
                            keys.append(key)
            
            return keys
        
        except Exception as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return []
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting from S3: {e}")
            return False


if __name__ == "__main__":
    # Example usage (requires AWS credentials)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize (would use actual bucket name and credentials)
    # s3 = S3Handler(bucket_name='my-mmm-bucket')
    
    # Upload example
    # df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    # s3.upload_dataframe(df, 'data/processed/sample.csv')
    
    # Download example
    # df = s3.download_dataframe('data/processed/sample.csv')
    # print(df)
    
    print("S3Handler module loaded. Configure with your AWS credentials to use.")

