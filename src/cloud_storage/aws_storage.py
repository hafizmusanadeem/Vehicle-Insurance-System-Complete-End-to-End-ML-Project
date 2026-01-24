import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union,List
import os,sys
from src.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from src.exception import MyException
from botocore.exceptions import ClientError
from pandas import DataFrame,read_csv
import pickle


class SimpleStorageService:
    """
    A class for interacting with AWS S3 storage, providing methods for file management, 
    data uploads, and data retrieval in S3 buckets.
    """

    def __init__(self):
        """
        Initializes the SimpleStorageService instance with S3 resource and client
        from the S3Client class.
        """
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Key path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise MyException(e, sys)
        
    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Loads a serialized model from the specified S3 bucket.

        Args:
            model_name (str): Name of the model file in the bucket.
            bucket_name (str): Name of the S3 bucket.
            model_dir (str): Directory path within the bucket.

        Returns:
            object: The deserialized model object.
        """
        try:
            model_file = model_dir + "/" + model_name if model_dir else model_name
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Production model loaded from S3 bucket.")
            return model
        except Exception as e:
            raise MyException(e, sys) from e
        
    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """
        Uploads a local file to the specified S3 bucket with an optional file deletion.

        Args:
            from_filename (str): Path of the local file.
            to_filename (str): Target file path in the bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool): If True, deletes the local file after upload.
        """
        logging.info("Entered the upload_file method of SimpleStorageService class")
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name}")

            # Delete the local file if remove is True
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
            logging.info("Exited the upload_file method of SimpleStorageService class")
        except Exception as e:
            raise MyException(e, sys) from e