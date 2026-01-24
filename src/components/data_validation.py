import json
import sys
import os

import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def _validate_columns(self, df: DataFrame, dataset_name: str) -> list[str]:
        """
        Description: This method Validate column names against schema.
        Output: Returns list of error messages.
        """
        errors = []
        logging.info("Entered the Column Validation Method")

        schema_columns = []
        for column_dict in self.schema["columns"]:
            schema_columns.extend(list(column_dict.keys()))
        expected_columns = set(schema_columns)
        actual_columns = set(df.columns)

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        if missing:
            errors.append(
                f"{dataset_name}: Missing columns: {sorted(missing)}"
            )

        if extra:
            errors.append(
                f"{dataset_name}: Unexpected columns: {sorted(extra)}"
            )

        return errors

    def _validate_column_groups(self, df: DataFrame, dataset_name: str) -> list[str]:
        """
        Description: This method validates the existence of numerical and categorical columns
        
        Output: Returns list of errors if any.
        """
        errors = []

        for col in self.schema["numerical_columns"]:
            if col not in df.columns:
                errors.append(f"{dataset_name}: Missing numerical column '{col}'")

        for col in self.schema["categorical_columns"]:
            if col not in df.columns:
                errors.append(f"{dataset_name}: Missing categorical column '{col}'")

        return errors

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Description: This method initiates the data validation component for the pipeline
        
        Output: Returns bool value based on validation results
        On Failure: Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
           
            errors = []

            errors.extend(self._validate_columns(train_df, "Train"))
            errors.extend(self._validate_columns(test_df, "Test"))

            errors.extend(self._validate_column_groups(train_df, "Train"))
            errors.extend(self._validate_column_groups(test_df, "Test"))

            validation_status = len(errors) == 0
            message = "; ".join(errors)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            os.makedirs(
                os.path.dirname(self.data_validation_config.validation_report_file_path),
                exist_ok=True
            )

            with open(artifact.validation_report_file_path, "w") as f:
                json.dump(
                    {
                        "validation_status": validation_status,
                        "errors": errors
                    },
                    f,
                    indent=4
                )

            logging.info("Data validation completed successfully")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
