import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_object, save_numpy_array_data
from src.components.transformers import GenderMapper, ColumnDropper   


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.ingestion_artifact = data_ingestion_artifact
            self.validation_artifact = data_validation_artifact
            self.config = data_transformation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)


    def get_preprocessor(self) -> Pipeline:
        """
        Returns a fully sklearn-native preprocessing pipeline
        """
        try:
            numeric_features = self.schema["num_features"]
            minmax_features = self.schema["mm_columns"]
            categorical_features = self.schema["categorical_columns"]
            drop_columns = self.schema["drop_columns"]

            numeric_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            minmax_pipeline = Pipeline(steps=[
                ("scaler", MinMaxScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_features),
                    ("mm", minmax_pipeline, minmax_features),
                    ("cat", categorical_pipeline, categorical_features),
                ],
                remainder="passthrough"
            )

            full_pipeline = Pipeline(steps=[
                ("gender_mapper", GenderMapper()),
                ("drop_columns", ColumnDropper(drop_columns)),
                ("preprocessor", preprocessor),
            ])

            return full_pipeline

        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.validation_artifact.validation_status:
                raise Exception(self.validation_artifact.message)

            logging.info("Starting Data Transformation")

            train_df = pd.read_csv(self.ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            pipeline = self.get_preprocessor()

            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)

            # Applying SMOTE on training data to handle imbalanced distribution
            smote = SMOTEENN(sampling_strategy="minority")
            result = smote.fit_resample(X_train_transformed, y_train)
            X_train_resampled, y_train_resampled = result[0], result[1]

            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_transformed, y_test]

            save_object(self.config.transformed_object_file_path, pipeline)
            save_numpy_array_data(self.config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.config.transformed_test_file_path, test_arr)

            logging.info("Data Transformation Completed Successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.config.transformed_object_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path,
            )

        except Exception as e:
            raise MyException(e, sys)




