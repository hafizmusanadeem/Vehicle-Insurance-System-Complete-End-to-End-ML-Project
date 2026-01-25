import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from src.exception import MyException
from src.logger import logging
from typing import Any

class TargetValueMapping:
    def __init__(self):
        # Professional standard: use a single dictionary for mapping
        self.mapping = {
            "yes": 0,
            "no": 1
        }
        # Pre-calculate reverse mapping to save time
        self.reverse_mapping_dict = {v: k for k, v in self.mapping.items()}

    def to_dict(self):
        return self.mapping

    def reverse_mapping(self):
        return self.reverse_mapping_dict

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame):
        """
        Applies preprocessing and returns model predictions.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Transform features
            # Many models fail if the input isn't exactly as expected (e.g., column names missing)
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Predict
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature) # type: ignore

            return predictions

        except Exception as e:
            # Using logging.error is good, but MyException already handles sys/traceback
            raise MyException(e, sys) from e

    def __repr__(self):
        # Improved repr to show both preprocessor and model info
        return f"MyModel(preprocessor={type(self.preprocessing_object).__name__}, model={type(self.trained_model_object).__name__})"