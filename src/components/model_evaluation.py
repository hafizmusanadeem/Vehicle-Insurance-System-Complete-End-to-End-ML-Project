import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Description: This method retrieves a working model (if-any) from production (S3) bucket

        Output: It returns the model estimator.  
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Description: This method compares the retrieved model from S3 bucket with the currently trained model.

        Output: This method returns the performing model's and the degraded model's metrics & their comparison.
        """
        try:
            
            logging.info("Loading raw test data for evaluation")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Loading the newly trained model from artifacts")
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            
            y_hat_trained_model = trained_model.predict(x)
            trained_model_f1_score = f1_score(y, y_hat_trained_model)
            
            best_model_f1_score = 0.0
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info("Production model found. Computing F1 score for comparison...")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            else:
                logging.info("No model found in S3. New model will be accepted by default.")

            difference = trained_model_f1_score - best_model_f1_score
            is_model_accepted = difference > self.model_eval_config.changed_threshold_score

            logging.info(f"New Model F1: {trained_model_f1_score}, Best Model F1: {best_model_f1_score}")
            
            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Description: This method Orchestrates the evaluation process.

        Output: It returns the model_evaluation artifacts.
        """
        try:
            logging.info("Starting Model Evaluation")
            evaluate_model_response = self.evaluate_model()
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Evaluation complete. Model accepted: {model_evaluation_artifact.is_model_accepted}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e