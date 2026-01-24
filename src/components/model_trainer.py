import sys
from typing import Tuple, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[Any, ClassificationMetricArtifact]:
        """
        Trains the model and returns the model object along with a detailed metric report.
        """
        try:
            logging.info("Splitting features and target for train and test sets")
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state
            )

            logging.info("Fitting the RandomForest model...")
            model.fit(x_train, y_train)

            # Performance on Test Data
            y_test_pred = model.predict(x_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # Performance on Train Data (to check for overfitting)
            y_train_pred = model.predict(x_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            logging.info(f"Model trained. Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

            # Check for Overfitting (threshold can be added to config)
            if abs(train_acc - test_acc) > 0.15: # 15% threshold
                logging.warning("Warning: Model might be overfitted. Difference between train and test accuracy is high.")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_test_pred),
                precision_score=precision_score(y_test, y_test_pred),
                recall_score=recall_score(y_test, y_test_pred)
            )

            return model, metric_artifact, test_acc
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer Component")
            
            # Load data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            # Train and evaluate
            trained_model, metric_artifact, test_acc = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Threshold Check
            if test_acc < self.model_trainer_config.expected_accuracy:
                message = f"Model accuracy {test_acc} is lower than base accuracy {self.model_trainer_config.expected_accuracy}"
                logging.info(message)
                raise Exception(message)

            # Final Wrapper
            logging.info("Wrapping model and preprocessor into MyModel object")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            
            save_object(self.model_trainer_config.trained_model_file_path, my_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
        
        except Exception as e:
            raise MyException(e, sys) from e