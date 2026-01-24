import sys
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact is the Output of the Model Evaluation stage
        :param model_pusher_config is the Configuration for pushing the model to cloud
        """
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            # Initializing the estimator which handles S3 communication
            self.proj1_estimator = Proj1Estimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path
            )
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Description : The method pushes the trained model to S3 if it passed evaluation.
        
        Output      : It returns the ModelPusherArtifact.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            # CRITICAL CHECK: Only push if the model was accepted during evaluation
            if self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model evaluation passed. Proceeding to upload model to S3.")
                
                # Uploading the model file to S3
                self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
                
                model_pusher_artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path
                )
                
                logging.info(f"Model successfully pushed to S3: {model_pusher_artifact}")
            else:
                logging.info("Model evaluation failed (is_model_accepted=False). Model will not be pushed to S3.")
                # We return the artifact with the existing S3 path but flag it was not updated
                model_pusher_artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path
                )

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e