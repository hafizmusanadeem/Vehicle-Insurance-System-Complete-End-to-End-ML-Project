import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame

class VehicleData:
    def __init__(self,
                 id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Annual_Premium, Policy_Sales_Channel, Vintage,Vehicle_Age, Vehicle_Damage):
        """
    
        Description: Vehicle Data constructor
        Input: This method get all features of the trained model for prediction

        """
        try:
            self.id = id
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        """
        
        This method returns a dictionary from VehicleData class input

        """
        try:
            input_data = {
                "id":[self.id],
                "Gender":[self.Gender],
                "Age":[self.Age],
                "Driving_License":[self.Driving_License],
                "Region_Code":[self.Region_Code],
                "Previously_Insured":[self.Previously_Insured],
                "Vehicle_Age":[self.Vehicle_Age],
                "Vehicle_Damage":[self.Vehicle_Damage],
                "Annual_Premium":[self.Annual_Premium],
                "Policy_Sales_Channel":[self.Policy_Sales_Channel],
                "Vintage":[self.Vintage]
            }
            logging.info("Created Vehicle Input Data Dictionary")
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_input_data_frame(self) -> DataFrame:
        """

        This method returns a DataFrame from VehicleData class input
        
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e
    
class VehicleDataClassifier:

    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig() ) -> None:
        """
        
        This method is used for the prediction pipeline config

        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys) from e
        
    def predict(self, dataframe) -> str:
        """
        
        This method returns a prediction on the Input Data

        """
        try:
            logging.info("Entered the Prediction_Method of VehicleDataClassifier")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            logging.info("Going for Prediction")
            result = model.predict(dataframe)
            logging.info("Result is Loaded")
            return result
        
        except Exception as e:
            raise MyException(e, sys) from e
        

        
