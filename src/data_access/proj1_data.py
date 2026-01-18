import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException
from src.logger.logger import logging  # Using your configured logger

class Proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self, database_name: Optional[str] = DATABASE_NAME) -> None:
        """
        Initializes the MongoDB database connection using the class-level MongoDBClient.
        """
        try:
            self.database_name = database_name
            self.db = MongoDBClient.connect(database_name=self.database_name)
            logging.info(f"MongoDB database '{self.database_name}' connected successfully in Proj1Data.")
        except Exception as e:
            logging.error("Failed to initialize MongoDB connection in Proj1Data", exc_info=True)
            raise MyException(e, sys)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to self.database_name.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Use specified database or default database
            db_to_use = (
                MongoDBClient.get_database()
                if database_name is None
                else MongoDBClient.connect(database_name=database_name)
            )

            logging.info(f"Fetching collection '{collection_name}' from database '{db_to_use.name}'...")

            # Access the collection
            collection = db_to_use[collection_name]

            # Fetch data and convert to DataFrame
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Fetched {len(df)} records from collection '{collection_name}'.")

            # Drop '_id' column if exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logging.debug(f"Dropped '_id' column from collection '{collection_name}'.")

            # Replace "na" strings with np.nan
            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Collection '{collection_name}' successfully converted to DataFrame.")
            return df

        except Exception as e:
            logging.error(f"Failed to export collection '{collection_name}' as DataFrame.", exc_info=True)
            raise MyException(e, sys)
