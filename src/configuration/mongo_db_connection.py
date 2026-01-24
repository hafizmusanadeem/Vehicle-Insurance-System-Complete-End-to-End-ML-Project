import os
import sys
import certifi
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.constants import DATABASE_NAME, MONGODB_URL_KEY
from src.logger.logger import configure_logger
from src.exception import MyException


# Initialize logging once
configure_logger()
logger = logging.getLogger(__name__)

# Certificate Authority file (required for MongoDB Atlas TLS)
CA_FILE = certifi.where()


class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a single shared MongoDB connection
    for the entire application lifecycle.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """

    _client = None
    _database = None

    @classmethod
    def connect(cls, database_name: str = DATABASE_NAME):
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set..
        """
        try:
            if cls._client is None:
                logger.debug("Initializing MongoDB connection")

                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if not mongo_db_url:
                    raise ValueError(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set"
                    )

                cls._client = MongoClient(
                    mongo_db_url,
                    tlsCAFile=CA_FILE,
                    serverSelectionTimeoutMS=50000,
                    connectTimeoutMS=100000,
                    socketTimeoutMS=100000,
                    maxPoolSize=50,
                    retryWrites=True
                )

                # Force a connection check
                cls._client.admin.command("ping")

                cls._database = cls._client[database_name]
                logger.info(f"MongoDB connected successfully to database '{database_name}'")

            return cls._database

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error("MongoDB connection failure", exc_info=True)
            cls._client = None
            cls._database = None
            raise MyException(e, sys)

        except Exception as e:
            logger.critical("Unexpected MongoDB error", exc_info=True)
            cls._client = None
            cls._database = None
            raise MyException(e, sys)

    @classmethod
    def get_database(cls):
        """
        Returns the active database instance.
        """
        try:
            if cls._database is None:
                logger.warning("MongoDB not connected. Establishing connection.")
                return cls.connect()

            return cls._database

        except Exception as e:
            logger.error("Failed to retrieve MongoDB database", exc_info=True)
            raise MyException(e, sys)

    @classmethod
    def close(cls):
        """
        Gracefully close the MongoDB connection.
        """
        try:
            if cls._client:
                cls._client.close()
                logger.debug("MongoDB connection closed")
                cls._client = None
                cls._database = None

        except Exception as e:
            logger.error("Error while closing MongoDB connection", exc_info=True)
            raise MyException(e, sys)
