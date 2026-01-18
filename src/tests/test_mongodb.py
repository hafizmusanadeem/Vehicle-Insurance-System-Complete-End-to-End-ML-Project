import os
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import MyException
from src.constants import MONGODB_URL_KEY, DATABASE_NAME

# Test 1: Successful connection
def test_successful_connection(mocker):
    mocker.patch.dict(os.environ, {MONGODB_URL_KEY: "mongodb://fake-url"})
    mock_client = mocker.MagicMock()
    mock_client.admin.command.return_value = {"ok": 1}
    mock_client.__getitem__.return_value = "mock_database"
    mocker.patch("src.configuration.mongo_db_connection.MongoClient", return_value=mock_client)

    db = MongoDBClient.connect(DATABASE_NAME)
    assert db == "mock_database"
    mock_client.admin.command.assert_called_once_with("ping")


# Test 2: Missing environment variable
def test_missing_env_variable():
    MongoDBClient._client = None
    MongoDBClient._database = None
    os.environ.pop(MONGODB_URL_KEY, None)
    with pytest.raises(MyException):
        MongoDBClient.connect()


# Test 3: MongoDB Timeout / Atlas Down
def test_connection_timeout(mocker):
    MongoDBClient._client = None
    MongoDBClient._database = None
    mocker.patch.dict(os.environ, {MONGODB_URL_KEY: "mongodb://fake-url"})
    mock_client = mocker.MagicMock()
    mock_client.admin.command.side_effect = ServerSelectionTimeoutError("Server not reachable")
    mocker.patch("src.configuration.mongo_db_connection.MongoClient", return_value=mock_client)

    with pytest.raises(MyException):
        MongoDBClient.connect()


# Test 4: get_database() triggers auto-connect
def test_get_database_triggers_connection(mocker):
    MongoDBClient._client = None
    MongoDBClient._database = None
    mocker.patch.dict(os.environ, {MONGODB_URL_KEY: "mongodb://fake-url"})
    mock_client = mocker.MagicMock()
    mock_client.admin.command.return_value = {"ok": 1}
    mock_client.__getitem__.return_value = "mock_db"
    mocker.patch("src.configuration.mongo_db_connection.MongoClient", return_value=mock_client)

    db = MongoDBClient.get_database()
    assert db == "mock_db"


# Test 5: close connection safely
def test_close_connection(mocker):
    mock_client = mocker.MagicMock()
    MongoDBClient._client = mock_client
    MongoDBClient._database = "mock_db"
    MongoDBClient.close()
    mock_client.close.assert_called_once()
    assert MongoDBClient._client is None
    assert MongoDBClient._database is None
