import os
import logging
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# Constant for Log Configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y+%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3

# Construct Log File Path
def configure_logger():
    """ Creates a Logging File Directory if not exists already and Configures logging with a rotating file handler and a console handler"""
    log_dir_path = os.path.join(from_root(), LOG_DIR)
    os.makedirs(log_dir_path, exist_ok=True)

    # File Path
    log_file_path = os.path.join(log_dir_path, LOG_FILE)
    
    # Logger Object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return
    
    # Define Formatter
    formatter = logging.Formatter(
        "[ %(asctime)s %(name)s = %(levelname)s - %(message)s]"
    )

    # File Handler with Rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Returning the Handlers back to Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
