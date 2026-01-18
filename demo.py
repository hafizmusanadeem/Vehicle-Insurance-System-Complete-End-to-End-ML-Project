# below code is to check the logging config
# from src.logger import logging

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# --------------------------------------------------------------------------------

# below code is to check the exception config
# import sys
# from src.logger import logging
# from src.exception import MyException

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e


# from src.logger.logger import configure_logger
# from src.exception import MyException
# import sys

# configure_logger()  # ✅ must call at start

# try:
#     a = 2+2
# except Exception as e:
#     raise MyException(e, sys)

# --------------------------------------------------------------------------------

from src.pipline.training_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()