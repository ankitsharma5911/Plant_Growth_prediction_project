from src.exception import CustomException
from src.logger import logging
import sys

try :
    a = 1/0
except Exception as e:
    logging.info("exception testing")
    print(CustomException(e,sys))
    print("done")
    