
import sys
from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    DEBUG
)
def get_logger(name):
    logger = getLogger(name)

    logger.setLevel(DEBUG)
    logger.propagate = False # to avoid duplicated log
    # handler = StreamHandler(sys.stdout)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s %(levelname)7s %(filename)12s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger