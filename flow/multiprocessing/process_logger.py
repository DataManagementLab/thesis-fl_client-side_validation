import logging
from multiprocessing import log_to_stderr, get_logger

def get_process_logger():
    logger = log_to_stderr()
    logger.setLevel(logging.INFO)
    return logger