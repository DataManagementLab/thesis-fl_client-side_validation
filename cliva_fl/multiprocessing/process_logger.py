import logging
from torch.multiprocessing import log_to_stderr

def get_process_logger():
    logger = log_to_stderr()
    logger.setLevel(logging.INFO)
    return logger