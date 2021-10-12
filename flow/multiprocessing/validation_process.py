import time, os, random

from pathlib import Path
from .process_logger import get_process_logger
from flow.validation import validate_buffer


def validation_process(queue, logger, **validation_kwargs):
    log = get_process_logger()
    log.info('Starting validator => {}'.format(os.getpid()))

    while True:
        buffer = queue.get()

        if buffer is None:
            log.info('Validator {} exiting...'.format(os.getpid()))
            return
        
        log.info('{} got {}'.format(os.getpid(), len(buffer)))
        validate_buffer(buffer, logger=logger, **validation_kwargs)


def consumer_process(queue, logger, **validation_kwarg):
    log = get_process_logger()
    log.info('Starting consumer => {}'.format(os.getpid()))

    while True:
        time.sleep(random.randint(0, 2))

        buffer = queue.get()

        if buffer is None:
            log.info('Consumer {} exiting...'.format(os.getpid()))
            return
        
        log.info('{} got {}'.format(os.getpid(), 'buffer'))
