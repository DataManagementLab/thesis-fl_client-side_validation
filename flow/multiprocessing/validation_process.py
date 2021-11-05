import time, os, random

from pathlib import Path
from .process_logger import get_process_logger
from flow.validation import validate_buffer
from flow.utils import TimeTracker, Logger


def validation_process(queue, logger: Logger, **validation_kwargs):
    log = get_process_logger()
    log.info('Starting validator => {}'.format(os.getpid()))
    tf_id = f'validation[pid {os.getpid()}]'

    time_tracker = TimeTracker()
    time_tracker.start_timeframe(tf_id)

    while True:
        buffer = queue.get()

        if buffer is None:
            log.info('Validator {} exiting...'.format(os.getpid()))
            time_tracker.stop_timeframe(tf_id)
            logger.save_timeframe(tf_id, **time_tracker.get_timeframe(tf_id, format="%Y-%m-%d %H:%M:%S"))
            return
        if type(buffer) == str:
            buffer = logger.get_queue(buffer)
        
        epoch = next(iter(buffer.values())).epoch
        #log.info('{} got {} of epoch {}'.format(os.getpid(), len(buffer), epoch))
        validate_buffer(buffer, logger=logger, time_tracker=time_tracker, **validation_kwargs)
        logger.log_times(epoch, time_tracker.total_times_history)
        time_tracker.clear()


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
