import time, os, random, torch

from pathlib import Path
from .process_logger import get_process_logger
from flow import models
from flow.validation import validate_buffer
from flow.utils import TimeTracker, Logger, partial_class


def validation_process(queue, logger: Logger, **validation_kwargs):
    log = get_process_logger()
    log.info('Starting validator => {}'.format(os.getpid()))
    tf_id = f'validation[pid {os.getpid()}]'

    time_tracker = TimeTracker()
    time_tracker.start_timeframe(tf_id)
    device = torch.device('cpu')
    model_cnf, optimizer_cnf = logger.load_config('model', 'optimizer')
    model_builder = partial_class(model_cnf['type'], getattr(models, model_cnf['type']), **model_cnf['params'])
    optimizer_builder = partial_class(optimizer_cnf['type'], getattr(torch.optim, optimizer_cnf['type']), **optimizer_cnf['params'])

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
        validate_buffer(buffer, logger=logger, time_tracker=time_tracker, model_builder=model_builder, optimizer_builder=optimizer_builder, **validation_kwargs)
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
