import time, os, random, torch, gc

from pathlib import Path
from .process_logger import get_process_logger
from cliva_fl import models
from cliva_fl.validation import validate_buffer
from cliva_fl.utils import TimeTracker, Logger, partial_class


def validation_process(queue, logger: Logger, **validation_kwargs):
    log = get_process_logger()
    log.info('Starting validator => {}'.format(os.getpid()))
    tf_id = f'validation[pid {os.getpid()}]'

    time_tracker = TimeTracker()
    time_tracker.start_timeframe(tf_id)
    
    model_cnf, optimizer_cnf = logger.load_config('model', 'optimizer')
    model = getattr(models, model_cnf['type'])(**model_cnf['params'])
    optimizer = getattr(torch.optim, optimizer_cnf['type'])(model.parameters(), **optimizer_cnf['params'])
    # model_builder = partial_class(model_cnf['type'], getattr(models, model_cnf['type']), **model_cnf['params'])
    # optimizer_builder = partial_class(optimizer_cnf['type'], getattr(torch.optim, optimizer_cnf['type']), **optimizer_cnf['params'])

    while True:
        buffer = queue.get()

        if buffer is None:
            log.info('Validator {} exiting...'.format(os.getpid()))
            time_tracker.stop_timeframe(tf_id)
            logger.save_timeframe(tf_id, **time_tracker.get_timeframe(tf_id, format="%Y-%m-%d %H:%M:%S"))
            return
        if type(buffer) == str:
            buffer = logger.get_queue(buffer)
        
        # log.info('{} got {} of epoch {}'.format(os.getpid(), buffer.size(), buffer.epoch))
        validate_buffer(buffer, logger=logger, time_tracker=time_tracker, model=model, optimizer=optimizer, **validation_kwargs)
        logger.log_times(buffer.epoch, time_tracker.total_times_history)
        time_tracker.clear()
        gc.collect()


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
