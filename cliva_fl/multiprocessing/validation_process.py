import time, os, random, torch, gc, psutil, tracemalloc, time

from threading import Thread, Event
from pathlib import Path
from .process_logger import get_process_logger
from cliva_fl import models
from cliva_fl.validation import validate_buffer
from cliva_fl.utils import TimeTracker, Logger, partial_class


def validation_process(queue, logger: Logger, monitor_memory=False, **validation_kwargs):
    log = get_process_logger()
    log.info('Starting validator => {}'.format(os.getpid()))
    tf_id = f'validation[pid {os.getpid()}]'
    
    if monitor_memory:
        mem_mo = MemThread(kwargs=dict(pid=os.getpid()))
        mem_mo.start()
        tracemalloc.start()

    time_tracker = TimeTracker()
    time_tracker.start_timeframe(tf_id)
    
    # INITIALIZATION
    time_tracker.start('mp_initialization')
    model_cnf, optimizer_cnf = logger.load_config('model', 'optimizer')
    model = getattr(models, model_cnf['type'])(**model_cnf['params'])
    optimizer = getattr(torch.optim, optimizer_cnf['type'])(model.parameters(), **optimizer_cnf['params'])
    time_tracker.stop('mp_initialization')

    while True:
        time_tracker.start('mp_get_queue')
        buffer = queue.get()
        time_tracker.stop('mp_get_queue')

        if buffer is None: break
        if type(buffer) == str:
            time_tracker.start('mp_read_buffer')
            buffer = logger.get_queue(buffer)
            time_tracker.stop('mp_read_buffer')

        time_tracker.start('mp_validate_buffer')
        validate_buffer(buffer, logger=logger, time_tracker=time_tracker, model=model, optimizer=optimizer, **validation_kwargs)
        time_tracker.stop('mp_validate_buffer')
        logger.log_times(buffer.epoch, time_tracker.total_times_history)
        del buffer
        gc.collect()
        time_tracker.clear()
    
    if monitor_memory:
        current, peak = tracemalloc.get_traced_memory()
        log.info(f"Tracemalloc: {current:0.2f} (current), {peak:0.2f} (peak)")
        tracemalloc.stop()
        mem_mo.stop()
        base, maxi = mem_mo.memory_info()
        logger.log_memory_usage(os.getpid(), base, maxi, maxi-base)
        log.info('MEM_Monitor: {} (base), {} (peak), {} (diff)'.format(base, maxi, maxi - base))
        mem_mo.join()

    log.info('Validator {} exiting...'.format(os.getpid()))
    time_tracker.stop_timeframe(tf_id)
    logger.save_timeframe(tf_id, **time_tracker.get_timeframe(tf_id, format="%Y-%m-%d %H:%M:%S"))

class MemThread(Thread):
 
    def __init__(self, *args, **kwargs):
        super(MemThread, self).__init__(*args, **kwargs)
        self._stopper = Event()
        self._mem_base = 1000000000000
        self._mem_peak = 0.0
 
    # function using _stop function
    def stop(self):
        self._stopper.set()
 
    def stopped(self):
        return self._stopper.isSet()
    
    def memory_info(self):
        return self._mem_base, self._mem_peak

    def run(self):
        pid = self._kwargs['pid']
        p = psutil.Process(pid)
        time.sleep(3)
        while True:
            if self.stopped(): return
            mem = p.memory_info().rss
            if mem > self._mem_peak:
                self._mem_peak = mem
            if mem < self._mem_base:
                self._mem_base = mem
            time.sleep(0.001)
