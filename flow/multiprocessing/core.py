from flow.utils import Logger
from torch.multiprocessing import Process, Queue, Lock, set_start_method
try: set_start_method('spawn')
except RuntimeError: pass
from .validation_process import validation_process, consumer_process
from .process_logger import get_process_logger

def start_validators(num, logger: Logger, **kwargs):
    queue = Queue()
    lock = Lock()
    consumers = []
    logger.set_lock(lock)

    for _ in range(num):
        consumers.append(Process(target=validation_process, args=(queue, logger), kwargs=kwargs))
    
    for c in consumers: c.start()
    return consumers, queue, lock

def stop_validators(consumers, queue):
    for c in consumers: queue.put(None)
    for c in consumers: c.join()