from cliva_fl.utils import Logger
from torch.multiprocessing import Process, Queue, Lock, set_start_method, set_sharing_strategy
try: set_start_method('spawn')
except RuntimeError: pass
try: set_sharing_strategy('file_system')
except RuntimeError: pass
from .validation_process import validation_process

def start_validators(num, queue: Queue, logger: Logger, **kwargs):
    lock = Lock()
    consumers = []
    logger.set_lock(lock)

    for _ in range(num):
        consumers.append(Process(target=validation_process, args=(queue, logger), kwargs=kwargs))
    
    for c in consumers: c.start()
    return consumers, lock

def stop_validators(consumers, queue):
    for c in consumers: queue.put_nowait(None)
    for c in consumers: c.join()
    