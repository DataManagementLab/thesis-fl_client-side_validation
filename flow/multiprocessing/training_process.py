import time, os, random
from .process_logger import get_process_logger

# Producer function that places data on the Queue
def training_process(queue, lock, names):
    logger = get_process_logger()

    # Synchronize access to the console
    logger.info('Starting producer => {}'.format(os.getpid()))
         
    # Place our names on the Queue
    for name in names:
        time.sleep(random.randint(0, 10))
        queue.put(name)
 
    # Synchronize access to the console
    logger.info('Producer {} exiting...'.format(os.getpid()))
