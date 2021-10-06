import time, os, random
 
# Producer function that places data on the Queue
def training_process(queue, lock, names):
    # Synchronize access to the console
    with lock:
        print('Starting producer => {}'.format(os.getpid()))
         
    # Place our names on the Queue
    for name in names:
        time.sleep(random.randint(0, 10))
        queue.put(name)
 
    # Synchronize access to the console
    with lock:
        print('Producer {} exiting...'.format(os.getpid()))
