import time, os, random

# The consumer function takes data off of the Queue
def validation_process(queue, lock):
    # Synchronize access to the console
    with lock:
        print('Starting consumer => {}'.format(os.getpid()))
     
    # Run indefinitely
    while True:
        time.sleep(random.randint(0, 20))
         
        # If the queue is empty, queue.get() will block until the queue has data
        name = queue.get()

        if name is None:
            with lock:
                print('Consumer {} exiting...'.format(os.getpid()))
            return
         
        # Synchronize access to the console
        with lock:
            print('{} got {}'.format(os.getpid(), name))
